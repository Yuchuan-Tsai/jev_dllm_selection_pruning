import dataclasses
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jev_dllm.evaluation.code_extraction import extract_python, fixtures
from jev_dllm.evaluation import humaneval_extracted as h
from jev_dllm.schema import Candidate, Question

ROOT = Path(__file__).resolve().parents[1]


class ExtractionTests(unittest.TestCase):
    def test_independent_fixture_matrix(self):
        for name, raw, mode, source, _ in fixtures():
            with self.subTest(name=name):
                r = extract_python(raw)
                self.assertEqual((r.mode, r.source), (mode, source))
                if source is not None:
                    self.assertEqual(source, raw[r.start:r.end])

    def test_whitespace_and_crlf_body_preserved_exactly(self):
        body = '\r\ndef increment(x):\r\n\treturn x + 1  \r\n'
        raw = 'Some explanation\r\n  ```PYTHON \r\n' + body + '  ``` \r\nMore explanation\r\n'
        r = extract_python(raw)
        self.assertEqual(r.mode, 'fenced')
        self.assertEqual(r.source, body)
        self.assertEqual(r.source.encode(), raw[r.start:r.end].encode())

    def test_valid_python_with_literal_fences_is_not_rewritten(self):
        raw = 'def text():\n    return """\n```python\nprint(1)\n```\n"""\n'
        self.assertEqual(extract_python(raw).source, raw)
        self.assertEqual(extract_python(raw).mode, 'raw')

    def test_unfenced_broken_program_is_not_repaired(self):
        raw = 'def increment(:\n    return x + 1\n'
        self.assertEqual(extract_python(raw).source, raw)
        self.assertEqual(extract_python(raw).mode, 'raw')

    def test_no_reference_or_entry_point_argument(self):
        self.assertEqual(list(inspect.signature(extract_python).parameters), ['text'])

    def test_parser_does_not_execute_plain_python(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / 'must-not-exist'
            raw = f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
            self.assertEqual(extract_python(raw).source, raw)
            self.assertFalse(marker.exists())

    def test_multiple_blocks_rejected_even_if_only_one_python_block(self):
        for raw in ['```text\nExplanation\n```\n```python\ndef x(): return 1\n```',
                    '```python\ndef x(): return 1\n```\n```python\ndef x(): return 2\n```']:
            r = extract_python(raw)
            self.assertEqual(r.mode, 'rejected')
            self.assertIsNone(r.source)

    def test_malformed_closing_and_unsupported_labels_rejected(self):
        for raw in ['````python\ndef x(): return 1\n```',
                    '```python\ndef x(): return 1\n```python',
                    '```{.python}\ndef x(): return 1\n```',
                    '```javascript\ndef x(): return 1\n```']:
            with self.subTest(raw=raw):
                self.assertEqual(extract_python(raw).mode, 'rejected')

    def test_non_string_rejected(self):
        with self.assertRaises(TypeError):
            extract_python(None)

    def test_old_scorer_pins_match_new_policy(self):
        p = h.load_policy(ROOT)
        self.assertEqual(p['analysis_type'], 'post_hoc_reanalysis')
        self.assertEqual(p['extractor'], 'python_single_block_v1')


class ReanalysisTests(unittest.TestCase):
    def candidate(self, text, **kwargs):
        return Candidate('fixture/q', 'fixture/q/c00', 123, text, kwargs.get('completed', True), kwargs.get('truncated', False))

    @patch.object(h.strict, 'score_one')
    def test_worker_receives_only_exact_extracted_source(self, score):
        score.return_value = {'status': 'passed', 'seconds': .01}
        body = 'def increment(x):\n    return x+1\n'
        c = self.candidate('Explanation\n```python\n'+body+'```\nDone.')
        before = dataclasses.asdict(c)
        result = h.score_candidate({}, 'worker', c, {'entry_point': 'increment'})
        self.assertEqual(score.call_args.args[2], body)
        self.assertEqual(dataclasses.asdict(c), before)
        self.assertEqual(result['raw_text_sha256'], h.sha_text(c.text))
        self.assertEqual(result['extracted_text_sha256'], h.sha_text(body))
        self.assertEqual(result['extraction_mode'], 'fenced')

    @patch.object(h.strict, 'score_one')
    def test_rejected_output_remains_a_failure_without_execution(self, score):
        c = self.candidate('```python\ndef increment(x): return x+1\n')
        r = h.score_candidate({}, 'worker', c, {})
        self.assertEqual(r['status'], 'extraction_failure')
        self.assertEqual(r['extraction_reason'], 'ambiguous_or_unclosed_fences')
        score.assert_not_called()

    @patch.object(h.strict, 'score_one')
    def test_generation_flags_still_fail(self, score):
        for c in [self.candidate('def f(): return 1', completed=False),
                  self.candidate('def f(): return 1', truncated=True), self.candidate('')]:
            self.assertEqual(h.score_candidate({}, 'worker', c, {})['status'], 'generation_flag_failure')
        score.assert_not_called()

    @patch.object(h.strict, 'score_one')
    def test_infrastructure_error_propagates_without_candidate_fallback(self, score):
        score.side_effect = h.strict.InfrastructureError('fixture')
        with self.assertRaises(h.strict.InfrastructureError):
            h.score_candidate({}, 'worker', self.candidate('def f(): return 1'), {})

    def pool(self):
        qs = [Question('fixture/q', 'fixture', 'fixture', 'q')]
        cs = [dataclasses.replace(self.candidate('text'), candidate_id=f'fixture/q/c{i:02}') for i in range(2)]
        rows = [{'question_id': c.question_id, 'candidate_id': c.candidate_id,
                 'status': s, 'extraction_mode': m, 'extraction_reason': reason}
                for c,s,m,reason in zip(cs, ['passed','extraction_failure'], ['raw','rejected'], [None,'unclosed'])]
        return qs, {'fixture/q': cs}, rows

    def test_rejected_answers_stay_in_denominator(self):
        qs, pools, rows = self.pool()
        m = h.summarize(qs, pools, rows, 20261008)
        self.assertEqual(m['candidate_count'], 2)
        self.assertEqual(m['passed_candidates'], 1)
        self.assertEqual(m['mean_single_sample_accuracy'], .5)
        self.assertEqual(m['empirical_oracle_at_pool_size'], 1)
        self.assertEqual(m['status_counts']['extraction_failure'], 1)
        self.assertFalse(m['jev_evaluated'])

    def test_missing_duplicate_reordered_unknown_rows_rejected(self):
        qs, pools, rows = self.pool()
        variants = [rows[:1], [rows[0],rows[0]], rows[::-1],
                    [rows[0], {**rows[1], 'status': 'infrastructure_failure'}]]
        for bad in variants:
            with self.assertRaises(ValueError):
                h.summarize(qs, pools, bad, 20261008)

    def test_prior_strict_evidence_and_inputs_must_be_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / 'archive'
            archive.mkdir()
            prior = root / 'strict-evaluation'
            prior.mkdir()
            code = root / 'strict.py'
            code.write_text('# immutable strict scorer\n')
            data = archive / 'candidates.jsonl'
            data.write_text('original candidates\n')
            results = prior / 'candidate-results.jsonl'
            results.write_text('original strict outcomes\n')
            manifest = {'status':'complete', 'source_run':str(archive),
                        'profile':{'version':'humaneval_full_source_strict_v1'},
                        'results_sha256':h.frozen.sha256(results),
                        'implementation_sha256':{'strict.py':h.frozen.sha256(code)},
                        'input_sha256':{str(data):h.frozen.sha256(data)}}
            (prior / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(h.strict, 'FILES', ('strict.py',)):
                h.verify_strict(root, archive, prior)
                for target in (code, data, results):
                    original = target.read_bytes()
                    target.write_bytes(original + b'changed')
                    with self.assertRaises(ValueError):
                        h.verify_strict(root, archive, prior)
                    target.write_bytes(original)

    def test_comparison_detects_gains_losses_and_raw_status_changes(self):
        _, _, rows = self.pool()
        old = [{**rows[0], 'status':'syntax_error'}, {**rows[1], 'status':'passed'}]
        diff = h.compare_strict(old, rows)
        self.assertEqual(diff['pass_gains'], 1)
        self.assertEqual(diff['pass_losses'], 1)
        self.assertEqual(diff['unchanged_source_status_changes'], [rows[0]['candidate_id']])
        with self.assertRaises(ValueError):
            h.compare_strict(old[::-1], rows)


if __name__ == '__main__':
    unittest.main()
