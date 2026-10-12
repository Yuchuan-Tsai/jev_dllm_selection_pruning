"""Selection must preserve public pools, provenance, and paid-call boundaries."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.experiments import humaneval_jev as w
from jev_dllm.schema import Question, Candidate


class HumanEvalJevTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root / 'archive'
        self.archive.mkdir()
        (self.archive / 'candidates.jsonl').write_text('public source\n')
        (self.root / 'prompts').mkdir()
        (self.root / w.PROMPT).write_text('Choose the correct public implementation.')
        self.questions = [Question('task', 'humaneval', 'heldout', 'def add(x): Return x + 1')]
        self.pools = {'task': [Candidate('task', 'task/c00', 1, 'Prose\n```python\ndef add(x):\n    return x + 1\n```\n', True, False),
                               Candidate('task', 'task/c01', 2, '```python\npass\n```\n```python\npass\n```', True, False)]}
        self.inputs = patch.object(w, 'public_inputs', return_value=(self.questions, self.pools)).start()
        self.pins = patch.object(w, 'implementation', return_value={'fixture': 'unchanged'}).start()
        self.addCleanup(patch.stopall)

    def response(self, request):
        ids = list(request['questions']['best_answer']['criteria'])
        raw = {'model': w.MODEL, 'answers': {'best_answer': {'type': 'choice', 'choice': ids[0],
               'probabilities': {cid: float(i == 0) for i, cid in enumerate(ids)}, 'confidence': 0.9}},
               'usage': {'input_tokens': 100, 'output_tokens': 5}}
        return {**w.parse_response(raw, request), 'raw_response': raw, 'request_seconds': 0.1, 'api_cost_usd': None}

    def test_cleaning_keeps_exact_source_and_all_ids(self):
        reqs, transforms = w.request_rows(self.questions, self.pools, 'Select code')
        candidates = {c['candidate_id']: c['text'] for c in reqs[0]['body']['state']['candidates']}
        self.assertEqual(candidates['task/c00'], 'def add(x):\n    return x + 1\n')
        self.assertEqual(candidates['task/c01'], self.pools['task'][1].text)
        self.assertEqual(len(transforms), 2)
        self.assertIsNone(transforms[1]['source_span'])
        self.assertEqual(reqs, w.request_rows(self.questions, self.pools, 'Select code')[0])

    def test_prepare_never_reads_private_files_or_calls_api(self):
        with patch.object(w.TypeSafeJudge, 'select', side_effect=AssertionError('network')):
            private = self.root / 'data/private'
            private.mkdir(parents=True)
            (private / 'labels.json').write_text('PRIVATE_SENTINEL')
            original = Path.open
            def restricted(path, *args, **kwargs):
                if path.is_relative_to(private):
                    raise AssertionError('Read private data during preparation')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'open', restricted):
                out = w.prepare(self.root, self.archive)
                w.inspect(self.root, out)
            self.assertNotIn('PRIVATE_SENTINEL', (out / 'requests.jsonl').read_text())
            state = w.read_rows(out / 'requests.jsonl')[0]['body']['state']
            self.assertEqual(set(state), {'stage', 'question', 'candidates'})
            self.assertTrue(all(set(c) == {'candidate_id', 'text'} for c in state['candidates']))

    def test_tampered_request_rejected_even_with_updated_hash(self):
        out = w.prepare(self.root, self.archive)
        rows = w.read_rows(out / 'requests.jsonl')
        rows[0]['body']['state']['hidden_tests'] = 'forbidden'
        (out / 'requests.jsonl').write_text(json.dumps(rows[0]) + '\n')
        p = json.loads((out / 'plan.json').read_text())
        p['requests_sha256'] = w.frozen.sha256(out / 'requests.jsonl')
        w.write_json(out / 'plan.json', p)
        client = Mock()
        with self.assertRaises(ValueError):
            w.execute(self.root, out, 1, client)
        client.select.assert_not_called()

    def test_changed_implementation_rejected(self):
        out = w.prepare(self.root, self.archive)
        self.pins.return_value = {'fixture': 'changed'}
        with self.assertRaises(ValueError):
            w.inspect(self.root, out)

    def test_wrong_approval_count_sends_nothing(self):
        out = w.prepare(self.root, self.archive)
        client = Mock()
        with self.assertRaises(ValueError):
            w.execute(self.root, out, 2, client)
        client.select.assert_not_called()
        self.assertFalse((out / 'live').exists())

    def test_duplicate_plan_cannot_replay_paid_batch(self):
        out = w.prepare(self.root, self.archive)
        client = Mock(select=Mock(side_effect=self.response))
        w.execute(self.root, out, 1, client)
        another = w.prepare(self.root, self.archive)
        with self.assertRaises(FileExistsError):
            w.execute(self.root, another, 1, client)
        self.assertEqual(client.select.call_count, 1)

    def test_failure_stops_preserves_prefix_and_forbids_retry(self):
        self.questions.append(Question('second', 'humaneval', 'heldout', 'def other(): Return 1'))
        self.pools['second'] = [replace(c, question_id='second', candidate_id=c.candidate_id.replace('task', 'second')) for c in self.pools['task']]
        out = w.prepare(self.root, self.archive)
        first = self.response(w.read_rows(out / 'requests.jsonl')[0]['body'])
        error = ValueError('malformed response')
        error.raw_response = {'model': 'invalid', 'bad_probability': float('nan')}
        error.request_seconds = .2
        client = Mock(select=Mock(side_effect=[first, error]))
        with self.assertRaises(ValueError):
            w.execute(self.root, out, 2, client)
        m = json.loads((out / 'live/manifest.json').read_text())
        self.assertEqual((m['status'], m['attempted_calls'], m['successful_calls']), ('failed', 2, 1))
        self.assertEqual(len(w.read_rows(out / 'live/selections.jsonl')), 1)
        self.assertTrue((out / 'live/failed_response.json').exists())
        failure = json.loads((out / 'live/failed_response.json').read_text())
        self.assertIn('NaN', failure['raw_response_json'])
        with self.assertRaises(FileExistsError):
            w.execute(self.root, out, 2, client)
        self.assertEqual(client.select.call_count, 2)

    def test_evaluate_uses_selected_original_id(self):
        out = w.prepare(self.root, self.archive)
        w.execute(self.root, out, 1, Mock(select=Mock(side_effect=self.response)))
        score = self.root / 'scores'
        score.mkdir()
        for name in ('manifest.json', 'candidate-results.jsonl'):
            (score / name).write_text('{}\n')
        labels = {'task/c00': True, 'task/c01': False}
        with patch.object(w, 'score_labels', return_value=(labels, {})), patch.object(w.TypeSafeJudge, 'select', side_effect=AssertionError('network')):
            evaluated = w.evaluate(self.root, out, score)
        selected = w.read_rows(out / 'live/selections.jsonl')[0]['candidate_id']
        metrics = json.loads((evaluated / 'metrics.json').read_text())
        self.assertEqual(metrics['jev']['correct'], int(labels[selected]))
        self.assertEqual(metrics['jev']['total'], 1)
        self.assertIsNone(metrics['api_cost_usd'])

    def test_evaluate_rejects_mismatched_response_before_labels(self):
        out = w.prepare(self.root, self.archive)
        w.execute(self.root, out, 1, Mock(select=Mock(side_effect=self.response)))
        rows = w.read_rows(out / 'live/selections.jsonl')
        rows[0]['candidate_id'] = 'wrong-task/c00'
        (out / 'live/selections.jsonl').write_text(json.dumps(rows[0]) + '\n')
        m = json.loads((out / 'live/manifest.json').read_text())
        m['selections_sha256'] = w.frozen.sha256(out / 'live/selections.jsonl')
        w.write_json(out / 'live/manifest.json', m)
        with patch.object(w, 'score_labels') as labels, self.assertRaises(ValueError):
            w.evaluate(self.root, out, self.root / 'scores')
        labels.assert_not_called()


if __name__ == '__main__':
    unittest.main()
