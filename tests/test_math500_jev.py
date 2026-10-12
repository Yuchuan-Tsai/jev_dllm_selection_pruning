"""Selection must preserve public pools, provenance, and paid-call boundaries."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_dllm.experiments import math500_jev as w
from jev_dllm.schema import Question, Candidate


class Math500JevTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root / 'archive'
        self.archive.mkdir()
        (self.archive / 'candidates.jsonl').write_text('public source\n')
        (self.root / 'prompts').mkdir()
        (self.root / w.PROMPT).write_text('Choose the mathematically correct answer.')
        self.questions = [Question('task', 'math500', 'heldout', 'Compute 2 + 3.')]
        self.pools = {'task': [Candidate('task', 'task/c00', 1, 'Reasoning: 2 + 3 = 5. Final: 5.', True, False),
                               Candidate('task', 'task/c01', 2, '', False, True)]}
        self.inputs = patch.object(w, 'public_inputs', return_value=(self.questions, self.pools)).start()
        self.receipt = patch.object(w.collection_recovery, 'receipt_for', return_value={'run_id': 'fixture', 'files': {}}).start()
        self.pins = patch.object(w, 'implementation', return_value={'fixture': 'unchanged'}).start()
        self.addCleanup(patch.stopall)

    def response(self, request):
        ids = list(request['questions']['best_answer']['criteria'])
        raw = {'model': w.MODEL, 'answers': {'best_answer': {'type': 'choice', 'choice': ids[0],
               'probabilities': {cid: float(i == 0) for i, cid in enumerate(ids)}, 'confidence': 0.9}},
               'usage': {'input_tokens': 100, 'output_tokens': 5}}
        return {**w.parse_response(raw, request), 'raw_response': raw, 'request_seconds': 0.1, 'api_cost_usd': None}

    def test_raw_text_and_all_ids_kept_even_when_truncated_or_empty(self):
        reqs = w.request_rows(self.questions, self.pools, 'Select math')
        candidates = {c['candidate_id']: c['text'] for c in reqs[0]['body']['state']['candidates']}
        self.assertEqual(candidates, {c.candidate_id: c.text for c in self.pools['task']})
        self.assertEqual(candidates['task/c01'], '')
        self.assertTrue(self.pools['task'][1].truncated)
        self.assertFalse(self.pools['task'][1].completed)
        self.assertEqual(reqs, w.request_rows(self.questions, self.pools, 'Select math'))

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
        self.questions.append(Question('second', 'math500', 'heldout', 'Compute 1 + 1.'))
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

    def test_selection_audit_preserves_chosen_text_and_reports_no_accuracy(self):
        out = w.prepare(self.root, self.archive)
        w.execute(self.root, out, 1, Mock(select=Mock(side_effect=self.response)))
        with patch.object(w.TypeSafeJudge, 'select', side_effect=AssertionError('network')):
            evaluated = w.audit_live(self.root, out)
        selected = w.read_rows(out / 'live/selections.jsonl')[0]['candidate_id']
        summary = json.loads((evaluated / 'selection-summary.json').read_text())
        saved = w.read_rows(evaluated / 'selected-answers.jsonl')[0]
        self.assertEqual(saved['candidate_id'], selected)
        self.assertEqual(saved['text'], next(c.text for c in self.pools['task'] if c.candidate_id == selected))
        self.assertEqual(summary['accepted_selections'], 1)
        self.assertEqual(summary['usage'], {'input_tokens': 100, 'output_tokens': 5})
        self.assertAlmostEqual(summary['request_seconds_sum'], .1)
        self.assertFalse(summary['correctness_scored'])
        self.assertIsNone(summary['accuracy'])
        self.assertIsNone(summary['api_cost_usd'])

    def test_audit_rejects_mismatched_response_even_with_refreshed_hash(self):
        out = w.prepare(self.root, self.archive)
        w.execute(self.root, out, 1, Mock(select=Mock(side_effect=self.response)))
        rows = w.read_rows(out / 'live/selections.jsonl')
        rows[0]['candidate_id'] = 'wrong-task/c00'
        (out / 'live/selections.jsonl').write_text(json.dumps(rows[0]) + '\n')
        m = json.loads((out / 'live/manifest.json').read_text())
        m['selections_sha256'] = w.frozen.sha256(out / 'live/selections.jsonl')
        w.write_json(out / 'live/manifest.json', m)
        with self.assertRaises(ValueError):
            w.audit_live(self.root, out)

    def test_invalid_latency_is_a_failure_not_a_saved_success(self):
        out = w.prepare(self.root, self.archive)
        result = self.response(w.read_rows(out / 'requests.jsonl')[0]['body'])
        result['request_seconds'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'latency'):
            w.execute(self.root, out, 1, Mock(select=Mock(return_value=result)))
        m = json.loads((out / 'live/manifest.json').read_text())
        self.assertEqual(m['status'], 'failed')
        self.assertEqual(m['successful_calls'], 0)

    def test_explicit_destination_is_idempotent_and_cannot_escape_runs(self):
        destination = self.root / 'artifacts/runs/math500-jev-fixture'
        self.assertEqual(w.prepare(self.root, self.archive, destination), destination)
        before = (destination / 'plan.json').read_bytes()
        w.prepare(self.root, self.archive, destination)
        self.assertEqual(before, (destination / 'plan.json').read_bytes())
        with self.assertRaisesRegex(ValueError, 'directly under'):
            w.prepare(self.root, self.archive, self.root / 'arbitrary-output')
        elsewhere = self.root / 'other-archive'
        elsewhere.mkdir()
        (elsewhere / 'candidates.jsonl').write_bytes((self.archive / 'candidates.jsonl').read_bytes())
        with self.assertRaisesRegex(ValueError, 'different archive'):
            w.prepare(self.root, elsewhere, destination)

    def test_source_snapshot_change_blocks_live_before_any_calls(self):
        out = w.prepare(self.root, self.archive)
        self.receipt.return_value = {'run_id': 'changed', 'files': {}}
        client = Mock()
        with self.assertRaisesRegex(ValueError, 'public inputs changed'):
            w.execute(self.root, out, 1, client)
        client.select.assert_not_called()

    def test_question_candidate_mix_and_duplicates_rejected(self):
        first, second = self.pools['task']
        for changed in ([first, first], [first, replace(second, question_id='other')]):
            with self.assertRaises(ValueError):
                w.request_rows(self.questions, {'task': changed}, 'Choose math')


class Math500ArchiveIntegrationTests(unittest.TestCase):
    """Exercise the real pinned collection/recovery auditors with synthetic data."""
    def setUp(self):
        import test_collection as fixtures
        self.fixtures = fixtures
        for name in ('read', 'save', 'relock'):
            setattr(self, name, getattr(fixtures.CollectionTests, name).__get__(self))
        fixtures.CollectionTests.setUp(self)
        self.project = Path(__file__).resolve().parents[1]
        import shutil
        for name in w.FILES:
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.project / name, dest)

    def test_full_pool_prepares_all_public_requests_without_references(self):
        run = self.fixtures.CollectionTests.execute(self, 'math500', self.fixtures.FakeBackend())
        self.receipt = self.fixtures.CollectionTests.receipt.__get__(self)
        archive = self.fixtures.CollectionTests.archive(self, run)
        # Private files do not exist in this fixture; no scoring dependency is available.
        self.assertFalse((self.root / 'data/private').exists())
        with patch.object(w.TypeSafeJudge, 'select', side_effect=AssertionError('network')):
            out = w.prepare(self.root, archive)
            plan, requests, questions, pools = w.inspect(self.root, out)
        self.assertEqual((plan['question_count'], plan['candidate_count']), (500, 4000))
        self.assertEqual(len(requests), 500)
        self.assertEqual(sum(len(r['body']['state']['candidates']) for r in requests), 4000)
        self.assertTrue(any(not c.completed for pool in pools.values() for c in pool))
        actual = {c['candidate_id']: c['text'] for r in requests for c in r['body']['state']['candidates']}
        self.assertEqual(actual, {c.candidate_id: c.text for pool in pools.values() for c in pool})
        # Turning the manifest into a partial run cannot bypass the actual archive audit.
        m = json.loads((archive / 'manifest.json').read_text())
        m['status'] = 'interrupted'
        (archive / 'manifest.json').write_text(json.dumps(m))
        self.receipt(archive)
        with self.assertRaisesRegex(ValueError, 'complete, verified MATH500'):
            w.public_inputs(self.root, archive)

    def test_recovery_archive_requires_unchanged_prefix_lineage(self):
        import test_collection_recovery as fixture
        helper = fixture.CollectionRecoveryTests(methodName='runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        archive = helper.copy_with_receipt(helper.execute(fixture.FakeBackend()))
        questions, pools = w.public_inputs(helper.root, archive)
        self.assertEqual((len(questions), sum(len(v) for v in pools.values())), (500, 4000))
        path = archive / 'candidates.jsonl'
        rows = w.read_rows(path)
        rows[1]['text'] = 'Changed old answer'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        helper.refresh_receipt(archive)
        with self.assertRaisesRegex(ValueError, 'source prefix changed'):
            w.public_inputs(helper.root, archive)


if __name__ == '__main__':
    unittest.main()
