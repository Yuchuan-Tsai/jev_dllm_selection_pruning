"""Continuation preserves evidence and paid-call limits without relaxing validation."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import test_humaneval_jev as fixtures
from jev_dllm.experiments import humaneval_jev as original
from jev_dllm.experiments import humaneval_jev_continuation as w
from jev_dllm.schema import Question


class HumanEvalContinuationTests(unittest.TestCase):
    def response(self, request):
        return fixtures.HumanEvalJevTests.response(self, request)

    def mismatch(self, request):
        raw = self.response(request)['raw_response']
        ids = list(request['questions']['best_answer']['criteria'])
        raw['answers']['best_answer']['choice'] = ids[1]
        error = ValueError('Jev choice is inconsistent with its probabilities')
        error.raw_response, error.request_seconds = raw, .2
        return error

    def setUp(self):
        fixtures.HumanEvalJevTests.setUp(self)
        for name in ('second', 'third'):
            self.questions.append(Question(name, 'humaneval', 'heldout', 'def increment(x): Return x+1'))
            self.pools[name] = [replace(c, question_id=name, candidate_id=c.candidate_id.replace('task',name)) for c in self.pools['task']]
        self.ext_pins = patch.object(w, 'implementation', return_value={'extension': 'unchanged'}).start()
        self.source = original.prepare(self.root, self.archive)
        self.requests = original.read_rows(self.source / 'requests.jsonl')
        client = Mock(select=Mock(side_effect=[self.response(self.requests[0]['body']), self.mismatch(self.requests[1]['body'])]))
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            original.execute(self.root, self.source, 3, client)
        self.original_inventory = w.inventory(self.source)
        self.plan = w.prepare(self.root, self.source)

    def execute(self):
        client = Mock(select=Mock(side_effect=self.response))
        with redirect_stdout(io.StringIO()):
            out = w.execute(self.root, self.plan, 2, client)
        return out, client

    def test_plan_has_exact_suffix_and_no_private_reads_or_calls(self):
        self.assertEqual(original.read_rows(self.plan / 'planned_requests.jsonl'), self.requests[1:])
        with patch.object(w.TypeSafeJudge, 'select', side_effect=AssertionError('network')), patch.object(original, 'score_labels', side_effect=AssertionError('private labels')):
            p, _, _ = w.inspect(self.root, self.plan)
        self.assertEqual((p['reused_selections'],p['prior_attempted_calls'],p['additional_paid_calls'],p['unattempted_questions'],p['total_attempts_if_successful']), (1,2,2,1,4))

    def test_success_preserves_prefix_and_sends_only_exact_suffix(self):
        out, client = self.execute()
        self.assertEqual([call.args[0] for call in client.select.call_args_list], [r['body'] for r in self.requests[1:]])
        self.assertTrue((out / 'selections.jsonl').read_bytes().startswith((self.source / 'live/selections.jsonl').read_bytes()))
        self.assertEqual(w.inventory(self.source), self.original_inventory)
        m = json.loads((out / 'manifest.json').read_text())
        self.assertEqual((m['successful_calls'],m['attempted_calls'],m['total_attempted_calls']), (3,2,4))

    def test_wrong_approval_never_calls(self):
        client = Mock()
        with self.assertRaises(ValueError):
            w.execute(self.root, self.plan, 3, client)
        client.select.assert_not_called()
        self.assertFalse((self.plan / 'live').exists())

    def test_repreparing_cannot_duplicate_the_continuation(self):
        _, client = self.execute()
        another = w.prepare(self.root, self.source)
        with self.assertRaises(FileExistsError):
            w.execute(self.root, another, 2, client)
        self.assertEqual(client.select.call_count, 2)

    def test_changed_original_evidence_stops_before_payment(self):
        (self.source / 'live/selections.jsonl').write_text('changed\n')
        client = Mock()
        with self.assertRaises(ValueError):
            w.execute(self.root, self.plan, 2, client)
        client.select.assert_not_called()

    def test_changed_suffix_rejected_even_if_hash_updated(self):
        rows = self.requests[1:][::-1]
        (self.plan / 'planned_requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        p = json.loads((self.plan / 'plan.json').read_text())
        p['planned_requests_sha256'] = w.sha256(self.plan / 'planned_requests.jsonl')
        w.write_json(self.plan / 'plan.json',p)
        with self.assertRaises(ValueError):
            w.inspect(self.root,self.plan)

    def test_original_prefix_must_validate_before_plan_creation(self):
        rows = original.read_rows(self.source / 'live/selections.jsonl')
        rows[0]['candidate_id'] = 'second/c00'
        (self.source / 'live/selections.jsonl').write_text(json.dumps(rows[0])+'\n')
        with self.assertRaises(ValueError):
            w.prepare(self.root,self.source)

    def test_other_failure_types_are_not_retried(self):
        p = self.source / 'live/failed_response.json'
        failure = json.loads(p.read_text())
        raw = json.loads(failure['raw_response_json'])
        raw['model'] = 'unexpected-model'
        failure['raw_response_json'] = json.dumps(raw)
        w.write_json(p, failure)
        with self.assertRaisesRegex(ValueError,'Only a Choice/probability mismatch'):
            w.prepare(self.root,self.source)

    def test_another_failure_stops_without_automatic_retry(self):
        error = self.mismatch(self.requests[1]['body'])
        client = Mock(select=Mock(side_effect=error))
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            w.execute(self.root,self.plan,2,client)
        self.assertEqual(client.select.call_count,1)
        live = self.plan / 'live'
        m = json.loads((live/'manifest.json').read_text())
        self.assertEqual((m['status'],m['attempted_calls'],m['total_attempted_calls']), ('failed',1,3))
        self.assertEqual((live/'selections.jsonl').read_bytes(), (self.source/'live/selections.jsonl').read_bytes())
        self.assertTrue((live/'failed_response.json').exists())
        with self.assertRaises(FileExistsError):
            w.execute(self.root,self.plan,2,client)
        self.assertEqual(w.inventory(self.source),self.original_inventory)

    def test_interruption_retains_new_successes(self):
        client = Mock(select=Mock(side_effect=[self.response(self.requests[1]['body']), KeyboardInterrupt()]))
        with redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            w.execute(self.root,self.plan,2,client)
        m = json.loads((self.plan/'live/manifest.json').read_text())
        self.assertEqual((m['status'],m['successful_calls'],m['total_attempted_calls']), ('interrupted',2,4))
        self.assertEqual(len(original.read_rows(self.plan/'live/selections.jsonl')),2)

    def test_evaluation_counts_all_attempts_including_rejected_usage(self):
        out,_ = self.execute()
        score=self.root/'scores';score.mkdir()
        for name in ('manifest.json','candidate-results.jsonl'):
            (score/name).write_text('{}\n')
        labels={c.candidate_id:True for pool in self.pools.values() for c in pool}
        with patch.object(original,'score_labels',return_value=(labels,{})), patch.object(w.TypeSafeJudge,'select',side_effect=AssertionError('network')), redirect_stdout(io.StringIO()):
            result=w.evaluate(self.root,self.plan,score)
        metrics=json.loads((result/'metrics.json').read_text())
        self.assertEqual(metrics['jev']['total'],3)
        self.assertEqual(metrics['total_attempted_calls'],4)
        self.assertEqual(metrics['usage_including_rejected'],{'input_tokens':400,'output_tokens':20})
        self.assertAlmostEqual(metrics['request_seconds_sum_including_rejected'],.5)
        self.assertIsNone(metrics['api_cost_usd'])

    def test_evaluation_rejects_changed_selection_before_private_labels(self):
        out,_=self.execute()
        rows=original.read_rows(out/'selections.jsonl')
        rows[-1]['candidate_id']='wrong-id'
        (out/'selections.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        m=json.loads((out/'manifest.json').read_text())
        m['selections_sha256']=w.sha256(out/'selections.jsonl')
        w.write_json(out/'manifest.json',m)
        with patch.object(original,'score_labels') as labels, self.assertRaises(ValueError):
            w.evaluate(self.root,self.plan,self.root/'scores')
        labels.assert_not_called()


if __name__ == '__main__':
    unittest.main()
