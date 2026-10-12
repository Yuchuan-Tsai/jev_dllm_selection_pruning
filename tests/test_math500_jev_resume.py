"""Further continuations preserve the full chain and never replay accepted calls."""
from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import Mock, patch

import test_math500_jev_continuation as fixtures
from jev_dllm.experiments import math500_jev_resume as w


class Math500ResumeTests(unittest.TestCase):
    response = fixtures.Math500ContinuationTests.response
    mismatch = fixtures.Math500ContinuationTests.mismatch

    def setUp(self):
        fixtures.Math500ContinuationTests.setUp(self)
        self.first_plan = self.plan
        client = Mock(select=Mock(side_effect=[self.response(self.requests[1]['body']), self.mismatch(self.requests[2]['body'])]))
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            w.previous.execute(self.root, self.first_plan, 2, client)
        self.resume_pins = patch.object(w, 'implementation', return_value={'resume': 'unchanged'}).start()
        self.first_inventory = w.previous.inventory(self.first_plan)
        self.plan = w.prepare(self.root, self.first_plan)

    def execute(self, plan=None):
        client = Mock(select=Mock(side_effect=self.response))
        with redirect_stdout(io.StringIO()):
            out = w.execute(self.root, plan or self.plan, 1, client)
        return out, client

    def audit(self, plan=None):
        with patch.object(w.TypeSafeJudge, 'select', side_effect=AssertionError('network')), redirect_stdout(io.StringIO()):
            out = w.audit_live(self.root, plan or self.plan)
        return out, json.loads((out / 'selection-summary.json').read_text())

    def test_plan_validates_every_ancestor_and_exact_unfinished_suffix(self):
        with patch.object(w.TypeSafeJudge, 'select', side_effect=AssertionError('network')):
            p, source, info = w.inspect(self.root, self.plan)
        self.assertEqual(source, self.first_plan.resolve())
        self.assertEqual((p['reused_selections'], p['prior_attempted_calls'], p['prior_rejected_responses'], p['additional_paid_calls'], p['total_attempts_if_successful']), (2,4,2,1,5))
        self.assertEqual(w.original.read_rows(self.plan / 'planned_requests.jsonl'), self.requests[2:])
        self.assertEqual(len(info['failures']), 2)
        self.assertEqual(w.previous.inventory(self.source), self.original_inventory)
        self.assertEqual(w.previous.inventory(self.first_plan), self.first_inventory)

    def test_sends_only_remaining_body_and_preserves_source_bytes(self):
        out, client = self.execute()
        self.assertEqual([c.args[0] for c in client.select.call_args_list], [self.requests[2]['body']])
        self.assertTrue((out/'selections.jsonl').read_bytes().startswith((self.first_plan/'live/selections.jsonl').read_bytes()))
        self.assertEqual(w.previous.inventory(self.first_plan), self.first_inventory)
        self.assertEqual(w.previous.inventory(self.source), self.original_inventory)

    def test_audit_counts_both_rejections_and_preserves_selected_text(self):
        self.execute()
        out, summary = self.audit()
        self.assertEqual((summary['accepted_selections'], summary['total_attempted_calls'], summary['rejected_responses'], summary['explicit_retries']), (3,5,2,2))
        self.assertEqual(summary['usage_including_rejected'], {'input_tokens':500, 'output_tokens':25})
        self.assertAlmostEqual(summary['request_seconds_sum_including_rejected'], .7)
        self.assertIsNone(summary['accuracy'])
        self.assertIsNone(summary['api_cost_usd'])
        self.assertFalse(summary['correctness_scored'])
        self.assertEqual(len(json.loads((out/'rejected-attempts.json').read_text())),2)
        for r in w.original.read_rows(out/'selected-answers.jsonl'):
            c = next(c for c in self.pools[r['question_id']] if c.candidate_id==r['candidate_id'])
            self.assertEqual(r['text'], c.text)

    def test_wrong_count_or_bool_never_sends(self):
        client = Mock()
        for count in (0, 2, True):
            with self.assertRaises(ValueError):
                w.execute(self.root, self.plan, count, client)
        client.select.assert_not_called()
        self.assertFalse((self.plan/'live').exists())

    def test_new_plan_cannot_replay_already_claimed_source(self):
        _, client = self.execute()
        another = w.prepare(self.root, self.first_plan)
        with self.assertRaises(FileExistsError):
            w.execute(self.root, another, 1, client)
        self.assertEqual(client.select.call_count,1)

    def test_new_mismatch_stops_then_separate_plan_can_continue(self):
        client = Mock(select=Mock(side_effect=self.mismatch(self.requests[2]['body'])))
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            w.execute(self.root, self.plan, 1, client)
        self.assertEqual(client.select.call_count,1)
        self.assertEqual((self.plan/'live/selections.jsonl').read_bytes(), (self.first_plan/'live/selections.jsonl').read_bytes())
        with self.assertRaises(FileExistsError):
            w.execute(self.root, self.plan, 1, client)
        next_plan = w.prepare(self.root, self.plan)
        p,_,_=w.inspect(self.root,next_plan)
        self.assertEqual((p['prior_attempted_calls'],p['prior_rejected_responses'],p['additional_paid_calls']), (5,3,1))
        self.execute(next_plan)
        _, summary=self.audit(next_plan)
        self.assertEqual((summary['accepted_selections'],summary['total_attempted_calls'],summary['rejected_responses']), (3,6,3))
        self.assertEqual(summary['usage_including_rejected'], {'input_tokens':600, 'output_tokens':30})
        self.assertAlmostEqual(summary['request_seconds_sum_including_rejected'],.9)

    def test_interruption_retains_evidence_but_is_not_automatically_retryable(self):
        client = Mock(select=Mock(side_effect=KeyboardInterrupt()))
        with redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            w.execute(self.root,self.plan,1,client)
        m=json.loads((self.plan/'live/manifest.json').read_text())
        self.assertEqual((m['status'],m['successful_calls'],m['total_attempted_calls']), ('interrupted',2,5))
        self.assertEqual((self.plan/'live/selections.jsonl').read_bytes(), (self.first_plan/'live/selections.jsonl').read_bytes())
        with self.assertRaises(ValueError):
            w.prepare(self.root,self.plan)
        self.assertEqual(client.select.call_count,1)

    def test_changed_ancestor_blocks_before_calls(self):
        (self.source/'live/failed_response.json').write_text('{}\n')
        client=Mock()
        with self.assertRaises(ValueError):
            w.execute(self.root,self.plan,1,client)
        client.select.assert_not_called()

    def test_wrong_source_accounting_cannot_be_replanned(self):
        p=self.first_plan/'live/manifest.json'
        m=json.loads(p.read_text()); m['total_attempted_calls']-=1; w.write_json(p,m)
        with self.assertRaisesRegex(ValueError,'accounting'):
            w.prepare(self.root,self.first_plan)

    def test_bad_prefix_and_partial_line_cannot_be_replanned(self):
        p=self.first_plan/'live/selections.jsonl'; before=p.read_bytes()
        rows=w.original.read_rows(p); rows[1]['candidate_id']='wrong-id'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(ValueError):
            w.prepare(self.root,self.first_plan)
        p.write_bytes(before.rstrip(b'\n'))
        with self.assertRaises(ValueError):
            w.prepare(self.root,self.first_plan)

    def test_modified_suffix_cannot_pass_even_with_fresh_hash(self):
        p=self.plan/'planned_requests.jsonl'
        changed=json.loads(p.read_text()); changed['body']['state']['hidden_answer']='forbidden'
        p.write_text(json.dumps(changed)+'\n')
        m=json.loads((self.plan/'plan.json').read_text()); m['planned_requests_sha256']=w.sha256(p)
        w.write_json(self.plan/'plan.json',m)
        with self.assertRaises(ValueError):
            w.inspect(self.root,self.plan)

    def test_changed_implementation_blocks_calls(self):
        self.resume_pins.return_value={'resume':'changed'}
        client=Mock()
        with self.assertRaises(ValueError):
            w.execute(self.root,self.plan,1,client)
        client.select.assert_not_called()

    def test_changed_copied_prior_evidence_blocks_audit(self):
        out,_=self.execute()
        p=out/'prior_manifest.json'; p.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'Copied prior evidence'):
            self.audit()

    def test_dropped_rejected_history_blocks_audit(self):
        out,_=self.execute()
        w.write_json(out/'prior_failures.json',[])
        with self.assertRaisesRegex(ValueError,'history changed'):
            self.audit()

    def test_changed_selected_id_blocks_audit_even_with_fresh_hash(self):
        out,_=self.execute(); p=out/'selections.jsonl'; rows=w.original.read_rows(p)
        rows[-1]['candidate_id']='wrong-id'; p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        m=json.loads((out/'manifest.json').read_text()); m['selections_sha256']=w.sha256(p)
        w.write_json(out/'manifest.json',m)
        with self.assertRaises(ValueError):
            self.audit()

    def test_other_errors_invalid_usage_or_latency_are_not_retryable(self):
        p=self.first_plan/'live/failed_response.json'; before=p.read_text()
        for field, value in (('model','wrong-model'), ('usage',{'input_tokens':-1,'output_tokens':5}), ('latency',float('nan'))):
            with self.subTest(field=field):
                f=json.loads(before); raw=json.loads(f['raw_response_json'])
                if field=='latency': f['request_seconds']=value
                else: raw[field]=value
                f['raw_response_json']=json.dumps(raw); p.write_text(json.dumps(f)+'\n')
                with self.assertRaises(ValueError):
                    w.prepare(self.root,self.first_plan)
        p.write_text(before)

    def test_successful_segment_cannot_be_resumed_again(self):
        self.execute()
        with self.assertRaises(ValueError):
            w.prepare(self.root,self.plan)

    def test_cyclic_source_and_symlink_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError,'Cyclic'):
            w.failed_source(self.root,self.first_plan,frozenset({self.first_plan.resolve()}))
        (self.first_plan/'unexpected-link').symlink_to(self.source/'plan.json')
        with self.assertRaisesRegex(ValueError,'ordinary files'):
            w.prepare(self.root,self.first_plan)


class Math500ResumeProgressTests(unittest.TestCase):
    response = fixtures.Math500ContinuationTests.response
    mismatch = fixtures.Math500ContinuationTests.mismatch

    def test_success_before_another_failure_is_never_sent_again(self):
        from dataclasses import replace
        from jev_dllm.schema import Question
        import test_math500_jev as base
        base.Math500JevTests.setUp(self)
        for name in ('second','third','fourth'):
            self.questions.append(Question(name,'math500','heldout','Compute 1+1.'))
            self.pools[name]=[replace(c,question_id=name,candidate_id=c.candidate_id.replace('task',name)) for c in self.pools['task']]
        patch.object(w.previous,'implementation',return_value={'first':'unchanged'}).start()
        patch.object(w,'implementation',return_value={'resume':'unchanged'}).start()
        source=w.original.prepare(self.root,self.archive)
        requests=w.original.read_rows(source/'requests.jsonl')
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                w.original.execute(self.root,source,4,Mock(select=Mock(side_effect=[self.response(requests[0]['body']),self.mismatch(requests[1]['body'])])))
            first=w.previous.prepare(self.root,source)
            with self.assertRaises(ValueError):
                w.previous.execute(self.root,first,3,Mock(select=Mock(side_effect=[self.response(requests[1]['body']),self.mismatch(requests[2]['body'])])))
            plan=w.prepare(self.root,first)
            with self.assertRaises(ValueError):
                w.execute(self.root,plan,2,Mock(select=Mock(side_effect=[self.response(requests[2]['body']),self.mismatch(requests[3]['body'])])))
            saved=(plan/'live/selections.jsonl').read_bytes()
            next_plan=w.prepare(self.root,plan)
            client=Mock(select=Mock(side_effect=self.response))
            out=w.execute(self.root,next_plan,1,client)
            audit=w.audit_live(self.root,next_plan)
        self.assertEqual([c.args[0] for c in client.select.call_args_list],[requests[3]['body']])
        self.assertTrue((out/'selections.jsonl').read_bytes().startswith(saved))
        summary=json.loads((audit/'selection-summary.json').read_text())
        self.assertEqual((summary['accepted_selections'],summary['total_attempted_calls'],summary['rejected_responses']),(4,7,3))


if __name__=='__main__':
    unittest.main()
