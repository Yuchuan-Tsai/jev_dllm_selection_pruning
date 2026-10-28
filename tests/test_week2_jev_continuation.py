"""Finishing a stopped Week 2 Jev live run: exact approval, prefix kept, one launch; no network."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from jev_dllm.evaluation import week3
from jev_dllm.experiments import week2_jev as w, week2_jev_continuation as c
import test_week2_jev as jt


class ContinuationTests(unittest.TestCase):
    setUp, prepare = jt.Week2WorkflowTests.setUp, jt.Week2WorkflowTests.prepare

    def stopped(self):
        plan_dir = self.prepare()
        with self.assertRaises(RuntimeError):
            w.live(self.root, plan_dir, 60, listwise_judge=jt.FakeListwise(), pointwise_judge=jt.FakePointwise(fail_at=3))
        return plan_dir

    def test_continuation_sends_only_the_remaining_calls_and_keeps_the_prefix(self):
        plan_dir = self.stopped()
        info = c.inspect(self.root, plan_dir)
        self.assertEqual((info['accepted_calls'], info['additional_calls'], info['retried_call_index']), (6, 54, 6))
        listwise, pointwise = jt.FakeListwise(), jt.FakePointwise()
        out = c.live(self.root, plan_dir, 54, listwise_judge=listwise, pointwise_judge=pointwise)
        self.assertEqual(listwise.calls + pointwise.calls, 54)
        rows = w.read_rows(out / 'results.jsonl')
        self.assertEqual([r['call_index'] for r in rows], list(range(60)))
        source = (plan_dir / 'live/results.jsonl').read_text()
        self.assertTrue((out / 'results.jsonl').read_text().startswith(source))
        m = json.loads((out / 'manifest.json').read_text())
        self.assertEqual((m['status'], m['successful_calls'], m['continued_from']['billing_unknown_calls']), ('complete', 60, [6]))
        self.assertEqual(m['usage']['listwise']['calls'] + m['usage']['pointwise']['calls'], 60)

    def test_wrong_approval_or_second_launch_is_refused_before_any_call(self):
        plan_dir = self.stopped()
        silent = jt.FakeListwise(), jt.FakePointwise()
        with self.assertRaisesRegex(ValueError, 'exactly the 54'):
            c.live(self.root, plan_dir, 60, listwise_judge=silent[0], pointwise_judge=silent[1])
        self.assertEqual(silent[0].calls + silent[1].calls, 0)
        c.live(self.root, plan_dir, 54, listwise_judge=jt.FakeListwise(), pointwise_judge=jt.FakePointwise())
        with self.assertRaisesRegex(ValueError, 'already has a continuation'):
            c.live(self.root, plan_dir, 54, listwise_judge=jt.FakeListwise(), pointwise_judge=jt.FakePointwise())

    def test_complete_or_tampered_sources_are_refused(self):
        plan_dir = self.prepare()
        w.live(self.root, plan_dir, 60, listwise_judge=jt.FakeListwise(), pointwise_judge=jt.FakePointwise())
        with self.assertRaisesRegex(ValueError, 'failed or interrupted'):
            c.inspect(self.root, plan_dir)

    def test_tampered_prefix_is_refused(self):
        plan_dir = self.stopped()
        results = plan_dir / 'live/results.jsonl'
        rows = results.read_text().splitlines()
        record = json.loads(rows[0])
        record['candidate_id'] = 'g9/c00'
        results.write_text('\n'.join([json.dumps(record)] + rows[1:]) + '\n')
        with self.assertRaises((ValueError, KeyError)):
            c.inspect(self.root, plan_dir)

    def test_analysis_reads_the_continued_output(self):
        plan_dir = self.stopped()
        c.live(self.root, plan_dir, 54, listwise_judge=jt.FakeListwise(), pointwise_judge=jt.FakePointwise())
        scores = week3.jev_scores(self.root, plan_dir, 'gsm8k', self.pool)
        self.assertEqual(scores['output'], c.OUTPUT)
        self.assertEqual(len(scores['correct']), 48)  # 3 questions x 16 pointwise calls.
        self.assertEqual(scores['continued_from']['retried_call_index'], 6)


if __name__ == '__main__':
    unittest.main()
