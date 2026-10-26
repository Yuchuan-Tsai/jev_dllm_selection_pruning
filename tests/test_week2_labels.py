"""Week 2 nested subsets, private labels and pool statistics; no Docker, math runtime or API."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from jev_dllm.evaluation import week2_labels
from jev_dllm.experiments import week2_pool
from jev_dllm.operations import remote
from jev_dllm.schema import Candidate, Question
from jev_dllm.selection.subsets import nested_subsets
import test_week2_pool as pool_fixtures
from test_week2_pool import FakeBackend


def pool(qid, n=16, texts=None):
    return [Candidate(qid, f'{qid}/c{i:02}', i, texts[i] if texts else f'text {i}', True, False, 0.5) for i in range(n)]


class SubsetTests(unittest.TestCase):
    def test_subsets_are_nested_ordered_and_deterministic(self):
        ids = [f'q/c{i:02}' for i in range(16)]
        subsets = nested_subsets('q', ids, 7)
        self.assertEqual(subsets, nested_subsets('q', ids, 7))
        self.assertEqual([len(subsets[n]) for n in (2, 4, 8, 16)], [2, 4, 8, 16])
        for small, large in ((2, 4), (4, 8), (8, 16)):
            self.assertTrue(set(subsets[small]) <= set(subsets[large]))
        for members in subsets.values():
            self.assertEqual(members, sorted(members))  # Original pool order, the shared tie-break order.
        self.assertNotEqual(subsets[2], nested_subsets('q', ids, 8)[2])  # Membership depends on the seed.
        with self.assertRaises(ValueError):
            nested_subsets('q', ids[:8], 7)


class StatisticsTests(unittest.TestCase):
    def test_per_size_accuracy_oracle_and_exclusion(self):
        questions = [Question(f'q{i}', 'gsm8k', 'dev', 'p') for i in range(3)]
        pools = {q.question_id: pool(q.question_id, texts=['same'] * 16 if q.question_id == 'q1' else None) for q in questions}
        correct = {c.candidate_id: c.candidate_id.endswith('c00') for c in pools['q0']}
        correct.update({c.candidate_id: True for c in pools['q1']})  # q2 is unlabeled and excluded.
        costs = [{'candidate_id': c.candidate_id, 'trajectory_nfe': 4, 'generation_gpu_seconds': 1.0,
                  'generation_wall_seconds': 2.0} for p in pools.values() for c in p]
        traces = [{'candidate_id': c.candidate_id, 'confidence': {'mean_probability': 0.9 if correct.get(c.candidate_id) else 0.6}}
                  for p in pools.values() for c in p]
        stats = week2_labels.statistics(questions, pools, correct, costs, traces, seed=3, run_note='shared GPU')
        self.assertEqual((stats['question_count'], stats['excluded_questions'], stats['candidate_count']), (2, 1, 32))
        full = stats['by_subset_size']['16']
        self.assertAlmostEqual(full['mean_single_sample_accuracy'], (1 / 16 + 1) / 2)
        self.assertEqual(full['empirical_oracle_at_pool_size'], 1)
        self.assertAlmostEqual(full['mean_unique_text_fraction'], (1 + 1 / 16) / 2)
        q0_two = nested_subsets('q0', [c.candidate_id for c in pools['q0']], 3)[2]
        self.assertEqual(stats['by_subset_size']['2']['empirical_oracle_at_pool_size'], (('q0/c00' in q0_two) + 1) / 2)
        self.assertEqual(stats['generation_cost']['total_nfe'], 128)
        self.assertEqual(stats['generation_cost']['timing_note'], 'shared GPU')
        self.assertEqual(stats['commit_confidence_by_correctness']['correct'], {'count': 17, 'mean': 0.9})


class LabelPipelineTests(unittest.TestCase):
    """End to end on a fake GSM8K pool: verified local copy, labels and metrics."""
    write, run_pool = pool_fixtures.Week2PoolTests.write, pool_fixtures.Week2PoolTests.run_pool

    def setUp(self):
        pool_fixtures.Week2PoolTests.setUp(self)
        refs = [{'question_id': f'gsm8k/train/{i:05}', 'answer': '7'} for i in range(3)]
        self.write('data/private/gsm8k-w2dev-references.json', json.dumps(refs))
        digest = hashlib.sha256((self.root / 'data/private/gsm8k-w2dev-references.json').read_bytes()).hexdigest()
        self.write('data/manifests/week02-dev.json', json.dumps({'files': {'data/private/gsm8k-w2dev-references.json': digest}}))

    def verified_copy(self, run):
        snapshot = remote.run_snapshot(self.root, run.name)
        local = self.root / 'artifacts/downloads/lab'
        shutil.copytree(run, local / run.name)
        (local / '.receipts').mkdir(exist_ok=True)
        (local / '.receipts' / (run.name + '.json')).write_text(json.dumps(
            {'snapshot': snapshot, 'digest': remote.snapshot_digest(snapshot)}))
        return local / run.name

    def test_gsm8k_labels_and_metrics_from_verified_copy(self):
        class SevenBackend(FakeBackend):
            def generate(self, question, *, candidate_id, seed):
                candidate, cost = super().generate(question, candidate_id=candidate_id, seed=seed)
                text = 'The answer is #### 7' if candidate_id.endswith(('c00', 'c01')) else 'The answer is #### 8'
                return candidate.__class__(**{**candidate.__dict__, 'text': text}), cost
        local = self.verified_copy(self.run_pool(SevenBackend()))
        with redirect_stdout(io.StringIO()):
            out = week2_labels.label(self.root, local)
        manifest = json.loads((out / 'manifest.json').read_text())
        self.assertEqual((manifest['status'], manifest['api_calls']), ('complete', 0))
        labels = [json.loads(line) for line in (out / 'evaluation_private/labels.jsonl').read_text().splitlines()]
        self.assertEqual(len(labels), 12)
        self.assertEqual(sum(r['correct'] for r in labels), 6)
        metrics = json.loads((out / 'metrics.json').read_text())
        self.assertEqual(sorted(metrics['by_subset_size']), ['2', '4'])  # Fixture pools hold 4 candidates.
        self.assertAlmostEqual(metrics['by_subset_size']['4']['mean_single_sample_accuracy'], 0.5)
        self.assertEqual(metrics['by_subset_size']['4']['empirical_oracle_at_pool_size'], 1)
        self.assertNotIn('correct_by_candidate', json.dumps(metrics))  # Per-candidate labels stay private.

    def test_unverified_or_tampered_copy_is_refused(self):
        local = self.verified_copy(self.run_pool(FakeBackend()))
        (local / 'candidates.jsonl').write_text((local / 'candidates.jsonl').read_text().replace('answer', 'Answer', 1))
        with self.assertRaises(ValueError):
            week2_labels.label(self.root, local)

    def test_changed_references_are_refused(self):
        local = self.verified_copy(self.run_pool(FakeBackend()))
        self.write('data/private/gsm8k-w2dev-references.json', '[]')
        with self.assertRaisesRegex(ValueError, 'differ from the development manifest'):
            week2_labels.label(self.root, local)


class ScorerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.out)
        (self.out / 'evaluation_private').mkdir()
        self.questions = [Question('mbpp/11', 'mbpp', 'dev', 'p'), Question('mbpp/12', 'mbpp', 'dev', 'p')]
        self.pools = {q.question_id: pool(q.question_id, n=2) for q in self.questions}

    def test_mbpp_excludes_tasks_whose_reference_fails_and_scores_the_rest(self):
        controls = [{'question_id': 'mbpp/11', 'status': 'passed'}, {'question_id': 'mbpp/12', 'status': 'timeout'}]
        scored = []

        def score(profile, worker, candidate, reference):
            scored.append(candidate.candidate_id)
            return {'status': 'passed' if candidate.candidate_id.endswith('c00') else 'assertion_failure', 'extraction_mode': 'raw'}
        with patch.object(week2_labels.strict, 'load_profile', return_value={'workers': 2}), \
             patch.object(week2_labels.humaneval_extracted, 'load_policy', return_value={'version': 'v2'}), \
             patch.object(week2_labels.strict, 'validate_runtime', return_value={'runtime': {'docker_version': 'x'}}), \
             patch.object(week2_labels.humaneval_extracted, 'validate_extraction', return_value={'status': 'passed'}), \
             patch.object(week2_labels.strict, 'reference_controls', return_value=controls), \
             patch.object(week2_labels.humaneval_extracted, 'score_candidate', side_effect=score):
            refs = {q.question_id: {'entry_point': 'f', 'hidden_tests': ''} for q in self.questions}
            rows, details = week2_labels.mbpp_labels(ROOT, self.questions, self.pools, refs, self.out)
        self.assertEqual(sorted(scored), ['mbpp/11/c00', 'mbpp/11/c01'])
        self.assertEqual([r['correct'] for r in rows], [True, False])
        self.assertEqual(details['excluded_questions_failing_reference_tests'], ['mbpp/12'])

    def test_math_stops_when_reference_controls_fail(self):
        questions = [Question('math/train/a/1.json', 'math', 'dev', 'p')]
        replies = {'validate': {'status': 'passed', 'runtime': {}}, 'reference-controls': {'status': 'failed'}}
        with patch.object(week2_labels.math500, 'worker', side_effect=lambda root, python, action, policy, **kw: replies[action]):
            with self.assertRaisesRegex(ValueError, 'reference controls failed'):
                week2_labels.math_labels(ROOT, Path('python'), questions, {questions[0].question_id: pool(questions[0].question_id, 2)},
                                         {questions[0].question_id: {'answer': '1'}}, self.out)



if __name__ == '__main__':
    unittest.main()
