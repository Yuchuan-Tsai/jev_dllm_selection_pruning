from dataclasses import replace
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jev_dllm.config import load, validate
from jev_dllm.data.splits import validate_splits
from jev_dllm.evaluation.answers import extract_final_answer, numeric_value, score_numeric
from jev_dllm.evaluation.base import HumanEvalEvaluator, Math500Evaluator, Reference
from jev_dllm.evaluation.metrics import gap_closed, pool_metrics
from jev_dllm.experiments.smoke import run
from jev_dllm.judges.payloads import final_request, partial_request
from jev_dllm.reproducibility import trajectory_seed
from jev_dllm.schema import Candidate, CostRecord, PartialDraft, Question


class AnswerTests(unittest.TestCase):
    def test_known_correct_numeric_examples(self):
        for text, reference in [(r'Reasoning. \boxed{\frac{1}{2}}', '0.5'),
                                ('#### 1,234', '1234'), ('Final answer: −2', '-2'),
                                ('Final answer: .25', '1/4')]:
            with self.subTest(text=text):
                self.assertTrue(score_numeric(text, reference))

    def test_known_incorrect_numeric_examples(self):
        for text in ('#### 6', '#### -5', '#### 0', '#### 1/0'):
            self.assertFalse(score_numeric(text, '5'))

    def test_does_not_guess_last_number_in_reasoning(self):
        self.assertIsNone(extract_final_answer('First compute 3, then 5.'))
        self.assertFalse(score_numeric('First compute 3, then 5.', '5'))

    def test_last_boxed_and_nested_fraction(self):
        self.assertEqual(extract_final_answer(r'\boxed{3} then \boxed{\frac{1}{2}}'), r'\frac{1}{2}')

    def test_malformed_box_and_empty_answer(self):
        for text in (r'\boxed{', r'\boxed{}', 'Final answer: '):
            self.assertIsNone(extract_final_answer(text))

    def test_ambiguous_or_malformed_numbers_rejected(self):
        for value in ('1,2', '5 or 6', '5 kg', 'NaN', '2+3', '1/0'):
            self.assertIsNone(numeric_value(value))

    def test_symbolic_reference_requires_real_evaluator(self):
        with self.assertRaises(ValueError):
            score_numeric(r'\boxed{x^2}', 'x*x')


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.dev = [Question('dev-1', 'synthetic', 'dev', 'What is 2 + 3?')]
        self.heldout = [Question('test-1', 'synthetic', 'heldout', 'What is 9 + 1?')]

    def test_valid_disjoint_sets(self):
        self.assertEqual(validate_splits(self.dev, self.heldout)['heldout_count'], 1)

    def test_id_leakage_rejected(self):
        self.heldout[0] = replace(self.heldout[0], question_id='dev-1')
        with self.assertRaises(ValueError):
            validate_splits(self.dev, self.heldout)

    def test_same_prompt_under_different_id_and_dataset_rejected(self):
        self.heldout[0] = replace(self.heldout[0], dataset='other', prompt='  WHAT is 2 + 3?  ')
        with self.assertRaises(ValueError):
            validate_splits(self.dev, self.heldout)

    def test_duplicate_and_wrong_split_rejected(self):
        for rows in (self.dev * 2, [replace(self.dev[0], split='heldout')], []):
            with self.assertRaises(ValueError):
                validate_splits(rows, self.heldout)


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.q = Question('q', 'synthetic', 'dev', 'Compute 2+3')
        self.c = Candidate('q', 'c0', 11, 'Final answer: 5', True, False, confidence=.8)

    def test_final_payload_has_only_allowed_fields(self):
        request = final_request(self.q, [self.c])
        self.assertEqual(set(request), {'stage', 'question', 'candidates'})
        self.assertEqual(set(request['candidates'][0]), {'candidate_id', 'text'})
        for forbidden in ('reference', 'hidden_tests', 'correct', 'confidence', 'seed'):
            self.assertNotIn(forbidden, json.dumps(request))

    def test_wrong_question_duplicate_truncated_and_incomplete_rejected(self):
        bad_pools = [[], [self.c, self.c], [replace(self.c, question_id='other')],
                     [replace(self.c, truncated=True)], [replace(self.c, completed=False)]]
        for pool in bad_pools:
            with self.assertRaises(ValueError):
                final_request(self.q, pool)

    def test_partial_budget_progress_only_and_no_local_state(self):
        draft = PartialDraft('q', 'c0', '2 + [MASK]', 2, 8, 100, 'trajectory_nfe', 'raw_masks')
        request = partial_request(self.q, [draft])
        self.assertEqual(request['candidates'][0]['completed_blocks'], 2)
        for forbidden in ('token_ids', 'rng_state', 'kv_cache', 'reference', 'correct'):
            self.assertNotIn(forbidden, json.dumps(request))
        for bad in (replace(draft, completed_blocks=8), replace(draft, remaining_budget=-1),
                    replace(draft, remaining_budget=float('nan'))):
            with self.assertRaises(ValueError):
                partial_request(self.q, [bad])


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config, self.pending = load(ROOT / 'configs/week01/protocol.json')

    def test_draft_valid_but_unresolved(self):
        self.assertFalse(self.config['frozen'])
        self.assertIn('evaluation.math_equivalence_revision', self.pending)
        self.assertNotIn('generator.model_revision', self.pending)
        self.assertNotIn('prompts.jev_partial.sha256', self.pending)

    def test_unresolved_protocol_cannot_be_frozen(self):
        self.config['frozen'] = True
        with self.assertRaises(ValueError):
            validate(self.config)

    def test_invalid_length_precision_model_and_temperature(self):
        for field, value in [('generation_length', 255), ('dtype', 'float16'),
                             ('model_id', 'some-other-model'), ('temperature', 0),
                             ('temperature', float('nan')), ('microbatch_size', True)]:
            candidate = copy.deepcopy(self.config)
            candidate['generator'][field] = value
            with self.assertRaises(ValueError):
                validate(candidate)

    def test_seed_stable_under_pruning_and_reordering(self):
        before = {i: trajectory_seed(42, 'q', i) for i in range(8)}
        self.assertEqual(len(set(before.values())), 8)
        after = {i: trajectory_seed(42, 'q', i) for i in (7, 2)}
        self.assertEqual(after, {7: before[7], 2: before[2]})
        self.assertNotEqual(before[0], trajectory_seed(42, 'other', 0))

    def test_unknown_costs_are_not_zero(self):
        self.assertIsNone(CostRecord().generation_gpu_seconds)
        self.assertIsNone(CostRecord().api_cost_usd)


class MetricsTests(unittest.TestCase):
    def test_weight_questions_equally(self):
        stats = pool_metrics({'a': ['a'], 'b': ['b', 'c', 'd']}, {'a': [True], 'b': [False]*3})
        self.assertEqual(stats['mean_single_sample_accuracy'], .5)
        self.assertEqual(stats['empirical_oracle_at_pool_size'], .5)

    def test_zero_gap_is_undefined(self):
        self.assertIsNone(gap_closed(.5, .5, .5))

    def test_mismatched_labels_fail(self):
        with self.assertRaises(ValueError):
            pool_metrics({'a': ['a']}, {'a': [True, False]})

    def test_real_evaluator_stubs_do_not_return_fake_scores(self):
        candidate = Candidate('q', 'c', 1, 'anything', True, False)
        for evaluator in (Math500Evaluator(), HumanEvalEvaluator()):
            with self.assertRaises(NotImplementedError):
                evaluator.score(candidate, Reference('q', 'answer'))


class SmokeTests(unittest.TestCase):
    def test_end_to_end_artifacts_and_private_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = [run(ROOT, Path(tmp)) for _ in range(2)]
            self.assertNotEqual(first, second)
            manifest = json.loads((first / 'manifest.json').read_text())
            self.assertFalse(manifest['is_benchmark_result'])
            self.assertFalse(manifest['judge_called'])
            self.assertIsNone(manifest['generator'])
            candidates = [json.loads(line) for line in (first / 'candidates.jsonl').read_text().splitlines()]
            self.assertEqual(len(candidates), 16)
            self.assertEqual((first / 'candidates.jsonl').read_text(), (second / 'candidates.jsonl').read_text())
            self.assertNotIn('reference', (first / 'judge_payloads.jsonl').read_text())
            self.assertTrue((first / 'evaluation_private/labels.json').is_file())
            stats = json.loads((first / 'metrics.json').read_text())
            self.assertEqual(stats['mean_single_sample_accuracy'], .5)
            self.assertEqual(stats['empirical_oracle_at_pool_size'], 1)
            costs = json.loads((first / 'costs.json').read_text())
            self.assertIsNone(costs['generation_gpu_seconds'])
            self.assertGreaterEqual(costs['end_to_end_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
