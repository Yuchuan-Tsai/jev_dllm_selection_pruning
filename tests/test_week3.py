"""Week 3 selectors, metrics, open-judge helpers and the local analysis; no GPU, Math-Verify or API."""
from contextlib import redirect_stdout
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from jev_dllm.evaluation import week2_labels, week3, week3_metrics as metrics
from jev_dllm.judges import open_models
from jev_dllm.selection import week3 as select
import test_week2_labels as label_fixtures
import test_week2_pool as pool_fixtures
from test_week2_pool import FakeBackend

IDS = [f'q/c{i:02}' for i in range(6)]


class SelectorTests(unittest.TestCase):
    def test_highest_breaks_ties_by_pool_order_and_ranks_missing_last(self):
        self.assertEqual(select.highest(IDS, {'q/c02': .9, 'q/c04': .9, 'q/c01': .1}), 'q/c02')
        self.assertEqual(select.highest(IDS, {'q/c05': None}), 'q/c00')  # Nothing scored: first candidate.

    def test_majority_and_weighted_vote(self):
        keys = {'q/c00': 1, 'q/c01': 2, 'q/c02': 2, 'q/c03': 1, 'q/c04': None, 'q/c05': 3}
        self.assertEqual(select.majority(IDS, keys), 'q/c00')  # 2-2 tie: the group seen first.
        self.assertEqual(select.majority(IDS[:3], keys), 'q/c01')
        self.assertEqual(select.majority(IDS, {}), 'q/c00')  # No votes.
        weights = {'q/c01': .4, 'q/c02': .4, 'q/c00': .3, 'q/c03': .3, 'q/c05': .9}
        self.assertEqual(select.weighted_vote(IDS, keys, weights), 'q/c05')  # 0.9 beats 0.8 and 0.6.
        self.assertEqual(select.weighted_vote(IDS[:4], keys, weights), 'q/c01')

    def test_seeded_random_is_deterministic_and_inside_the_subset(self):
        pick = select.seeded_random('q', IDS[:4], 7)
        self.assertEqual(pick, select.seeded_random('q', IDS[:4], 7))
        self.assertIn(pick, IDS[:4])

    def test_prefix_group_keys_match_greedy_grouping_of_each_prefix(self):
        # Complete-link groups over the full pool; restricting to a prefix gives the prefix's own groups.
        keys = select.prefix_group_keys(IDS, [[0, 3], [1, 2, 5], [4]])
        self.assertEqual(select.groups(IDS[:4], keys), [['q/c00', 'q/c03'], ['q/c01', 'q/c02']])
        self.assertEqual(select.majority(IDS, keys), 'q/c01')


class MetricTests(unittest.TestCase):
    def test_auroc_counts_ties_half_and_needs_both_classes(self):
        self.assertEqual(metrics.auroc([.9, .8, .1], [True, False, False]), 1.0)
        self.assertEqual(metrics.auroc([.5, .5], [True, False]), .5)
        self.assertIsNone(metrics.auroc([.5, .6], [True, True]))

    def test_ranking_uses_only_mixed_questions_and_ranks_missing_scores_last(self):
        pools = {'a': ['a1', 'a2'], 'b': ['b1', 'b2']}
        correct = {'a1': True, 'a2': False, 'b1': True, 'b2': True}
        result = metrics.within_question_ranking({'a1': None, 'a2': .2}, pools, correct)
        self.assertEqual((result['mean_within_question_auroc'], result['mixed_questions']), (0.0, 1))

    def test_calibration_brier_and_bins(self):
        result = metrics.calibration({'x': 1.0, 'y': 0.0, 'z': .5}, {'x': True, 'y': False, 'z': True})
        self.assertAlmostEqual(result['brier'], .25 / 3)
        self.assertEqual(sum(b['count'] for b in result['reliability_bins']), 3)

    def test_compare_reports_gap_closed_and_failure_tags(self):
        subsets = {'a': ['a1', 'a2'], 'b': ['b1', 'b2'], 'c': ['c1', 'c2']}
        correct = {'a1': False, 'a2': True, 'b1': True, 'b2': False, 'c1': False, 'c2': False}
        picks = {'seeded_random': {'a': 'a1', 'b': 'b1', 'c': 'c1'}, 'good': {'a': 'a2', 'b': 'b1', 'c': 'c2'}}
        result = metrics.compare(picks, subsets, correct, seed=3)
        good = result['selectors']['good']
        self.assertAlmostEqual(result['oracle_accuracy'], 2 / 3)
        self.assertAlmostEqual(good['gap_closed'], 1.0)
        self.assertEqual(good['failures'], {'correct': 2, 'pool_failure': 1, 'selector_failure': 0})
        self.assertEqual(good['vs_seeded_random']['wins'], 1)


class OpenJudgeHelperTests(unittest.TestCase):
    def test_prompt_keeps_braces_literal(self):
        text = open_models.judge_prompt('Q: {question}\nA: {candidate}', 'f({x})', 'return {1: 2}')
        self.assertEqual(text, 'Q: f({x})\nA: return {1: 2}')

    def test_steps_aggregation_and_yes_probability(self):
        self.assertEqual(open_models.split_steps('a\n\n b \n \n\nc'), ['a', 'b', 'c'])
        self.assertEqual(open_models.split_steps('   '), ['(empty answer)'])
        self.assertEqual(open_models.aggregate([.9, .2, .8], 'min'), .2)
        self.assertAlmostEqual(open_models.aggregate([.5, .5], 'product'), .25)
        self.assertAlmostEqual(open_models.yes_probability(2.0, 2.0), .5)
        self.assertAlmostEqual(open_models.yes_probability(math.log(3), 0.0), .75)
        with self.assertRaises(ValueError):
            open_models.aggregate([.5], 'mean')

    def test_environment_check_reports_every_unpinned_package(self):
        with self.assertRaisesRegex(RuntimeError, 'differs from its pins'):
            open_models.check_environment(ROOT)  # The dependency-free test interpreter lacks the judge stack.

    def test_repository_profile_is_pinned(self):
        profile = open_models.load_profile(ROOT)
        self.assertEqual(profile['prm']['aggregation'], 'min')
        self.assertEqual(set(profile['judge']['prompts']), {'gsm8k', 'math', 'mbpp'})


class FakeScorer:
    def __init__(self, spec, root, task):
        pass

    def score(self, question, candidate):
        return {'p_yes': .9 if '#### 7' in candidate else .2, 'logit_yes': 0.0, 'logit_no': 0.0, 'input_tokens': 5}


class SevenBackend(FakeBackend):
    def generate(self, question, *, candidate_id, seed):
        candidate, cost = super().generate(question, candidate_id=candidate_id, seed=seed)
        text = 'The answer is #### 7' if candidate_id.endswith(('c01', 'c02')) else 'The answer is #### 8'
        return candidate.__class__(**{**candidate.__dict__, 'text': text}), cost


class AnalysisPipelineTests(unittest.TestCase):
    """Fake GSM8K pool → verified copy → labels → fake judge run → verified copy → analysis."""
    write, run_pool = pool_fixtures.Week2PoolTests.write, pool_fixtures.Week2PoolTests.run_pool
    verified_copy = label_fixtures.LabelPipelineTests.verified_copy

    def setUp(self):
        label_fixtures.LabelPipelineTests.setUp(self)
        for name in (open_models.PROFILE, 'requirements/week3-judges.in', 'prompts/week03/qwen_judge_math.txt',
                     'prompts/week03/qwen_judge_code.txt'):
            self.write(name, (ROOT / name).read_text())

    def test_analysis_runs_selectors_with_and_without_a_judge(self):
        pool = self.run_pool(SevenBackend())
        local = self.verified_copy(pool)
        with redirect_stdout(io.StringIO()):
            labels = week2_labels.label(self.root, local)
            judge = open_models.run(self.root, 'judge', pool, scorer_factory=FakeScorer)
        judge_copy = self.verified_copy(judge)
        with redirect_stdout(io.StringIO()):
            out = week3.analyze(self.root, labels, judge_run=judge_copy)
        m = json.loads((out / 'metrics.json').read_text())
        n4 = m['by_subset_size']['4']['selectors']
        self.assertEqual(n4['first']['accuracy'], 0)  # c00 always answers 8.
        self.assertEqual(n4['majority']['accuracy'], 0)  # 7 vs 8 is a 2-2 tie; 8 appears first.
        self.assertEqual(n4['qwen_judge']['accuracy'], 1)
        self.assertEqual(n4['qwen_judge']['gap_closed'], 1)
        self.assertEqual(m['within_question_ranking_full_pool']['qwen_judge']['mean_within_question_auroc'], 1)
        self.assertIn('No Jev live run supplied', m['skipped'])
        self.assertIn('| qwen_judge |', (out / 'summary.md').read_text())
        self.assertEqual(json.loads((judge / 'manifest.json').read_text())['api_calls'], 0)

    def test_limited_or_unverified_judge_runs_are_refused(self):
        pool = self.run_pool(SevenBackend())
        local = self.verified_copy(pool)
        with redirect_stdout(io.StringIO()):
            labels = week2_labels.label(self.root, local)
            limited = open_models.run(self.root, 'judge', pool, limit=1, scorer_factory=FakeScorer)
        with self.assertRaisesRegex(ValueError, 'full-pool'):
            week3.analyze(self.root, labels, judge_run=self.verified_copy(limited))
        unverified = self.root / 'artifacts/downloads/lab' / ('w3judge-gsm8k-20990101T000000Z-' + 'ab' * 4)
        shutil.copytree(limited, unverified)
        with self.assertRaises((ValueError, OSError)):
            week3.analyze(self.root, labels, judge_run=unverified)

    def test_math_voting_requires_the_math_interpreter(self):
        with self.assertRaisesRegex(ValueError, 'Math-Verify'):
            week3.answer_keys(self.root, 'math', [], {}, None)
        keys = week3.answer_keys(self.root, 'gsm8k', [], {}, None)
        self.assertEqual(keys, {})
        self.assertIsNone(week3.answer_keys(self.root, 'mbpp', [], {}, None))


class JevSelectorWiringTests(unittest.TestCase):
    def test_jev_selectors_use_listwise_choice_pointwise_scores_and_votes(self):
        from jev_dllm.schema import Candidate, Question
        q = Question('q', 'gsm8k', 'dev', 'p')
        pools = {'q': [Candidate('q', cid, i, 't', True, False, .5) for i, cid in enumerate(IDS[:4])]}
        keys = {'q/c00': Fraction(1), 'q/c01': Fraction(2), 'q/c02': Fraction(2), 'q/c03': Fraction(1)}
        sources = {'confidence': {v: {} for v in ('mean_probability', 'min_probability', 'mean_log_probability')},
                   'jev': {'listwise': {('q', 4): {'choice': 'q/c03', 'probabilities': {}}},
                           'correct': {'q/c00': .1, 'q/c01': .4, 'q/c02': .4, 'q/c03': .5}}}
        _, picks = week3.selections_for(4, [q], pools, keys, sources, 'gsm8k')
        self.assertEqual(picks['jev_listwise']['q'], 'q/c03')
        self.assertEqual(picks['jev_pointwise']['q'], 'q/c03')  # Highest single score.
        self.assertEqual(picks['jev_weighted_vote']['q'], 'q/c01')  # Group 2 sums 0.8 against 0.6.
        self.assertEqual(picks['majority']['q'], 'q/c00')  # 2-2 tie: the group seen first.


if __name__ == '__main__':
    unittest.main()
