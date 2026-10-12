"""Offline scoring refuses mismatched pools and keeps voting reference-free."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from jev_dllm.evaluation import math500 as w
from jev_dllm.schema import Candidate, Question


class Math500ScoringTests(unittest.TestCase):
    def setUp(self):
        self.policy = {'seed': 20261008, 'version': 'test', 'question_timeout_seconds': 10}
        self.questions = [Question(f'q{i}', 'math500', 'fixture', 'question') for i in range(2)]
        self.pools = {q.question_id: [Candidate(q.question_id, f'{q.question_id}/c{j:02}', j, str(j), True, False) for j in range(3)] for q in self.questions}
        self.info = {'questions': self.questions, 'pools': self.pools, 'failures': [], 'total_attempts': 2,
                     'accepted': [{'question_id': q.question_id, 'candidate_id': f'{q.question_id}/c01',
                                   'usage': {'input_tokens': 10, 'output_tokens': 2}, 'request_seconds': .1} for q in self.questions]}
        self.records = [{'question_id': q.question_id, 'candidates': [
            {'candidate_id': c.candidate_id, 'correct': j == i, 'extraction_status': 'symbolic', 'verification_status': 'checked'}
            for j, c in enumerate(self.pools[q.question_id])],
            'public_selection': {'majority_candidate_id': f'{q.question_id}/c00', 'groups': [[0], [1], [2]]}}
            for i, q in enumerate(self.questions)]

    def test_same_pool_metrics_and_question_unit(self):
        metrics, details = w.summarize(self.info, self.records, self.policy)
        self.assertEqual(metrics['candidate_count'], 6)
        self.assertEqual(metrics['passed_candidates'], 2)
        self.assertEqual(metrics['correct_counts']['jev'], 1)
        self.assertEqual(metrics['empirical_oracle_at_pool_size'], 1)
        self.assertEqual(metrics['unit_of_analysis'], 'question')
        self.assertEqual(len(details), 2)
        self.assertEqual(metrics['jev_execution']['new_api_calls'], 0)

    def test_rejections_counted_even_without_accepted_label(self):
        self.info['failures'] = [{'response': {'raw_response_json': json.dumps({'usage': {'input_tokens': 5, 'output_tokens': 3}}), 'request_seconds': .2}}]
        self.info['total_attempts'] = 3
        metrics, _ = w.summarize(self.info, self.records, self.policy)
        cost = metrics['jev_execution']
        self.assertEqual(cost['usage_including_rejected'], {'input_tokens': 25, 'output_tokens': 7})
        self.assertEqual(cost['total_attempted_calls'], 3)
        self.assertEqual(cost['rejected_responses'], 1)
        self.assertAlmostEqual(cost['request_seconds_sum_including_rejected'], .4)
        self.assertIsNone(cost['api_cost_usd'])

    def test_missing_or_duplicate_selections_rejected(self):
        for accepted in (self.info['accepted'][:1], [self.info['accepted'][0]] * 2):
            with self.assertRaises(ValueError):
                w.summarize({**self.info, 'accepted': accepted}, self.records, self.policy)

    def test_reordered_or_missing_candidate_rejected(self):
        for records in (self.records[::-1], self.records[:1]):
            with self.assertRaises(ValueError):
                w.summarize(self.info, records, self.policy)
        self.records[0]['candidates'].pop()
        with self.assertRaises(ValueError):
            w.summarize(self.info, self.records, self.policy)

    def test_non_boolean_label_rejected(self):
        self.records[0]['candidates'][0]['correct'] = 1
        with self.assertRaises(ValueError):
            w.summarize(self.info, self.records, self.policy)

    def test_out_of_pool_vote_rejected(self):
        self.records[0]['public_selection']['majority_candidate_id'] = 'q0/absent'
        with self.assertRaises(ValueError):
            w.summarize(self.info, self.records, self.policy)

    def test_worker_does_not_inherit_credentials(self):
        with patch.dict('os.environ', {'TYPESAFE_API_KEY': 'fixture-only', 'HF_TOKEN': 'fixture-only'}), patch.object(w.subprocess, 'run', return_value=Mock(returncode=0, stdout='{}')) as run:
            w.worker(Path('/project'), Path('/python'), 'runtime', self.policy)
        self.assertNotIn('TYPESAFE_API_KEY', run.call_args.kwargs['env'])
        self.assertNotIn('HF_TOKEN', run.call_args.kwargs['env'])
        self.assertEqual(run.call_args.kwargs['timeout'], 10)

    def test_worker_wall_timeout_is_not_an_incorrect_answer(self):
        with patch.object(w.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker', 10)):
            with self.assertRaisesRegex(RuntimeError, 'no score was invented'):
                w.worker(Path('/project'), Path('/python'), 'question', self.policy)

    def test_changed_policy_refused_before_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            w.write_json(root / 'plan.json', {'implementation_sha256': {'x': 'old'}, 'policy': {}})
            with patch.object(w, 'implementation', return_value={'x': 'changed'}), patch.object(w, 'worker') as worker:
                with self.assertRaisesRegex(ValueError, 'implementation'):
                    w.inspect(root, root/'python', root)
                worker.assert_not_called()

    def test_symlinked_implementation_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'a').write_text('fixture')
            (root/'b').symlink_to(root/'a')
            with patch.object(w, 'FILES', ('b',)), self.assertRaises(ValueError):
                w.implementation(root)

    def test_mismatched_generation_pool_refused(self):
        with patch.object(w.selection, 'inspect', return_value=({}, None, {})), patch.object(w.selection, '_check_segment', return_value=self.info), patch.object(w.selection.original, 'public_inputs', return_value=(self.questions, {})):
            with self.assertRaisesRegex(ValueError, 'exactly the same pool'):
                w.context(Path('/project'), Path('/archive'), Path('/jev'))


class Math500VotingTests(unittest.TestCase):
    def setUp(self):
        # Core tests do not install/import the optional math runtime. Real engine
        # semantics are checked separately by the independent fixture command.
        spec = importlib.util.spec_from_file_location('test_math_worker', Path(__file__).parents[1]/w.WORKER)
        self.worker = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'math_verify': Mock()}):
            spec.loader.exec_module(self.worker)
        self.candidates = [{'candidate_id': str(i), 'text': str(i), 'completed': True, 'truncated': False} for i in range(3)]

    def test_complete_link_does_not_chain_nontransitive_matches(self):
        with patch.object(self.worker, 'extract', side_effect=lambda t, p: ([int(t)], 'symbolic')), patch.object(self.worker, 'equivalent', side_effect=lambda a, b, p: (abs(a[0]-b[0]) <= 1, 'checked')):
            _, vote = self.worker.public_analysis(self.candidates, {})
        self.assertEqual(vote['groups'], [[0, 1], [2]])

    def test_asymmetric_equivalence_does_not_join_votes(self):
        with patch.object(self.worker, 'extract', side_effect=lambda t, p: ([int(t)], 'symbolic')), patch.object(self.worker, 'equivalent', side_effect=lambda a, b, p: (a[0] <= b[0], 'checked')):
            _, vote = self.worker.public_analysis(self.candidates, {})
        self.assertEqual(vote['groups'], [[0], [1], [2]])
        self.assertEqual(vote['majority_candidate_id'], '0')

    def test_incomplete_truncated_and_empty_do_not_vote(self):
        self.candidates[0]['completed'] = False
        self.candidates[1]['truncated'] = True
        self.candidates[2]['text'] = ''
        with patch.object(self.worker, 'extract') as extract:
            _, vote = self.worker.public_analysis(self.candidates, {})
        extract.assert_not_called()
        self.assertEqual(vote['groups'], [])
        self.assertEqual(vote['majority_support'], 0)

    def test_reference_is_read_only_after_public_voting(self):
        events = []
        class Request(dict):
            def __getitem__(self, key):
                if key == 'reference': events.append('reference')
                return super().__getitem__(key)
        request = Request(candidates=self.candidates, reference='private', question_id='fixture')
        public = {'extraction_statuses': [], 'parsed_values': []}
        with patch.object(self.worker, 'public_analysis', side_effect=lambda c, p: (events.append('vote') or [], public)), patch.object(self.worker, 'extract', return_value=([1], 'symbolic')):
            self.worker.grade_question(request, {})
        self.assertEqual(events, ['vote', 'reference'])
