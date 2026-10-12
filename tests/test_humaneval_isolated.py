import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jev_dllm.evaluation import humaneval_isolated as h
from jev_dllm.schema import Candidate, Question

ROOT = Path(__file__).resolve().parents[1]


class HumanEvalIsolationTests(unittest.TestCase):
    def setUp(self):
        self.profile = h.load_profile(ROOT)
        self.reference = {'entry_point': 'increment', 'hidden_tests': 'def check(candidate): assert candidate(2) == 3'}

    def response(self, **overrides):
        d = {'status': 'passed', 'seconds': 0.01, 'python': '3.11.99', 'machine': 'aarch64'}
        d.update(overrides)
        return subprocess.CompletedProcess([], 0, json.dumps(d), '')

    def cleaned(self):
        return subprocess.CompletedProcess([], 1, '', 'Error response from daemon: No such container: fixture')

    def test_restricted_command_has_no_mounts_or_secrets(self):
        args = h.docker_command(self.profile, 'jev-he-fixture', 'worker')
        for flag in ['--network=none', '--read-only', '--user=65534:65534', '--cap-drop=ALL',
                     '--security-opt=no-new-privileges', '--pull=never', '--memory=512m',
                     '--memory-swap=512m', '--pids-limit=64', '--cpus=1']:
            self.assertIn(flag, args)
        self.assertFalse(any(x.startswith(('--volume', '--mount', '--env-file', '--privileged', '-v=')) for x in args))
        self.assertEqual(args[-5:], [self.profile['image'], 'python', '-I', '-c', 'worker'])

    @patch.object(h.subprocess, 'run')
    def test_raw_source_is_preserved_and_references_only_on_stdin(self, run):
        run.side_effect = [self.response(), self.cleaned()]
        source = '```python\ndef increment(x): return x+1\n```'
        result = h.score_one(self.profile, 'worker', source, self.reference)
        self.assertEqual(result['status'], 'passed')
        payload = json.loads(run.call_args_list[0].kwargs['input'])
        self.assertEqual(payload['source'], source)
        self.assertEqual(payload['hidden_tests'], self.reference['hidden_tests'])
        command = run.call_args_list[0].args[0]
        self.assertNotIn(source, command)
        name = command[command.index('--name') + 1]
        self.assertEqual(run.call_args_list[1].args[0], ['docker', 'rm', '-f', name])

    @patch.object(h.subprocess, 'run')
    def test_launch_failure_is_not_a_wrong_answer(self, run):
        run.side_effect = [subprocess.CompletedProcess([], 125, '', 'failure'), self.cleaned()]
        with self.assertRaises(h.InfrastructureError):
            h.score_one(self.profile, 'worker', 'source', self.reference)

    @patch.object(h.subprocess, 'run')
    def test_watchdog_stops_and_cleans_own_container(self, run):
        run.side_effect = [subprocess.TimeoutExpired('docker', 30), self.cleaned()]
        with self.assertRaises(h.InfrastructureError):
            h.score_one(self.profile, 'worker', 'source', self.reference)
        self.assertEqual(run.call_count, 2)

    def test_invalid_worker_response_rejected(self):
        for values in [{'status': 'infrastructure_error'}, {'seconds': float('nan')}, {'python': '3.14.2'}]:
            with self.subTest(values=values), patch.object(h.subprocess, 'run') as run:
                run.side_effect = [self.response(**values), self.cleaned()]
                with self.assertRaises(h.InfrastructureError):
                    h.score_one(self.profile, 'worker', 'source', self.reference)

    @patch.object(h.subprocess, 'run')
    def test_cleanup_failure_is_visible(self, run):
        run.side_effect = [self.response(), subprocess.CompletedProcess([], 1, '', 'daemon stopped')]
        with self.assertRaises(h.InfrastructureError):
            h.score_one(self.profile, 'worker', 'source', self.reference)

    def test_full_denominators_and_oracle(self):
        qs = [Question('fixture/a', 'fixture', 'fixture', 'a'), Question('fixture/b', 'fixture', 'fixture', 'b')]
        pools = {q.question_id: [Candidate(q.question_id, q.question_id+f'/c{i}', i, f'source{i}', True, False)
                                for i in range(2)] for q in qs}
        results = [{'candidate_id': c.candidate_id, 'status': s}
                   for c, s in zip([c for p in pools.values() for c in p],
                                   ['passed', 'syntax_error', 'timeout', 'generation_flag_failure'])]
        metrics = h.summarize(qs, pools, results, self.profile['master_seed'])
        self.assertEqual(metrics['candidate_count'], 4)
        self.assertEqual(metrics['passed_candidates'], 1)
        self.assertEqual(metrics['mean_single_sample_accuracy'], .25)
        self.assertEqual(metrics['empirical_oracle_at_pool_size'], .5)
        self.assertEqual(metrics['first_sample']['correct'], 1)
        self.assertFalse(metrics['jev_evaluated'])
        for bad in [results[:-1], results + [results[0]], [results[0]]*4]:
            with self.assertRaises(ValueError):
                h.summarize(qs, pools, bad, self.profile['master_seed'])

    def test_independent_fixtures_do_not_load_benchmark_data(self):
        fs = h.fixtures()
        self.assertEqual(len({f['name'] for f in fs}), len(fs))
        self.assertTrue({'passed', 'assertion_failure', 'syntax_error', 'runtime_error', 'timeout',
                         'missing_entry_point', 'memory_limit'} <= {f['expected'] for f in fs})
        self.assertTrue(all(f['entry_point'] == 'increment' for f in fs))


if __name__ == '__main__':
    unittest.main()
