"""Candidate-boundary recovery tests: synthetic pools, no GPU, network or labels."""
from dataclasses import asdict
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack, redirect_stdout

import test_gsm8k500 as fixtures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jev_dllm.experiments import frozen, recovery
from jev_dllm.operations import remote
from jev_dllm.operations import recovery as operations
from jev_dllm.schema import Candidate, CostRecord


class FakeBackend:
    def __init__(self, fail_after=None, failure=KeyboardInterrupt):
        self.calls = []
        self.fail_after = fail_after
        self.failure = failure

    def generate(self, question, *, candidate_id, seed):
        if candidate_id != 'warmup':
            if self.fail_after is not None and len(self.calls) == self.fail_after:
                raise self.failure('synthetic interruption')
            self.calls.append((question.question_id, candidate_id, seed))
        return (Candidate(question.question_id, candidate_id, seed, 'Synthetic only', True, False),
                CostRecord(trajectory_nfe=2, generation_gpu_seconds=0.5,
                           generation_wall_seconds=0.6, peak_memory_bytes=100))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        fixtures.FrozenGSM8KTests.setUp(self)
        for name in recovery.EXTENSION_FILES:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        self.protocol = frozen.check_protocol(self.root, self.config, self.questions)
        self.work = recovery.ordered_work(recovery.load_questions(self.questions), self.config)
        self.source = self.root / 'artifacts/runs/fast-llada-gsm8k500-fixture'
        self.source.mkdir(parents=True)
        self.preserved = 3997  # Interrupted partway through a question's eight answers.
        rows, costs = [], []
        for index, (q, cid, seed) in enumerate(self.work[:self.preserved]):
            # Even truncated/invalid answers must survive; never filter on quality.
            rows.append(asdict(Candidate(q.question_id, cid, seed, 'Synthetic only', index != 0, index == 0)))
            costs.append({'question_id': q.question_id, 'candidate_id': cid,
                          **asdict(CostRecord(trajectory_nfe=2, generation_gpu_seconds=1,
                                             generation_wall_seconds=1.1, peak_memory_bytes=100))})
        self.save_rows(self.source / 'candidates.jsonl', rows)
        self.save_rows(self.source / 'costs.jsonl', costs)
        manifest = {'status': 'interrupted', 'error_type': 'KeyboardInterrupt',
                    'mode': frozen.PHASE, 'protocol': self.protocol, 'backend': 'fast_dllm_v1',
                    'question_count': 500, 'completed_candidates': self.preserved,
                    'questions_sha256': self.config['datasets']['questions_sha256'],
                    'dependencies': {name: 'fixture' for name in ('torch', 'transformers', 'numpy')},
                    'total_seconds_including_load_and_warmup': 12345}
        (self.source / 'manifest.json').write_text(json.dumps(manifest))
        (self.source / 'config.snapshot.json').write_text(json.dumps(self.config))
        (self.source / 'dependencies.txt').write_text('fixture\n')
        self.archive = self.copy_with_receipt(self.source)
        self.plan = self.make_plan(self.archive)

    def save_rows(self, path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def refresh_receipt(self, run):
        snapshot = {'run_id': run.name, 'status': json.loads((run / 'manifest.json').read_text())['status'],
                    'files': {p.name: {'bytes': p.stat().st_size, 'sha256': frozen.sha256(p)}
                              for p in run.iterdir() if p.is_file()}}
        receipts = run.parent / '.receipts'
        receipts.mkdir(exist_ok=True)
        (receipts / (run.name + '.json')).write_text(json.dumps({'snapshot': snapshot,
                                                               'digest': remote.snapshot_digest(snapshot)}))

    def copy_with_receipt(self, source):
        target = self.root / 'artifacts/downloads/lab' / source.name
        shutil.copytree(source, target)
        self.refresh_receipt(target)
        return target

    def make_plan(self, archive):
        return Path(recovery.prepare_plan(self.root, archive)['plan'])

    def execute(self, backend):
        with ExitStack() as stack:
            stack.enter_context(patch.object(recovery, 'runtime_check', return_value={'gpu_name': 'synthetic'}))
            stack.enter_context(patch.object(recovery, 'load_backend', return_value=backend))
            stack.enter_context(patch.object(recovery.platform, 'platform', return_value='synthetic'))
            stack.enter_context(patch.object(recovery.importlib.metadata, 'version', return_value='fixture'))
            stack.enter_context(patch.object(recovery.subprocess, 'run', return_value=SimpleNamespace(stdout='fixture\n')))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return recovery.recover(self.root, self.plan)

    def test_plan_is_idempotent_and_checks_no_labels(self):
        before = self.plan.read_bytes()
        self.make_plan(self.archive)
        self.assertEqual(before, self.plan.read_bytes())
        self.assertFalse((self.root / 'data/private').exists())
        plan = json.loads(before)
        self.assertEqual(plan['next_candidate_id'], self.work[self.preserved][1])
        self.assertEqual(plan['next_seed'], self.work[self.preserved][2])

    def test_only_missing_candidates_generated_prefix_retained_and_frozen_pool_accepted(self):
        original = {p.name: p.read_bytes() for p in self.source.iterdir()}
        backend = FakeBackend()
        run = self.execute(backend)
        self.assertEqual(backend.calls, [(q.question_id, cid, seed) for q, cid, seed in self.work[self.preserved:]])
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.source.iterdir()})
        for name in ('candidates.jsonl', 'costs.jsonl'):
            self.assertTrue((run / name).read_bytes().startswith(original[name]))
        archive = self.copy_with_receipt(run)
        info = recovery.inspect_run(self.root, archive)
        self.assertEqual(info['count'], 4000)
        self.assertEqual(info['manifest']['status'], 'complete')
        self.assertEqual(info['manifest']['newly_generated_candidates'], 3)
        self.assertEqual(info['manifest']['prior_segments'][0]['segment_seconds_including_load_and_warmup'], 12345)
        self.assertIsNone(info['manifest']['recovery']['unrecorded_partial_candidate_gpu_seconds'])
        self.assertEqual(frozen.check_pool(self.root, self.config, self.questions, archive / 'candidates.jsonl'), self.protocol)
        self.assertFalse(json.loads((archive / 'candidates.jsonl').read_text().splitlines()[0])['completed'])

    def test_duplicate_start_is_rejected(self):
        self.execute(FakeBackend())
        backend = FakeBackend()
        with self.assertRaisesRegex(ValueError, 'already attempted'):
            self.execute(backend)
        self.assertEqual(backend.calls, [])

    def test_repeated_interruption_preserves_each_segment_and_missing_seeds(self):
        with self.assertRaises(KeyboardInterrupt):
            self.execute(FakeBackend(fail_after=1))
        claim = json.loads((self.root / 'artifacts/recovery-claims' / (self.source.name + '.json')).read_text())
        first = self.root / 'artifacts/runs' / claim['run_id']
        copied = self.copy_with_receipt(first)
        first_info = recovery.inspect_run(self.root, copied)
        self.assertEqual(first_info['count'], 3998)
        self.assertEqual(first_info['manifest']['status'], 'interrupted')
        self.plan = self.make_plan(copied)
        backend = FakeBackend()
        second = self.execute(backend)
        self.assertEqual(backend.calls, [(q.question_id, cid, seed) for q, cid, seed in self.work[3998:]])
        info = recovery.inspect_run(self.root, self.copy_with_receipt(second))
        self.assertEqual(len(info['manifest']['prior_segments']), 2)
        self.assertEqual(info['count'], 4000)

    def test_failed_backend_retains_outputs_but_cannot_blindly_retry(self):
        with self.assertRaises(RuntimeError):
            self.execute(FakeBackend(fail_after=1, failure=RuntimeError))
        claim = json.loads((self.root / 'artifacts/recovery-claims' / (self.source.name + '.json')).read_text())
        archive = self.copy_with_receipt(self.root / 'artifacts/runs' / claim['run_id'])
        self.assertEqual(recovery.inspect_run(self.root, archive)['count'], 3998)
        with self.assertRaisesRegex(ValueError, 'interrupted, incomplete'):
            self.make_plan(archive)

    def test_source_changed_on_server_stops_before_runtime(self):
        with (self.source / 'candidates.jsonl').open('a') as stream:
            stream.write('\n')
        with patch.object(recovery, 'runtime_check') as hardware:
            with self.assertRaisesRegex(ValueError, 'Server source differs'):
                recovery.recover(self.root, self.plan)
            hardware.assert_not_called()

    def test_local_copy_without_matching_receipt_rejected(self):
        with (self.archive / 'candidates.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'Local copy differs'):
            self.make_plan(self.archive)

    def test_fresh_receipt_does_not_allow_seed_gap_or_misaligned_cost(self):
        path = self.archive / 'candidates.jsonl'
        rows = recovery.jsonl(path)
        original = path.read_bytes()
        for mode in ('seed', 'order', 'gap'):
            changed = [dict(row) for row in rows]
            if mode == 'seed':
                changed[-1]['seed'] += 1
            elif mode == 'order':
                changed[0], changed[1] = changed[1], changed[0]
            else:
                changed.pop(1)
            self.save_rows(path, changed)
            self.refresh_receipt(self.archive)
            with self.assertRaises(ValueError):
                recovery.inspect_run(self.root, self.archive)
        path.write_bytes(original)
        cost_path = self.archive / 'costs.jsonl'
        costs = recovery.jsonl(cost_path)
        costs[-1]['candidate_id'] = 'wrong-id'
        self.save_rows(cost_path, costs)
        self.refresh_receipt(self.archive)
        with self.assertRaisesRegex(ValueError, 'identities/schema'):
            recovery.inspect_run(self.root, self.archive)

    def test_partial_jsonl_is_rejected_not_silently_salvaged(self):
        path = self.archive / 'candidates.jsonl'
        path.write_bytes(path.read_bytes().rstrip(b'\n'))
        self.refresh_receipt(self.archive)
        with self.assertRaisesRegex(ValueError, 'Incomplete last line'):
            self.make_plan(self.archive)

    def test_extension_or_frozen_implementation_changes_rejected(self):
        path = self.root / 'scripts/recovery.py'
        path.write_text(path.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'implementation differs'):
            recovery.check_server_source(self.root, self.plan)
        shutil.copyfile(ROOT / 'scripts/recovery.py', path)
        # The shared Makefile is checked in both layers; use a core-only file here.
        core = self.root / 'src/jev_dllm/reproducibility.py'
        core.write_text(core.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
            recovery.check_server_source(self.root, self.plan)

    def test_changed_plan_refused_even_with_valid_source_receipt(self):
        plan = json.loads(self.plan.read_text())
        plan['next_seed'] += 1
        self.plan.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'plan does not match'):
            recovery.check_server_source(self.root, self.plan)
        with self.assertRaisesRegex(ValueError, 'Existing recovery plan differs'):
            self.make_plan(self.archive)

    def test_recovery_lineage_rejects_changed_prefix_even_with_fresh_receipt(self):
        run = self.execute(FakeBackend())
        archive = self.copy_with_receipt(run)
        path = archive / 'candidates.jsonl'
        rows = recovery.jsonl(path)
        rows[0]['text'] = 'Changed retained answer'
        self.save_rows(path, rows)
        self.refresh_receipt(archive)
        with self.assertRaisesRegex(ValueError, 'source prefix changed'):
            recovery.inspect_run(self.root, archive)

    def test_runtime_version_change_rejected_before_model_load(self):
        with patch.object(recovery, 'runtime_check', return_value={'gpu_name': 'synthetic'}), \
                patch.object(recovery.importlib.metadata, 'version', return_value='wrong'), \
                patch.object(recovery, 'load_backend') as backend:
            with self.assertRaisesRegex(ValueError, 'Runtime versions differ'):
                recovery.recover(self.root, self.plan)
            backend.assert_not_called()

    def test_upload_is_explicit_and_contains_only_validated_metadata(self):
        calls = []
        def command(args, **kwargs):
            calls.append(args)
            return SimpleNamespace(stdout='\0'.join(recovery.EXTENSION_FILES).encode())
        config = {'local_results': str(self.archive.parent), 'remote_root': '/scratch/person/project', 'ssh_host': 'fixture'}
        with patch.object(operations.remote, 'upload') as project, \
                patch.object(operations.subprocess, 'run', side_effect=command):
            result = operations.upload_plan(self.root, self.plan, config)
            project.assert_called_once_with(self.root, config)
        rsync = [args for args in calls if args[0] == 'rsync']
        self.assertEqual(len(rsync), 1)
        self.assertEqual(rsync[0][-2], str(self.plan.resolve()))
        self.assertTrue(result['server_plan'].endswith(self.source.name + '.json'))
        # Extra fields, including secrets, cannot hitch a ride in metadata.
        plan = json.loads(self.plan.read_text())
        plan['unapproved_extra'] = 'never upload this'
        self.plan.write_text(json.dumps(plan))
        with patch.object(operations.subprocess, 'run') as network:
            with self.assertRaisesRegex(ValueError, 'Existing recovery plan differs'):
                operations.upload_plan(self.root, self.plan, config)
            network.assert_not_called()

    def test_cli_audit_and_manual_plan_require_no_gpu(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/recovery.py'), '--root', str(self.root),
                                 'audit', '--run-dir', str(self.archive)], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)['remaining_candidates'], 3)


if __name__ == '__main__':
    unittest.main()
