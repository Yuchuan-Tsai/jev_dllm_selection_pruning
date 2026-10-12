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

import test_collection as fixtures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jev_dllm.experiments import frozen, collection, collection_recovery as recovery
from jev_dllm.operations import remote
from jev_dllm.operations import collection_recovery as operations
from jev_dllm.schema import Candidate, CostRecord


class FakeBackend:
    def __init__(self, fail_after=None, failure=KeyboardInterrupt):
        self.calls = []
        self.warmups = []
        self.fail_after = fail_after
        self.failure = failure

    def generate(self, question, *, candidate_id, seed):
        if candidate_id == 'warmup':
            self.warmups.append(question.question_id)
        else:
            if self.fail_after is not None and len(self.calls) == self.fail_after:
                raise self.failure('synthetic interruption')
            self.calls.append((question.question_id, candidate_id, seed))
        return (Candidate(question.question_id, candidate_id, seed, 'Synthetic only', True, False),
                CostRecord(trajectory_nfe=2, generation_gpu_seconds=0.5,
                           generation_wall_seconds=0.6, peak_memory_bytes=100))


class CollectionRecoveryTests(unittest.TestCase):
    read = fixtures.CollectionTests.read
    save = fixtures.CollectionTests.save
    relock = fixtures.CollectionTests.relock

    def setUp(self):
        fixtures.CollectionTests.setUp(self)
        for name in recovery.EXTENSION_FILES:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        self.spec, self.questions = collection.check(self.root, 'math500')
        self.config = self.spec['backend_profile']
        self.protocol = collection.contract(self.root, 'math500')
        self.work = recovery.ordered_work(self.questions, self.config)
        self.source = self.root / 'artifacts/runs/fast-llada-math500-fixture'
        self.source.mkdir(parents=True)
        self.preserved = 3997  # Interrupted partway through a question's eight answers.
        rows, costs = [], []
        for index, (q, cid, seed) in enumerate(self.work[:self.preserved]):
            # Even truncated/invalid answers must survive; never filter on quality.
            rows.append(asdict(Candidate(q.question_id, cid, seed, '' if index == 0 else 'Synthetic only', index != 0, index == 0)))
            costs.append({'question_id': q.question_id, 'candidate_id': cid,
                          **asdict(CostRecord(trajectory_nfe=512, generation_gpu_seconds=1,
                                             generation_wall_seconds=1.1, peak_memory_bytes=100))})
        self.save_rows(self.source / 'candidates.jsonl', rows)
        self.save_rows(self.source / 'costs.jsonl', costs)
        manifest = {'status': 'interrupted', 'error_type': 'KeyboardInterrupt',
                    'mode': collection.SCOPE, 'protocol': self.protocol, 'backend': 'fast_dllm_v1',
                    'task': 'math500', 'candidates_per_question': 8, 'evaluation_ready': False,
                    'is_benchmark_result': False, 'api_calls': 0, 'incomplete_candidates': 1,
                    'truncated_candidates': 1, 'empty_candidates': 1,
                    'question_count': 500, 'completed_candidates': self.preserved,
                    'questions_sha256': self.config['datasets']['questions_sha256'],
                    'dependencies': {name: 'fixture' for name in ('torch', 'transformers', 'numpy')},
                    'total_seconds_including_load_and_warmup': 12345}
        (self.source / 'manifest.json').write_text(json.dumps(manifest))
        (self.source / 'config.snapshot.json').write_text(json.dumps(self.config))
        (self.source / 'dependencies.txt').write_text('fixture\n')
        shutil.copyfile(self.root / collection.profile_path('math500'), self.source / 'collection-profile.json')
        shutil.copyfile(self.root / collection.LOCK, self.source / 'generation-lock.json')
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
            stack.enter_context(patch.object(collection, 'runtime_check', return_value={'gpu_name': 'synthetic'}))
            stack.enter_context(patch.object(collection, 'make_backend', return_value=backend))
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
        self.assertEqual(backend.warmups, ['runtime/math500/warmup'])
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
        self.assertEqual(collection.audit(self.root, archive)['saved_candidates'], 4000)
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
        claim = json.loads((self.root / 'artifacts/collection-recovery-claims' / (self.source.name + '.json')).read_text())
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
        claim = json.loads((self.root / 'artifacts/collection-recovery-claims' / (self.source.name + '.json')).read_text())
        archive = self.copy_with_receipt(self.root / 'artifacts/runs' / claim['run_id'])
        self.assertEqual(recovery.inspect_run(self.root, archive)['count'], 3998)
        with self.assertRaisesRegex(ValueError, 'interrupted, incomplete'):
            self.make_plan(archive)

    def test_source_changed_on_server_stops_before_runtime(self):
        with (self.source / 'candidates.jsonl').open('a') as stream:
            stream.write('\n')
        with patch.object(collection, 'runtime_check') as hardware:
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
        path = self.root / 'scripts/recover_collection.py'
        path.write_text(path.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'implementation differs'):
            recovery.check_server_source(self.root, self.plan)
        shutil.copyfile(ROOT / 'scripts/recover_collection.py', path)
        # The shared Makefile is checked in both layers; use a core-only file here.
        core = self.root / 'src/jev_dllm/reproducibility.py'
        core.write_text(core.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'Pinned collection file changed'):
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
        rows[1]['text'] = 'Changed retained answer'
        self.save_rows(path, rows)
        self.refresh_receipt(archive)
        with self.assertRaisesRegex(ValueError, 'source prefix changed'):
            recovery.inspect_run(self.root, archive)

    def test_runtime_version_change_rejected_before_model_load(self):
        with patch.object(collection, 'runtime_check', return_value={'gpu_name': 'synthetic'}), \
                patch.object(recovery.importlib.metadata, 'version', return_value='wrong'), \
                patch.object(collection, 'make_backend') as backend:
            with self.assertRaisesRegex(ValueError, 'Runtime versions differ'):
                recovery.recover(self.root, self.plan)
            backend.assert_not_called()

    def test_upload_contains_recovery_code_shared_makefile_and_validated_metadata(self):
        calls, transferred = [], []
        def command(args, **kwargs):
            calls.append(args)
            if args[:2] == ['git', 'ls-files']:
                return SimpleNamespace(stdout=('\0'.join(recovery.EXTENSION_FILES) + '\0.env\0data/private/answers.json\0').encode())
            if '--files-from' in args:
                transferred.extend(Path(args[args.index('--files-from') + 1]).read_text().splitlines())
            self.assertNotIn('--delete', args)
            return SimpleNamespace(returncode=0)
        config = {'local_results': str(self.archive.parent), 'remote_root': '/scratch/person/project', 'ssh_host': 'fixture'}
        with patch.object(operations.remote, 'upload') as general_upload, \
                patch.object(operations.subprocess, 'run', side_effect=command):
            result = operations.upload_plan(self.root, self.plan, config)
            general_upload.assert_not_called()
        self.assertEqual(set(transferred), set(recovery.EXTENSION_FILES))
        self.assertEqual(set(transferred) & set(collection.locked_paths(self.root)), {'Makefile'})
        rsync = [args for args in calls if args[0] == 'rsync']
        self.assertEqual(len(rsync), 2)
        self.assertEqual(rsync[1][-2], str(self.plan.resolve()))
        self.assertEqual(result['server_plan'], 'artifacts/collection-recovery-plans/' + self.source.name + '.json')
        plan = json.loads(self.plan.read_text())
        plan['unapproved_extra'] = 'never upload this'
        self.plan.write_text(json.dumps(plan))
        with patch.object(operations.subprocess, 'run') as network:
            with self.assertRaisesRegex(ValueError, 'Existing recovery plan differs'):
                operations.upload_plan(self.root, self.plan, config)
            network.assert_not_called()

    def test_cli_audit_and_manual_plan_require_no_gpu(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/recover_collection.py'), '--root', str(self.root),
                                 'audit', '--run-dir', str(self.archive)], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)['remaining_candidates'], 3)

    def test_logged_boundary_of_2424_preserves_303_full_questions(self):
        for name in ('candidates.jsonl', 'costs.jsonl'):
            path = self.archive / name
            self.save_rows(path, recovery.jsonl(path)[:2424])
        path = self.archive / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['completed_candidates'] = 2424
        path.write_text(json.dumps(manifest))
        self.refresh_receipt(self.archive)
        result = recovery.prepare_plan(self.root, self.archive, self.root / 'artifacts/boundary-plan.json')
        self.assertEqual((result['preserved_candidates'], result['remaining_candidates']), (2424, 1576))
        self.assertEqual(result['next_candidate_id'], self.questions[303].question_id + '/c00')
        self.assertEqual(result['next_seed'], self.work[2424][2])

    def test_false_status_and_completion_flags_rejected(self):
        path = self.archive / 'manifest.json'
        original = path.read_bytes()
        for changes in ({'status': 'complete'}, {'empty_candidates': 0},
                        {'task': 'humaneval'}, {'evaluation_ready': True}, {'completed_candidates': 3998}):
            with self.subTest(changes=changes):
                m = json.loads(original)
                m.update(changes)
                path.write_text(json.dumps(m))
                self.refresh_receipt(self.archive)
                with self.assertRaises(ValueError):
                    recovery.inspect_run(self.root, self.archive)
        path.write_bytes(original)

    def test_modified_segment_history_rejected_even_after_new_receipt(self):
        archive = self.copy_with_receipt(self.execute(FakeBackend()))
        path = archive / 'manifest.json'
        m = json.loads(path.read_text())
        m['prior_segments'][0]['segment_seconds_including_load_and_warmup'] = 0
        path.write_text(json.dumps(m))
        self.refresh_receipt(archive)
        with self.assertRaisesRegex(ValueError, 'timing history differs'):
            recovery.inspect_run(self.root, archive)

    def test_invalid_new_backend_record_is_not_appended(self):
        class InvalidBackend(FakeBackend):
            def generate(self, question, *, candidate_id, seed):
                candidate, cost = super().generate(question, candidate_id=candidate_id, seed=seed)
                if candidate_id != 'warmup':
                    candidate = Candidate(**{**asdict(candidate), "seed": seed + 1})
                return candidate, cost
        with self.assertRaises(ValueError):
            self.execute(InvalidBackend())
        claim = json.loads((self.root / 'artifacts/collection-recovery-claims' / (self.source.name + '.json')).read_text())
        run = self.root / 'artifacts/runs' / claim['run_id']
        self.assertEqual((run / 'candidates.jsonl').read_bytes(), (self.source / 'candidates.jsonl').read_bytes())
        self.assertEqual(json.loads((run / 'manifest.json').read_text())['status'], 'failed')

    def test_humaneval_plan_uses_its_own_pinned_order_and_total(self):
        with self.assertRaises(KeyboardInterrupt):
            fixtures.CollectionTests.execute(self, 'humaneval', fixtures.FakeBackend(fail_after=3))
        source = next((self.root / 'artifacts/runs').glob('fast-llada-humaneval-*'))
        result = recovery.prepare_plan(self.root, self.copy_with_receipt(source))
        self.assertEqual(result['task'], 'humaneval')
        self.assertEqual(result['remaining_candidates'], 1309)
        self.assertEqual(result['next_candidate_id'], 'humaneval/test/00000/c03')


if __name__ == '__main__':
    unittest.main()
