"""Independent task collection, provenance and manual launch boundaries; no GPU/API."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_gsm8k500 as fixtures
from jev_dllm.experiments import collection, frozen
from jev_dllm.operations import collection as operations, remote
from jev_dllm.reproducibility import trajectory_seed
from jev_dllm.schema import Candidate, CostRecord

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    def __init__(self, fail_after=None, failure=KeyboardInterrupt):
        self.calls = []
        self.fail_after, self.failure = fail_after, failure

    def generate(self, question, *, candidate_id, seed):
        if candidate_id != 'warmup':
            if len(self.calls) == self.fail_after:
                raise self.failure('synthetic stop')
            self.calls.append((question.question_id, candidate_id, seed))
        # Retain incomplete/empty outputs rather than retrying based on quality.
        flagged = len(self.calls) % 3 == 0
        return (Candidate(question.question_id, candidate_id, seed,
                          '' if flagged else 'raise RuntimeError("Never execute generated code here")',
                          not flagged, flagged),
                CostRecord(trajectory_nfe=2, generation_gpu_seconds=.1,
                           generation_wall_seconds=.2, peak_memory_bytes=100))


class CollectionTests(unittest.TestCase):
    def setUp(self):
        fixtures.FrozenGSM8KTests.setUp(self)
        for name in collection.EXTENSION:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        for task, count in collection.TASKS.items():
            spec = self.read(collection.profile_path(task))
            config = spec['backend_profile']
            config['backends']['fast_dllm_v1']['source_manifest_sha256'] = frozen.sha256(
                self.root / 'configs/week01/fast-source.json')
            rows = [{'question_id': f'{task}/test/{i:05}', 'dataset': task, 'split': 'heldout',
                     'prompt': f'Synthetic {task} public problem {i}'} for i in range(count)]
            d = config['datasets']
            d['questions_sha256'] = self.save(d['questions_path'], rows)
            d['manifest_sha256'] = self.save(d['manifest_path'], {'question_ids': [q['question_id'] for q in rows]})
            self.save(collection.profile_path(task), spec)
        self.relock()

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def save(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return frozen.sha256(path)

    def relock(self):
        # Fixture-only preparation; real collection pins must not change after a run.
        self.save(collection.LOCK, {'scope': collection.SCOPE, 'schema_version': 1,
            'files_sha256': {p: frozen.sha256(self.root / p) for p in collection.locked_paths(self.root)}})

    def edit_questions(self, task, change):
        spec = self.read(collection.profile_path(task))
        d = spec['backend_profile']['datasets']
        rows = self.read(d['questions_path'])
        change(rows)
        d['questions_sha256'] = self.save(d['questions_path'], rows)
        self.save(collection.profile_path(task), spec)
        self.relock()

    def execute(self, task, backend):
        with ExitStack() as stack:
            stack.enter_context(patch.object(collection, 'runtime_check', return_value={'gpu_name': 'synthetic'}))
            stack.enter_context(patch.object(collection, 'make_backend', return_value=backend))
            stack.enter_context(patch.object(collection.platform, 'platform', return_value='synthetic'))
            stack.enter_context(patch.object(collection.importlib.metadata, 'version', return_value='fixture'))
            stack.enter_context(patch.object(collection.subprocess, 'run', return_value=SimpleNamespace(stdout='fixture\n')))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return collection.generate(self.root, task)

    def archive(self, run):
        target = self.root / 'artifacts/downloads/lab' / run.name
        shutil.copytree(run, target)
        self.receipt(target)
        return target

    def receipt(self, run):
        snapshot = {'run_id': run.name, 'status': json.loads((run / 'manifest.json').read_text())['status'],
                    'files': {p.name: {'bytes': p.stat().st_size, 'sha256': frozen.sha256(p)} for p in run.iterdir()}}
        receipts = run.parent / '.receipts'
        receipts.mkdir(exist_ok=True)
        (receipts / (run.name + '.json')).write_text(json.dumps({'snapshot': snapshot, 'digest': remote.snapshot_digest(snapshot)}))

    def partial(self, failure=KeyboardInterrupt):
        with self.assertRaises(failure):
            self.execute('humaneval', FakeBackend(fail_after=3, failure=failure))
        return next((self.root / 'artifacts/runs').iterdir())

    def test_task_inputs_are_disjoint_public_only_and_settings_are_not_scoring_readiness(self):
        seeds = []
        for task, count in collection.TASKS.items():
            spec, questions = collection.check(self.root, task)
            self.assertEqual(len(questions), count)
            self.assertFalse(spec['evaluation_ready'])
            self.assertFalse((self.root / 'data/private').exists())
            config = spec['backend_profile']
            seeds.extend(trajectory_seed(config['runtime']['master_seed'], q.question_id, i)
                         for q in questions for i in range(8))
            self.assertIn(task, config['prompts']['generation_math']['path'])
        self.assertEqual(len(seeds), len(set(seeds)))
        with self.assertRaises(ValueError):
            collection.check(self.root, 'gsm8k')

    def test_frozen_gsm8k_code_cannot_change_before_collection(self):
        (self.root / 'Makefile').write_text('changed')
        with patch.object(collection, 'runtime_check') as runtime:
            with self.assertRaisesRegex(ValueError, 'Pinned collection file changed'):
                collection.generate(self.root, 'math500')
            runtime.assert_not_called()

    def test_private_reference_field_is_rejected_even_with_matching_hashes(self):
        self.edit_questions('humaneval', lambda rows: rows[0].update(canonical_solution='private'))
        with self.assertRaises(TypeError):
            collection.check(self.root, 'humaneval')

    def test_duplicate_prompts_or_wrong_order_are_rejected(self):
        self.edit_questions('math500', lambda rows: rows[0].update(prompt=rows[1]['prompt']))
        with self.assertRaisesRegex(ValueError, 'uniqueness'):
            collection.check(self.root, 'math500')
        self.edit_questions('humaneval', lambda rows: rows.reverse())
        with self.assertRaisesRegex(ValueError, 'order'):
            collection.check(self.root, 'humaneval')

    def test_task_prompt_cannot_be_silently_substituted(self):
        path = self.root / 'prompts/collection/humaneval.txt'
        path.write_text('Compute a number: {question}')
        with self.assertRaisesRegex(ValueError, 'Pinned collection file changed'):
            collection.check(self.root, 'humaneval')
        self.relock()
        with self.assertRaisesRegex(ValueError, 'Task-specific generation prompt'):
            collection.check(self.root, 'humaneval')

    def test_complete_pools_have_exact_ids_seeds_costs_and_retain_flagged_answers(self):
        original = {p: (self.root / p).read_bytes() for p in (frozen.PROFILE, frozen.RECORD, 'Makefile')}
        for task, count in collection.TASKS.items():
            backend = FakeBackend()
            run = self.execute(task, backend)
            spec, questions = collection.check(self.root, task)
            expected = [(q.question_id, f'{q.question_id}/c{i:02}',
                         trajectory_seed(spec['backend_profile']['runtime']['master_seed'], q.question_id, i))
                        for q in questions for i in range(8)]
            self.assertEqual(backend.calls, expected)
            info = collection.audit(self.root, self.archive(run))
            self.assertEqual(info['saved_candidates'], count * 8)
            self.assertEqual(info['truncated_candidates'], count * 8 // 3)
            self.assertEqual(info['empty_candidates'], info['truncated_candidates'])
            self.assertFalse(info['correctness_scored'])
            self.assertEqual(info['jev_requests_sent'], 0)
        self.assertEqual(original, {p: (self.root / p).read_bytes() for p in original})

    def test_interruption_flushes_records_and_preserves_terminal_manifest(self):
        run = self.partial()
        info = collection.audit(self.root, self.archive(run))
        self.assertEqual((info['status'], info['saved_candidates']), ('interrupted', 3))
        self.assertEqual(info['truncated_candidates'], 1)
        original = {p.name: p.read_bytes() for p in run.iterdir()}
        with self.assertRaises(KeyboardInterrupt):
            self.execute('humaneval', FakeBackend(fail_after=1))
        self.assertEqual(len(list((self.root / 'artifacts/runs').iterdir())), 2)
        self.assertEqual(original, {p.name: p.read_bytes() for p in run.iterdir()})

    def test_backend_error_retains_partial_data_without_retry(self):
        run = self.partial(RuntimeError)
        self.assertEqual(collection.audit(self.root, self.archive(run))['status'], 'failed')

    def test_copied_archive_tampering_or_wrong_seed_is_rejected(self):
        copied = self.archive(self.partial())
        path = copied / 'candidates.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['seed'] += 1
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'Local copy differs'):
            collection.audit(self.root, copied)
        self.receipt(copied)
        with self.assertRaisesRegex(ValueError, 'seed'):
            collection.audit(self.root, copied)

    def test_cost_mismatch_and_incomplete_full_run_are_rejected(self):
        copied = self.archive(self.partial())
        path = copied / 'costs.jsonl'
        before = path.read_text()
        rows = [json.loads(line) for line in before.splitlines()]
        rows[0]['candidate_id'] = 'wrong'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.receipt(copied)
        with self.assertRaisesRegex(ValueError, 'Mismatched cost'):
            collection.audit(self.root, copied)
        path.write_text(before)
        mpath = copied / 'manifest.json'
        m = json.loads(mpath.read_text())
        m['status'] = 'complete'
        mpath.write_text(json.dumps(m))
        self.receipt(copied)
        with self.assertRaisesRegex(ValueError, 'counts disagree'):
            collection.audit(self.root, copied)

    def test_task_upload_excludes_gsm8k_inputs_private_data_and_weights(self):
        transferred = []
        required = collection.EXTENSION + (collection.LOCK,)
        def command(args, **kwargs):
            if args[:2] == ['git', 'ls-files']:
                return SimpleNamespace(stdout=('\0'.join(required) + '\0.env\0data/private/answers.json\0').encode())
            if args[:2] == ['git', 'rev-parse']:
                return SimpleNamespace(stdout='fixture-commit\n')
            if args[0] == 'rsync':
                transferred.extend(Path(args[args.index('--files-from') + 1]).read_text().splitlines())
                self.assertNotIn('--delete', args)
            return SimpleNamespace(returncode=0)
        with patch.object(operations.subprocess, 'run', side_effect=command):
            operations.upload(self.root, {'ssh_host': 'fixture', 'remote_root': '/scratch/person/project'})
        self.assertEqual(set(transferred), set(required + collection.INPUTS + ('artifacts/collection-deployment.json',)))
        self.assertFalse(any('gsm8k' in p or p == '.env' or 'private' in p or 'weights' in p for p in transferred))

    def test_tmux_launch_passes_gpu_cache_explicitly_and_refuses_existing_session(self):
        env = {'GPU_ID': '2', 'JEV_SHARED_HUB_CACHE': '/scratch/personal hub'}
        with patch.dict(os.environ, env), patch.object(operations.subprocess, 'run') as run:
            run.side_effect = [SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]
            result = operations.start(self.root, 'math500')
            self.assertEqual(result['physical_gpu_id'], '2')
            args = run.call_args_list[-1].args[0]
            self.assertEqual(args[:5], ['tmux', 'new-session', '-d', '-s', 'jev-math500'])
            command = shlex.split(args[-1])
            self.assertEqual(command[command.index('--gpu-id') + 1], '2')
            self.assertEqual(command[command.index('--hub-cache') + 1], env['JEV_SHARED_HUB_CACHE'])
            run.reset_mock(side_effect=True)
            run.return_value = SimpleNamespace(returncode=0)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                operations.start(self.root, 'math500')
            self.assertEqual(run.call_count, 1)

    def test_multi_gpu_setting_is_rejected_before_any_launch(self):
        with patch.dict(os.environ, {'GPU_ID': '2,4'}), patch.object(operations.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                operations.start(self.root, 'humaneval')
            run.assert_not_called()

    def test_session_keeps_error_pane_and_runs_selected_task_without_restart(self):
        with patch.object(operations.subprocess, 'run') as run, redirect_stdout(io.StringIO()):
            run.return_value = SimpleNamespace(returncode=0)
            operations.session_run(self.root, 'humaneval', '4', '/scratch/cache')
            self.assertEqual(run.call_args_list[0].args[0], ['tmux', 'set-window-option', 'remain-on-exit', 'on'])
            self.assertEqual(run.call_args_list[1].args[0], ['make', '-f', 'Makefile', 'gpu-humaneval', 'GPU_ID=4'])
            self.assertEqual(run.call_args_list[1].kwargs['env']['JEV_SHARED_HUB_CACHE'], '/scratch/cache')
            self.assertEqual(run.call_count, 2)


if __name__ == '__main__':
    unittest.main()
