"""Offline freeze, archive integrity and data-boundary regression checks."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jev_dllm.experiments import frozen
from jev_dllm.experiments.generation import generate_pool
from jev_dllm.experiments.selection import judge_pools, load_questions
from jev_dllm.operations import remote
from jev_dllm.reproducibility import trajectory_seed


class FrozenGSM8KTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        record = json.loads((ROOT / frozen.RECORD).read_text())
        # A fresh clone has no ignored datasets or upstream source bundle.
        # Build public synthetic splits and a hash-only fake source bundle here.
        files = [frozen.PROFILE, frozen.RECORD, *record['public_files_sha256']]
        for name in files:
            if name.startswith('data/'):
                continue
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        self.config = json.loads((self.root / frozen.PROFILE).read_text())
        def save(name, value):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return frozen.sha256(path)
        heldout = [{'question_id': f'gsm8k/test/{i:05}', 'dataset': 'gsm8k',
                    'split': 'heldout', 'prompt': f'Synthetic held-out fixture {i}?'} for i in range(500)]
        dev = [{'question_id': f'gsm8k/train/{i:05}', 'dataset': 'gsm8k',
                'split': 'dev', 'prompt': f'Synthetic development fixture {i}?'} for i in range(20)]
        save('data/processed/gsm8k-dev.json', dev)
        self.config['datasets']['questions_sha256'] = save('data/processed/gsm8k-heldout.json', heldout)
        self.config['datasets']['manifest_sha256'] = save('data/manifests/gsm8k-week01.json',
            {'question_ids': {'heldout': [q['question_id'] for q in heldout]}})
        source = json.loads((self.root / 'configs/week01/fast-source.json').read_text())
        for name, info in source['files'].items():
            path = self.root / 'third_party/fast_dllm_v1' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic hash-check fixture; never imported\n')
            info['sha256'] = frozen.sha256(path)
        source_hash = save('configs/week01/fast-source.json', source)
        self.config['backends']['fast_dllm_v1']['source_manifest_sha256'] = source_hash
        development = json.loads((self.root / 'configs/week01/fast-diversity.json').read_text())
        development['backends']['fast_dllm_v1']['source_manifest_sha256'] = source_hash
        save('configs/week01/fast-diversity.json', development)
        record['profile_sha256'] = save(frozen.PROFILE, self.config)
        record['public_files_sha256'] = {name: frozen.sha256(self.root / name)
            for name in record['public_files_sha256']}
        save(frozen.RECORD, record)
        self.questions = self.root / self.config['datasets']['questions_path']

    def check(self):
        return frozen.check_protocol(self.root, self.config, self.questions)

    def test_public_check_needs_no_private_data_or_model(self):
        self.assertFalse((self.root / 'data/private').exists())
        self.assertEqual(self.check()['scope'], frozen.PHASE)

    def test_changed_settings_or_code_are_rejected(self):
        self.config['generator']['temperature'] = 0.7
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            self.check()
        self.config['generator']['temperature'] = 1.0
        path = self.root / 'src/jev_dllm/evaluation/answers.py'
        path.write_text(path.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
            self.check()

    def test_dev_questions_cannot_enter_500_profile(self):
        with self.assertRaisesRegex(ValueError, '500-question'):
            frozen.check_protocol(self.root, self.config, self.root / 'data/processed/gsm8k-dev.json')

    def test_boolean_freeze_cannot_bypass_generation_gate(self):
        config = json.loads((self.root / 'configs/week01/fast-diversity.json').read_text())
        config['frozen'] = True
        with patch('jev_dllm.experiments.generation.FastLLaDA') as model:
            with self.assertRaisesRegex(ValueError, 'scoped frozen'):
                generate_pool(self.root, config, self.questions, self.root / 'runs')
            model.assert_not_called()
        self.assertFalse((self.root / 'runs').exists())

    def archive(self):
        run = self.root / 'artifacts/downloads/lab/fast-llada-gsm8k500-fixture'
        run.mkdir(parents=True)
        protocol = self.check()
        (run / 'manifest.json').write_text(json.dumps({'status': 'complete', 'question_count': 500,
            'completed_candidates': 4000, 'questions_sha256': self.config['datasets']['questions_sha256'],
            'protocol': protocol}))
        (run / 'config.snapshot.json').write_text(json.dumps(self.config))
        rows = [{'question_id': q.question_id, 'candidate_id': f'{q.question_id}/c{i:02}',
                 'seed': trajectory_seed(self.config['runtime']['master_seed'], q.question_id, i),
                 'text': 'Synthetic fixture only. Final answer: 2', 'completed': True, 'truncated': False}
                for q in load_questions(self.questions) for i in range(8)]
        (run / 'candidates.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.receipt(run)
        return run, rows

    def receipt(self, run):
        snapshot = {'run_id': run.name, 'status': 'complete', 'files': {
            p.name: {'bytes': p.stat().st_size, 'sha256': frozen.sha256(p)} for p in run.iterdir()}}
        receipts = run.parent / '.receipts'
        receipts.mkdir(exist_ok=True)
        (receipts / (run.name + '.json')).write_text(json.dumps({
            'snapshot': snapshot, 'digest': remote.snapshot_digest(snapshot)}))

    def test_complete_archive_requires_matching_receipt_and_no_local_edits(self):
        run, _ = self.archive()
        candidates = run / 'candidates.jsonl'
        self.assertEqual(frozen.check_pool(self.root, self.config, self.questions, candidates), self.check())
        candidates.write_text(candidates.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Local copy differs'):
            frozen.check_pool(self.root, self.config, self.questions, candidates)

    def test_verified_but_wrong_seed_or_incomplete_run_rejected(self):
        run, rows = self.archive()
        rows[0]['seed'] += 1
        candidates = run / 'candidates.jsonl'
        candidates.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.receipt(run)
        with self.assertRaisesRegex(ValueError, 'seed differs'):
            frozen.check_pool(self.root, self.config, self.questions, candidates)
        manifest = json.loads((run / 'manifest.json').read_text())
        manifest['status'] = 'interrupted'
        (run / 'manifest.json').write_text(json.dumps(manifest))
        self.receipt(run)
        with self.assertRaisesRegex(ValueError, 'completed 500-question'):
            frozen.check_pool(self.root, self.config, self.questions, candidates)

    def test_scorer_and_private_reference_changes_rejected(self):
        with self.assertRaisesRegex(ValueError, 'requires gsm8k_units_v3'):
            frozen.check_evaluation(self.config, self.root / 'missing', 'gsm8k_strict_v1')
        references = self.root / 'synthetic-references.json'
        references.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'references differ'):
            frozen.check_evaluation(self.config, references, 'gsm8k_units_v3')

    def test_truncation_blocks_entire_batch_before_first_charge(self):
        run, rows = self.archive()
        rows[-1]['truncated'] = True
        candidates = run / 'candidates.jsonl'
        candidates.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with patch('jev_dllm.experiments.selection.TypeSafeJudge') as api:
            with self.assertRaises(ValueError):
                judge_pools(self.questions, candidates, self.root / 'prompts/jev_final.txt',
                    self.root / 'runs', 'jev-1.13.0', live=True)
            api.assert_not_called()
        self.assertFalse((self.root / 'runs').exists())

    def test_hardware_contract_rejects_multiple_or_wrong_gpus(self):
        for count, name in ((2, 'NVIDIA A100-SXM4-80GB'), (1, 'Tesla V100'), (1, 'NVIDIA A100-SXM4-40GB')):
            cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: count, get_device_name=lambda _: name)
            with self.assertRaises(RuntimeError):
                frozen.check_hardware(self.config, SimpleNamespace(cuda=cuda))
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda _: 'NVIDIA A100-SXM4-80GB')
        frozen.check_hardware(self.config, SimpleNamespace(cuda=cuda))

    def test_upload_includes_only_public_subset_not_private_references(self):
        sampler = self.root / 'third_party/llada/generate.py'
        sampler.parent.mkdir(parents=True)
        sampler.write_text('# fixture only\n')
        transferred = []
        def command(args, **kwargs):
            if args[:2] == ['git', 'ls-files']:
                return SimpleNamespace(stdout=b'Makefile\0.env\0data/private/secret.json\0reports/private.md\0')
            if args[:2] == ['git', 'rev-parse']:
                return SimpleNamespace(stdout='fixture-commit\n')
            if args[0] == 'rsync':
                transferred.extend(Path(args[args.index('--files-from') + 1]).read_text().splitlines())
            return SimpleNamespace(returncode=0)
        with patch('jev_dllm.operations.remote.subprocess.run', side_effect=command):
            remote.upload(self.root, {'ssh_host': 'fixture', 'remote_root': '/scratch/person/project'})
        self.assertIn('data/processed/gsm8k-heldout.json', transferred)
        self.assertFalse(any('/private/' in p or p == '.env' or p.startswith('reports/') for p in transferred))


if __name__ == '__main__':
    unittest.main()
