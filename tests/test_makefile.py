"""Command routing and implementation pins for the public single-Makefile build."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jev_dllm.experiments import collection, frozen


class MakefileTests(unittest.TestCase):
    def make(self, *args):
        return subprocess.run(['make', '--no-print-directory', *args], cwd=ROOT,
                              capture_output=True, text=True)

    def test_default_command_is_offline_help_and_only_one_makefile_exists(self):
        self.assertEqual(sorted(p.name for p in ROOT.glob('Makefile*')), ['Makefile'])
        result = self.make()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')  # Includes duplicate-rule warnings.
        self.assertIn('Single command entry point', result.stdout)
        self.assertNotIn('scripts/', result.stdout)

    def test_recovery_commands_route_to_the_correct_task(self):
        routes = {
            'gsm8k500-recovery-plan': 'scripts/recovery.py plan',
            'gsm8k500-recovery-audit': 'scripts/recovery.py audit',
            'gsm8k500-recovery-upload': 'scripts/recovery.py upload',
            'gpu-gsm8k500-recovery-check': 'scripts/recovery.py source-check',
            'gpu-gsm8k500-resume': 'scripts/recovery.py run',
            'collection-recovery-plan': 'scripts/recover_collection.py plan',
            'collection-recovery-audit': 'scripts/recover_collection.py audit',
            'collection-recovery-upload': 'scripts/recover_collection.py upload',
            'gpu-collection-recovery-check': 'scripts/recover_collection.py source-check',
            'gpu-collection-resume': 'scripts/recover_collection.py run',
        }
        for prefix, script, actions in (
            ('humaneval-jev-continuation', 'continue_humaneval_jev', ('prepare', 'check', 'live', 'evaluate')),
            ('math500-jev-continuation', 'continue_math500_jev', ('prepare', 'check', 'live', 'audit')),
            ('math500-jev-resume', 'resume_math500_jev', ('prepare', 'check', 'live', 'audit')),
        ):
            routes.update({f'{prefix}-{action}': f'scripts/{script}.py {action}' for action in actions})
        for target, expected in routes.items():
            with self.subTest(target=target):
                result = self.make('-n', target, 'RUN_DIR=fixture archive', 'SOURCE_PLAN=fixture source',
                                   'PLAN_DIR=fixture plan', 'RECOVERY_PLAN=fixture recovery.json',
                                   'SCORE_RUN=fixture scores', 'APPROVED_CALLS=1', 'GPU_ID=3')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, '')
                self.assertIn(expected + ' ', result.stdout)
                # This exercises make's command expansion without running Python/GPU/API work.
                if target.endswith('-live'):
                    self.assertIn('--plan "fixture plan" --approved-calls "1"', result.stdout)

    def test_live_recovery_requires_an_explicit_request_count_before_python(self):
        for target in ('jev-continuation-live', 'humaneval-jev-continuation-live',
                       'math500-jev-continuation-live', 'math500-jev-resume-live'):
            with self.subTest(target=target):
                result = self.make(target, 'PLAN=fixture', 'PLAN_DIR=fixture', 'APPROVED_CALLS=',
                                   'PYTHON=must-not-be-started')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('explicitly approved', result.stdout)
                self.assertNotIn('must-not-be-started', result.stdout + result.stderr)

    def test_new_build_pins_match_files_and_preserve_historical_records(self):
        for name, key in ((frozen.RECORD, 'public_files_sha256'), (collection.LOCK, 'files_sha256')):
            with self.subTest(record=name):
                record = json.loads((ROOT / name).read_text())
                self.assertEqual(record['implementation_revision'], 4)
                prior = record['supersedes']
                self.assertNotEqual(name, prior['record'])
                self.assertEqual(frozen.sha256(ROOT / prior['record']), prior['sha256'])
                old = json.loads((ROOT / prior['record']).read_text())
                for path, expected in record[key].items():
                    if path.startswith('data/processed/'):
                        # Fresh clones regenerate these ignored public inputs; their pins stay unchanged.
                        self.assertEqual(expected, old[key][path])
                    else:
                        self.assertEqual(frozen.sha256(ROOT / path), expected, path)
        self.assertEqual(set(json.loads((ROOT / collection.LOCK).read_text())['files_sha256']),
                         set(collection.locked_paths(ROOT)))
        record = json.loads((ROOT / frozen.RECORD).read_text())
        old = json.loads((ROOT / record['supersedes']['record']).read_text())
        self.assertEqual(record['profile_sha256'], old['profile_sha256'])
        self.assertEqual(record['profile_sha256'], frozen.sha256(ROOT / frozen.PROFILE))


if __name__ == '__main__':
    unittest.main()
