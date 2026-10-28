"""Week 4 Decision Point A: the frozen selector record, its pins and its evidence; offline, no API."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jev_dllm.experiments import frozen
from jev_dllm.selection import frozen_selector as fs


class FrozenSelectorTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name in [fs.RECORD, *fs.pinned_paths()]:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, self.root / name)
        self.record = json.loads((self.root / fs.RECORD).read_text())

    def rewrite(self, change):
        change(self.record)
        (self.root / fs.RECORD).write_text(json.dumps(self.record))

    def test_repository_record_passes_and_pins_every_prompt_and_rule_file(self):
        record = fs.check(ROOT)
        self.assertEqual((record['selector']['name'], record['selector']['score']), ('jev_pointwise', 'correct'))
        self.assertFalse(record['held_out_used_for_choice'])
        for benchmark in fs.BENCHMARKS:
            for path in fs.prompts(benchmark).values():
                self.assertEqual(record['files_sha256'][path], frozen.sha256(ROOT / path))
        self.assertIn('src/jev_dllm/selection/week3.py', record['files_sha256'])

    def test_selection_is_highest_correct_with_pool_order_ties(self):
        ids = [f'q/c{i:02}' for i in range(4)]
        self.assertEqual(fs.select(ids, {'q/c01': .7, 'q/c03': .9}), 'q/c03')
        self.assertEqual(fs.select(ids, {'q/c02': .9, 'q/c03': .9}), 'q/c02')
        self.assertEqual(fs.select(ids, {}), 'q/c00')

    def test_changed_prompt_or_code_is_refused(self):
        for name in ('prompts/week02/jev_pointwise_correct_code.txt', 'src/jev_dllm/selection/week3.py'):
            with self.subTest(file=name):
                original = (self.root / name).read_bytes()
                (self.root / name).write_bytes(original + b'\n')
                with self.assertRaisesRegex(ValueError, 'Frozen selector file changed'):
                    fs.check(self.root)
                (self.root / name).write_bytes(original)
        fs.check(self.root)

    def test_record_that_disagrees_with_the_code_is_refused(self):
        changes = {
            'Jev model or protocol': lambda r: r['jev'].update(model='jev-9.9.9'),
            'unexpected scope or rule': lambda r: r['selector'].update(score='consistent'),
            'Benchmark-to-prompt map': lambda r: r['benchmarks']['humaneval'].update(
                correct='prompts/week02/jev_pointwise_correct_math.txt'),
            'unexpected file list': lambda r: r['files_sha256'].pop('src/jev_dllm/schema.py'),
        }
        for message, change in changes.items():
            with self.subTest(message=message):
                self.setUp()
                self.rewrite(change)
                with self.assertRaisesRegex(ValueError, message):
                    fs.check(self.root)
        self.setUp()
        self.rewrite(lambda r: r.update(held_out_used_for_choice=True))
        with self.assertRaises(ValueError):
            fs.check(self.root)

    def write_analysis(self, task, selectors):
        run = self.root / 'artifacts/runs' / self.record['evidence']['analyses'][task]['run']
        run.mkdir(parents=True)
        metrics = {'by_subset_size': {'16': {'selectors': {k: {'accuracy': v} for k, v in selectors.items()}}}}
        (run / 'metrics.json').write_text(json.dumps(metrics))
        self.record['evidence']['analyses'][task]['metrics_sha256'] = frozen.sha256(run / 'metrics.json')

    def test_evidence_is_verified_where_present_and_refused_when_it_disagrees(self):
        self.assertEqual(set(fs.verify_evidence(self.root, self.record).values()), {'not available locally'})
        accuracy = self.record['evidence']['analyses']['gsm8k']['accuracy']
        self.write_analysis('gsm8k', {'jev_pointwise': accuracy, 'jev_listwise': accuracy, 'majority': .5})
        self.assertEqual(fs.verify_evidence(self.root, self.record)['gsm8k'], 'verified')  # Ties allowed.
        self.write_analysis('math', {'jev_pointwise': self.record['evidence']['analyses']['math']['accuracy'],
                                     'prm_min': .99})
        with self.assertRaisesRegex(ValueError, 'no longer shows jev_pointwise'):
            fs.verify_evidence(self.root, self.record)
        self.record['evidence']['analyses']['math']['metrics_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'differs from the one the freeze cites'):
            fs.verify_evidence(self.root, self.record)

    def test_make_target_routes_to_the_offline_check(self):
        result = subprocess.run(['make', '--no-print-directory', '-n', 'week4-selector-check', 'PYTHON=python3'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('python3 scripts/week4.py selector-check', result.stdout)


if __name__ == '__main__':
    unittest.main()
