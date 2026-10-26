"""Week 2 pool generation, resume and equivalence with a fake backend; no GPU or API."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import unittest.mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jev_dllm.backends.instrumented import checkpoint_blocks
from jev_dllm.experiments import week2_pool
from jev_dllm.reproducibility import trajectory_seed
from jev_dllm.schema import Candidate, CostRecord, Question


class FakeBackend:
    def __init__(self, fail_after=None, tweak=None):
        self.calls, self.fail_after, self.tweak, self.last_trace = [], fail_after, tweak, None

    def generate(self, question, *, candidate_id, seed):
        if candidate_id != 'warmup':
            if len(self.calls) == self.fail_after:
                raise KeyboardInterrupt('synthetic stop')
            self.calls.append(candidate_id)
        text = f'answer {seed % 997}' + (' changed' if candidate_id == self.tweak else '')
        self.last_trace = {'question_id': question.question_id, 'candidate_id': candidate_id, 'seed': seed,
                           'commit_probability': [0.5], 'checkpoints': []}
        return (Candidate(question.question_id, candidate_id, seed, text, True, False, 0.5),
                CostRecord(trajectory_nfe=3, generation_gpu_seconds=.1, generation_wall_seconds=.1, peak_memory_bytes=1))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Week2PoolTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.write('prompts/p.txt', 'Solve {question}')
        self.base = {'generator': {'selected_backend': 'fast_dllm_v1', 'generation_length': 16, 'block_size': 2,
                                   'candidates_per_question': 2},
                     'prompts': {'generation_math': {'path': 'prompts/p.txt', 'sha256': sha(self.root / 'prompts/p.txt')}},
                     'runtime': {'master_seed': 5}}
        self.write('configs/base.json', json.dumps(self.base))
        questions = [{'question_id': f'gsm8k/train/{i:05}', 'dataset': 'gsm8k', 'split': 'dev', 'prompt': f'Q{i}'} for i in range(3)]
        self.write('data/processed/gsm8k-w2dev.json', json.dumps(questions))
        self.profile = {'schema_version': 1, 'scope': week2_pool.SCOPE, 'task': 'gsm8k',
                        'base_profile': {'path': 'configs/base.json', 'key': None, 'sha256': sha(self.root / 'configs/base.json')},
                        'questions': {'path': 'data/processed/gsm8k-w2dev.json', 'count': 3,
                                      'sha256': sha(self.root / 'data/processed/gsm8k-w2dev.json')},
                        'candidates_per_question': 4, 'checkpoint_completed_blocks': [1, 2, 3]}
        self.write(week2_pool.profile_path('gsm8k'), json.dumps(self.profile))

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def run_pool(self, backend, resume_from=None):
        with redirect_stdout(io.StringIO()):
            return week2_pool.generate(self.root, 'gsm8k', resume_from, backend_factory=lambda: backend,
                                       runtime_check=lambda root: {'gpu_name': 'fake'})

    def rows(self, run, name):
        return [json.loads(line) for line in (run / name).read_text().splitlines()]

    def runs(self):
        base = self.root / 'artifacts/runs'
        return set(base.glob('w2pool-*')) if base.exists() else set()

    def interrupted_run(self, fail_after):
        before = self.runs()
        with self.assertRaises(KeyboardInterrupt):
            self.run_pool(FakeBackend(fail_after=fail_after))
        (source,) = self.runs() - before
        return source

    def test_complete_pool_writes_aligned_records_in_seed_order(self):
        run = self.run_pool(FakeBackend())
        manifest = json.loads((run / 'manifest.json').read_text())
        self.assertEqual((manifest['status'], manifest['completed_candidates']), ('complete', 12))
        candidates = self.rows(run, 'candidates.jsonl')
        expected = [(f'gsm8k/train/{q:05}/c{i:02}', trajectory_seed(5, f'gsm8k/train/{q:05}', i))
                    for q in range(3) for i in range(4)]
        self.assertEqual([(c['candidate_id'], c['seed']) for c in candidates], expected)
        self.assertEqual([t['candidate_id'] for t in self.rows(run, 'trajectories.jsonl')], [e[0] for e in expected])
        self.assertEqual(len(self.rows(run, 'costs.jsonl')), 12)
        self.assertEqual((run / 'pool-profile.json').read_bytes(), (self.root / week2_pool.profile_path('gsm8k')).read_bytes())
        self.assertIsNone(manifest['run_note'])

    def test_run_note_is_recorded(self):
        with unittest.mock.patch.dict('os.environ', {'JEV_RUN_NOTE': 'three tasks share GPU 0'}):
            run = self.run_pool(FakeBackend())
        self.assertEqual(json.loads((run / 'manifest.json').read_text())['run_note'], 'three tasks share GPU 0')

    def test_interrupted_pool_resumes_to_the_same_records(self):
        reference = self.run_pool(FakeBackend())
        source = self.interrupted_run(5)
        self.assertEqual(json.loads((source / 'manifest.json').read_text())['status'], 'interrupted')
        before = {name: (source / name).read_bytes() for name in week2_pool.FILES}
        backend = FakeBackend()
        resumed = self.run_pool(backend, source)
        self.assertEqual(len(backend.calls), 7)  # Only the missing candidates are generated.
        for name in ('candidates.jsonl', 'trajectories.jsonl'):
            self.assertEqual((resumed / name).read_bytes(), (reference / name).read_bytes())
        manifest = json.loads((resumed / 'manifest.json').read_text())
        self.assertEqual((manifest['status'], manifest['resumed_from']['copied_candidates']), ('complete', 5))
        self.assertEqual({name: (source / name).read_bytes() for name in week2_pool.FILES}, before)

    def test_resume_ignores_a_partial_trailing_line(self):
        source = self.interrupted_run(5)
        with (source / 'candidates.jsonl').open('a') as stream:
            stream.write('{"candidate_id": "partial')
        resumed = self.run_pool(FakeBackend(), source)
        self.assertEqual(json.loads((resumed / 'manifest.json').read_text())['resumed_from']['copied_candidates'], 5)

    def test_resume_refuses_complete_changed_or_reordered_sources(self):
        complete = self.run_pool(FakeBackend())
        with self.assertRaisesRegex(ValueError, 'failed or interrupted'):
            self.run_pool(FakeBackend(), complete)
        source = self.interrupted_run(5)
        rows = (source / 'candidates.jsonl').read_text().splitlines()
        (source / 'candidates.jsonl').write_text('\n'.join([rows[1], rows[0]] + rows[2:]) + '\n')
        with self.assertRaisesRegex(ValueError, 'planned order'):
            self.run_pool(FakeBackend(), source)
        self.write(week2_pool.profile_path('gsm8k'), json.dumps(dict(self.profile, candidates_per_question=3)))
        with self.assertRaisesRegex(ValueError, 'different pool profile'):
            self.run_pool(FakeBackend(), source)

    def test_changed_inputs_are_refused_before_generation(self):
        self.write('data/processed/gsm8k-w2dev.json', '[]')
        backend = FakeBackend()
        with self.assertRaisesRegex(ValueError, 'differ from the profile pin'):
            self.run_pool(backend)
        self.assertEqual(backend.calls, [])

    def equivalence_fixture(self, tweak=None):
        self.write(week2_pool.EQUIVALENCE['base_profile'], json.dumps(self.base))
        questions = json.loads((self.root / 'data/processed/gsm8k-w2dev.json').read_text())
        self.write(week2_pool.EQUIVALENCE['questions'], json.dumps(questions))
        reference = self.root / 'artifacts/runs/fast-llada-dev-20261026T000000Z-00000000'
        backend = FakeBackend()
        rows = []
        for q in questions:
            for i in range(2):
                cid, seed = f"{q['question_id']}/c{i:02}", trajectory_seed(5, q['question_id'], i)
                candidate, _ = backend.generate(Question(q['question_id'], 'gsm8k', 'dev', q['prompt']),
                                                candidate_id=cid, seed=seed)
                rows.append(json.dumps(candidate.__dict__))
        self.write(str(reference.relative_to(self.root) / 'candidates.jsonl'), '\n'.join(rows) + '\n')
        self.write(str(reference.relative_to(self.root) / 'config.snapshot.json'), json.dumps(self.base))
        self.write(str(reference.relative_to(self.root) / 'manifest.json'), json.dumps(
            {'status': 'complete', 'backend': 'fast_dllm_v1', 'gpu': 'fake',
             'questions_sha256': sha(self.root / week2_pool.EQUIVALENCE['questions'])}))
        with redirect_stdout(io.StringIO()):
            return week2_pool.equivalence(self.root, reference, backend_factory=lambda: FakeBackend(tweak=tweak),
                                          runtime_check=lambda root: {'gpu_name': 'fake'})

    def test_equivalence_reports_identical_and_different_candidates(self):
        _, report = self.equivalence_fixture()
        self.assertEqual((report['status'], report['compared'], report['identical']), ('identical', 6, 6))
        _, report = self.equivalence_fixture(tweak='gsm8k/train/00001/c01')
        self.assertEqual(report['status'], 'different')
        self.assertEqual(report['mismatches'], [{'candidate_id': 'gsm8k/train/00001/c01', 'fields': ['text']}])

    def test_repository_profiles_pin_week1_settings(self):
        for task in week2_pool.TASKS:
            profile = json.loads((ROOT / week2_pool.profile_path(task)).read_text())
            self.assertEqual(profile['candidates_per_question'], 16)
            self.assertEqual(sha(ROOT / profile['base_profile']['path']), profile['base_profile']['sha256'])
            base = json.loads((ROOT / profile['base_profile']['path']).read_text())
            base = base[profile['base_profile']['key']] if profile['base_profile']['key'] else base
            g = base['generator']
            checkpoint_blocks(g['generation_length'] // g['block_size'], profile['checkpoint_completed_blocks'])
            self.assertEqual(sha(ROOT / base['prompts']['generation_math']['path']), base['prompts']['generation_math']['sha256'])


if __name__ == '__main__':
    unittest.main()
