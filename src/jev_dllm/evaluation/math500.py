"""Frozen offline MATH500 scoring of verified pools and saved Jev selections."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from ..experiments import math500_jev_resume as selection
from ..experiments.selection import new_run, paired_summary, write_json
from ..reproducibility import trajectory_seed
from ..selection.baselines import random_choice

POLICY = 'configs/evaluation/math500-math-verify-v1.json'
CONTROLS = 'tests/fixtures/math500-controls.json'
WORKER = 'src/jev_dllm/evaluation/math500_worker.py'
FILES = (POLICY, CONTROLS, WORKER, 'src/jev_dllm/evaluation/math500.py',
         'scripts/score_math500.py', 'Makefile', 'requirements/math500-cpu.lock')
sha256 = selection.sha256


def read(path):
    return json.loads(path.read_text())


def implementation(root):
    if any((root / name).is_symlink() for name in FILES):
        raise ValueError('Scoring implementation files must not be symlinks')
    return {name: sha256(root / name) for name in FILES}


def worker(root, python, action, policy, **kwargs):
    request = {'action': action, 'policy': policy, **kwargs}
    # Do not inherit API credentials, HF tokens or project settings into math parsing.
    env = {'PATH': os.defpath, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}
    with tempfile.TemporaryDirectory(prefix='jev-math-') as work:
        try:
            process = subprocess.run([str(python), str(root / WORKER)], input=json.dumps(request),
                text=True, capture_output=True, cwd=work, env=env,
                timeout=policy['question_timeout_seconds'], check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Math worker exceeded the frozen wall-time limit; no score was invented') from exc
    if process.returncode:
        raise RuntimeError(f'Math worker failed with status {process.returncode}: {process.stderr[-1500:]}')
    return json.loads(process.stdout)


def validate(root, python):
    policy = read(root / POLICY)
    controls = read(root / CONTROLS)
    out = new_run(root / 'artifacts/runs', 'math500-validation')
    result = worker(root, python, 'validate', policy, controls=controls)
    write_json(out / 'controls.json', result)
    manifest = {'status': result['status'], 'independent_control_count': len(result['checks']),
                'implementation_sha256': implementation(root), 'runtime': result['runtime'],
                'controls_sha256': sha256(out / 'controls.json'),
                'heldout_references_read': False, 'heldout_candidates_scored': False, 'new_api_calls': 0}
    write_json(out / 'manifest.json', manifest)
    if result['status'] != 'passed':
        raise ValueError(f'Independent scorer controls failed; inspect {out}')
    return out


def context(root, archive, jev_plan):
    # Includes the complete generation/recovery and API retry ancestry audits.
    amendment, parent, prior = selection.inspect(root, jev_plan)
    info = selection._check_segment(jev_plan, amendment, parent, prior, failed=False)
    questions, pools = selection.original.public_inputs(root, archive)
    if questions != info['questions'] or pools != info['pools']:
        raise ValueError('Jev selection and supplied generation archive must describe exactly the same pool')
    return info


def prepare(root, python, archive, jev_plan, validation):
    policy = read(root / POLICY)
    v = read(validation / 'manifest.json')
    controls = read(validation / 'controls.json')
    runtime = worker(root, python, 'runtime', policy)
    if (v['status'] != 'passed' or controls['status'] != 'passed'
            or len(controls['checks']) != v['independent_control_count']
            or len({c['case_id'] for c in controls['checks']}) != len(controls['checks'])
            or not all(c['passed'] is True for c in controls['checks'])
            or v['independent_control_count'] != len(read(root / CONTROLS)) + 5
            or v['controls_sha256'] != sha256(validation / 'controls.json')
            or v['runtime'] != runtime or v['implementation_sha256'] != implementation(root)):
        raise ValueError('Require passing independent controls for this exact policy, implementation and runtime')
    info = context(root, archive, jev_plan)
    reference = root / 'data/private/math500-heldout-references.json'
    manifest = read(root / 'data/manifests/math500-week01.json')
    if sha256(reference) != manifest['files']['data/private/math500-heldout-references.json']:
        raise ValueError('Pinned MATH500 reference hash changed')
    out = new_run(root / 'artifacts/runs', 'math500-scoring-plan')
    write_json(out / 'plan.json', {'version': policy['version'], 'policy': policy,
        'implementation_sha256': implementation(root), 'runtime': runtime,
        'archive': str(archive.resolve()), 'archive_files_sha256': selection.previous.inventory(archive),
        'jev_plan': str(jev_plan.resolve()), 'jev_files_sha256': selection.previous.inventory(jev_plan),
        'references_sha256': sha256(reference), 'validation': str(validation.resolve()),
        'validation_files_sha256': selection.previous.inventory(validation),
        'question_count': len(info['questions']), 'candidate_count': sum(map(len, info['pools'].values())),
        'new_api_calls': 0, 'frozen_before_heldout_correctness_scoring': True,
        'note': 'Scorer choices were tested on independent handwritten controls. Generation and Jev selection predate this scoring freeze.'})
    return out


def inspect(root, python, plan):
    p = read(plan / 'plan.json')
    if p['implementation_sha256'] != implementation(root) or p['policy'] != read(root / POLICY):
        raise ValueError('Frozen mathematical scoring implementation or policy changed')
    if p['runtime'] != worker(root, python, 'runtime', p['policy']):
        raise ValueError('Frozen mathematical runtime changed')
    for source, hashes in (('archive', 'archive_files_sha256'), ('jev_plan', 'jev_files_sha256'), ('validation', 'validation_files_sha256')):
        if selection.previous.inventory(Path(p[source])) != p[hashes]:
            raise ValueError('Frozen scoring source evidence changed')
    if sha256(root / 'data/private/math500-heldout-references.json') != p['references_sha256']:
        raise ValueError('Frozen reference answers changed')
    info = context(root, Path(p['archive']), Path(p['jev_plan']))
    if (p['version'] != p['policy']['version'] or p['question_count'] != len(info['questions'])
            or p['candidate_count'] != sum(map(len, info['pools'].values()))):
        raise ValueError('Frozen scoring scope changed')
    return p, info


def summarize(info, records, policy):
    questions, pools, accepted = info['questions'], info['pools'], info['accepted']
    chosen = {r['question_id']: r['candidate_id'] for r in accepted}
    if len(chosen) != len(accepted) or set(chosen) != set(pools) or len(records) != len(questions):
        raise ValueError('Require exactly one complete selection and score record per question')
    outcomes = {name: [] for name in ('first_sample', 'seeded_random', 'majority_vote', 'jev')}
    details, count_histogram, states, verification_states = [], Counter(), Counter(), Counter()
    correct_total = candidates = 0
    for q, record in zip(questions, records):
        pool = pools[q.question_id]
        if record['question_id'] != q.question_id or [r['candidate_id'] for r in record['candidates']] != [c.candidate_id for c in pool]:
            raise ValueError('Scored candidates differ from the exact ordered source pool')
        if any(type(r['correct']) is not bool for r in record['candidates']):
            raise ValueError('Correctness labels must be explicit booleans')
        correct = {r['candidate_id']: r['correct'] for r in record['candidates']}
        ids = {'first_sample':pool[0].candidate_id,
               'seeded_random':random_choice(pool, trajectory_seed(policy['seed'], q.question_id, 1)),
               'majority_vote':record['public_selection']['majority_candidate_id'], 'jev':chosen[q.question_id]}
        if any(cid not in correct for cid in ids.values()):
            raise ValueError('Selected answer is outside the source pool')
        for method, cid in ids.items():
            outcomes[method].append(correct[cid])
        solved = sum(correct.values())
        count_histogram[solved] += 1
        correct_total += solved
        candidates += len(pool)
        states.update(r['extraction_status'] for r in record['candidates'])
        verification_states.update(r['verification_status'] for r in record['candidates'])
        details.append({'question_id':q.question_id, 'selected_ids':ids, 'correct_by_candidate':correct,
                        'majority_groups':record['public_selection']['groups']})
    metrics = {'status':'complete', 'scorer_version':policy['version'], 'question_count':len(questions),
        'candidate_count':candidates, 'passed_candidates':correct_total,
        'mean_single_sample_accuracy':correct_total/candidates,
        'accuracy':{k:sum(v)/len(v) for k,v in outcomes.items()}, 'correct_counts':{k:sum(v) for k,v in outcomes.items()},
        'empirical_oracle_at_pool_size':1-count_histogram[0]/len(questions), 'solvable_pools':len(questions)-count_histogram[0],
        'correct_candidate_histogram':dict(sorted(count_histogram.items())),
        'mean_unique_text_fraction':sum(len({c.text for c in pool})/len(pool) for pool in pools.values())/len(pools),
        'extraction_status_counts':dict(states), 'verification_status_counts':dict(verification_states),
        'unit_of_analysis':'question', 'is_benchmark_result':False,
        'analysis_type':'post_collection_fixed_scorer_with_explicit_jev_retry_amendments',
        'note':'Same original pools for all selectors. Scoring was frozen after collection/selection but before correctness analysis. Private labels never enter majority voting or Jev.'}
    for method in ('majority_vote','first_sample','seeded_random'):
        metrics['jev_vs_'+method] = paired_summary(outcomes['jev'], outcomes[method], policy['seed'])
    failures = info['failures']
    usages = [r['usage'] for r in accepted] + [json.loads(f['response']['raw_response_json'])['usage'] for f in failures]
    metrics['jev_execution'] = {'accepted':len(accepted), 'total_attempted_calls':info['total_attempts'],
        'rejected_responses':len(failures), 'explicit_retries':len(failures), 'automatic_retries':0,
        'usage_including_rejected':{k:sum(u[k] for u in usages) for k in ('input_tokens','output_tokens')},
        'request_seconds_sum_including_rejected':sum(r['request_seconds'] for r in accepted)+sum(f['response']['request_seconds'] for f in failures),
        'api_cost_usd':None, 'new_api_calls':0}
    return metrics, details


def evaluate(root, python, plan):
    p, info = inspect(root, python, plan)
    refs = read(root / 'data/private/math500-heldout-references.json')
    references = {r['question_id']:r['answer'] for r in refs}
    if len(references) != len(refs) or set(references) != set(info['pools']):
        raise ValueError('References must cover exactly the saved questions')
    out = new_run(root / 'artifacts/runs', 'math500-evaluation')
    (out / 'evaluation_private').mkdir()
    m = {'status':'running', 'plan':str(plan.resolve()), 'plan_sha256':sha256(plan/'plan.json'),
         'policy':p['policy'], 'implementation_sha256':implementation(root), 'runtime':p['runtime'],
         'completed_questions':0, 'new_api_calls':0, 'new_generation':False}
    write_json(out/'manifest.json', m)
    print(f'MATH500 evaluation directory: {out}', flush=True)
    started = time.monotonic()
    records = []
    try:
        control = worker(root, python, 'reference-controls', p['policy'], references=refs)
        write_json(out/'evaluation_private/reference-controls.json',control)
        if control['status'] != 'passed':
            raise ValueError('Reference self/negative controls failed; inspect before scoring')
        def score(q):
            candidates = [{k:asdict(c)[k] for k in ('candidate_id','text','completed','truncated')} for c in info['pools'][q.question_id]]
            question = {'question_id':q.question_id,'candidates':candidates,'reference':references[q.question_id]}
            return worker(root, python, 'question', p['policy'], question=question)
        with ThreadPoolExecutor(max_workers=p['policy']['workers']) as executor, (out/'evaluation_private/question-scores.jsonl').open('x') as stream:
            for record in executor.map(score, info['questions']):
                records.append(record)
                stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
                m['completed_questions'] = len(records)
                write_json(out/'manifest.json',m)
                if len(records)%10 == 0 or len(records)==len(info['questions']):
                    print(f'MATH500 scoring: {len(records)}/{len(info["questions"])} questions',flush=True)
        metrics, details = summarize(info, records, p['policy'])
        inspect(root, python, plan)
        write_json(out/'metrics.json',metrics)
        write_json(out/'evaluation_private/question-results.json',details)
        m.update(status='complete', elapsed_seconds=time.monotonic()-started,
            outputs_sha256={str(f.relative_to(out)):sha256(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='manifest.json'})
    except BaseException as exc:
        m.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error_type=type(exc).__name__)
        raise
    finally:
        write_json(out/'manifest.json',m)
    return out
