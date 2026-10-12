"""Explicit one-retry amendment for a failed, validated-prefix GSM8K Jev batch."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

from . import frozen
from .selection import load_pools, new_run, write_json
from ..judges.typesafe import TypeSafeJudge, build_request, parse_response
from ..reproducibility import fingerprint, trajectory_seed


IMPLEMENTATION = ('src/jev_dllm/experiments/jev_continuation.py', 'scripts/continue_jev.py', 'Makefile')


def jsonl(path):
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Partial last JSONL record; retain and inspect the source')
    return [json.loads(line) for line in raw.splitlines()]


def project_path(root, name, directory):
    relative = Path(name)
    path = root / relative
    if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
            or not path.resolve().is_relative_to((root / directory).resolve())
            or any((root / parent).is_symlink() for parent in relative.parents)):
        raise ValueError('Expected an ordinary project artifact path')
    return path


def inspect(root, plan_dir):
    plan_dir = project_path(root, plan_dir.relative_to(root), 'artifacts/runs')
    plan = json.loads((plan_dir / 'plan.json').read_text())
    if plan.get('action') != 'one_manual_retry_then_remaining_requests':
        raise ValueError('Unsupported continuation policy')
    source = project_path(root, plan['source_run'], 'artifacts/runs')
    names = {p.name for p in source.iterdir() if p.is_file()}
    if names != set(plan['source_files_sha256']):
        raise ValueError('Source file inventory changed')
    for name, expected in plan['source_files_sha256'].items():
        if Path(name).name != name or (source / name).is_symlink() or frozen.sha256(source / name) != expected:
            raise ValueError('Original batch changed; do not retry it with this plan')
    if frozen.sha256(plan_dir / 'planned_requests.jsonl') != plan['planned_requests_sha256']:
        raise ValueError('Continuation request plan changed')
    config = json.loads((root / frozen.PROFILE).read_text())
    archive = project_path(root, 'artifacts/downloads/lab/' + plan['generation_run'], 'artifacts/downloads/lab')
    questions_path = root / config['datasets']['questions_path']
    protocol = frozen.check_pool(root, config, questions_path, archive / 'candidates.jsonl')
    m = json.loads((source / 'manifest.json').read_text())
    requests, accepted = jsonl(source / 'requests.jsonl'), jsonl(source / 'selections.jsonl')
    n = len(accepted)
    if (m.get('mode') != 'live_jev' or m.get('status') != 'failed'
            or m.get('protocol') != protocol or plan['frozen_protocol'] != protocol
            or m.get('model') != config['judge']['model_version'] or plan['model'] != m['model']
            or m.get('master_seed') != config['runtime']['master_seed']
            or m.get('question_count') != 500 or len(requests) != 500
            or not 0 <= n < 500 or m.get('successful_calls') != n
            or m.get('attempted_calls') != n + 1
            or plan.get('saved_accepted_selections') != n
            or plan.get('already_attempted_requests') != n + 1
            or plan.get('planned_additional_paid_requests') != 500 - n
            or plan.get('remaining_unattempted_questions') != 499 - n
            or plan.get('total_attempts_if_successful') != 501
            or plan.get('explicit_retries_in_this_plan') != 1
            or plan.get('additional_automatic_retries') != 0):
        raise ValueError('Batch or planned call counts do not match the explicit one-retry amendment')
    questions, pools = load_pools(questions_path, archive / 'candidates.jsonl')
    prompt = (root / config['prompts']['jev_final']['path']).read_text()
    if (m.get('questions_sha256') != fingerprint([asdict(q) for q in questions])
            or m.get('pools_sha256') != fingerprint({k:[asdict(c) for c in v] for k,v in pools.items()})
            or m.get('prompt_sha256') != fingerprint(prompt)):
        raise ValueError('Original batch does not describe these exact public inputs')
    for q, row in zip(questions, requests):
        candidates = list(pools[q.question_id])
        random.Random(trajectory_seed(config['runtime']['master_seed'], q.question_id, 0)).shuffle(candidates)
        if row != {'question_id':q.question_id, 'body':build_request(q, candidates, prompt, m['model'])}:
            raise ValueError('Request payload differs from the frozen public-only request')
    for row, request in zip(accepted, requests):
        parsed = parse_response(row['raw_response'], request['body'])
        if row['question_id'] != request['question_id'] or any(row.get(k) != v for k,v in parsed.items()):
            raise ValueError('Saved accepted selection does not match its response/request')
    failure = json.loads((source / 'failed_response.json').read_text())
    if not (failure['question_id'] == m.get('failed_question_id') == plan.get('failed_question_id') == requests[n]['question_id']):
        raise ValueError('Failure position does not follow the saved prefix')
    try:
        parse_response(json.loads(failure['raw_response_json']), requests[n]['body'])
    except ValueError as exc:
        if str(exc) != 'Jev choice is inconsistent with its probabilities':
            raise ValueError('This plan handles only the recorded choice/probability mismatch') from exc
    else:
        raise ValueError('The recorded failure no longer reproduces')
    if jsonl(plan_dir / 'planned_requests.jsonl') != requests[n:]:
        raise ValueError('Planned requests do not exactly cover the failed and unattempted suffix')
    return plan, source, m, requests, accepted


def execute(root, plan_dir, approved_calls, client=None):
    plan, source, original, requests, accepted = inspect(root, plan_dir)
    if type(approved_calls) is not int or approved_calls != plan['planned_additional_paid_requests']:
        raise ValueError('Explicit approval count must equal this exact planned batch')
    claims = root / 'artifacts/jev-continuation-claims'
    claims.mkdir(parents=True, exist_ok=True)
    # A failed/interrupted continuation requires investigation and a new decision.
    # Never replay this plan blindly, even when an earlier attempt sent no calls.
    claim_path = claims / (plan_dir.name + '.json')
    with claim_path.open('x') as claim:
        claim.write(json.dumps({'plan_sha256':frozen.sha256(plan_dir/'plan.json'),
                                'created_at_utc':datetime.now(timezone.utc).isoformat()}) + '\n')
    run = new_run(root / 'artifacts/runs', 'jev-continuation')
    write_json(claim_path, {'plan_sha256':frozen.sha256(plan_dir/'plan.json'),'run_id':run.name})
    print('Continuation directory: ' + str(run), flush=True)
    m = dict(original)
    for key in ('error_type', 'failed_question_id'):
        m.pop(key, None)
    amendment = {'kind':'one_explicit_retry_then_unattempted_suffix',
        'source_run':source.name, 'source_manifest_sha256':frozen.sha256(source/'manifest.json'),
        'source_selections_sha256':frozen.sha256(source/'selections.jsonl'),
        'source_failure_sha256':frozen.sha256(source/'failed_response.json'),
        'plan_sha256':frozen.sha256(plan_dir/'plan.json'),
        'implementation_sha256':{p:frozen.sha256(root/p) for p in IMPLEMENTATION},
        'prior_attempted_calls':original['attempted_calls'], 'reused_selections':len(accepted),
        'approved_additional_calls':approved_calls, 'retry_question_id':plan['failed_question_id'],
        'max_automatic_retries':0, 'unchanged_generation_and_selection_settings':True,
        'authorization':'User explicitly approved this one-retry continuation and its exact additional call count.',
        'note':'Amended execution, not an uninterrupted completion of the original no-retry batch.'}
    m.update(status='running', successful_calls=len(accepted), attempted_calls=0,
             newly_successful_calls=0, total_attempted_calls=original['attempted_calls'],
             api_cost_usd=None, retry_amendment=amendment,
             note='successful_calls includes reused selections; attempted_calls counts network calls in this segment only.')
    write_json(run/'manifest.json',m)
    try:
        shutil.copyfile(source/'requests.jsonl',run/'requests.jsonl')
        shutil.copyfile(source/'selections.jsonl',run/'selections.jsonl')
        shutil.copyfile(source/'failed_response.json',run/'prior_failed_response.json')
        shutil.copyfile(source/'manifest.json',run/'prior_manifest.json')
        shutil.copyfile(plan_dir/'plan.json',run/'continuation-plan.json')
        client = client or TypeSafeJudge()
        with (run/'selections.jsonl').open('a') as stream:
            for row in requests[len(accepted):]:
                m['attempted_calls'] += 1
                m['total_attempted_calls'] += 1
                write_json(run/'manifest.json',m)
                result = client.select(row['body'])
                parsed = parse_response(result['raw_response'],row['body'])
                if any(result.get(k) != v for k,v in parsed.items()):
                    raise ValueError('Transport result differs from its validated raw response')
                stream.write(json.dumps({'question_id':row['question_id'],**result},allow_nan=False)+'\n')
                stream.flush()
                m['successful_calls'] += 1
                m['newly_successful_calls'] += 1
                write_json(run/'manifest.json',m)
                print(f"Jev selections saved: {m['successful_calls']}/500; new calls: {m['attempted_calls']}/{approved_calls}",flush=True)
        m['status']='complete'
    except BaseException as exc:
        m.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error_type=type(exc).__name__)
        if 'row' in locals():
            m['failed_question_id']=row['question_id']
        if hasattr(exc,'raw_response'):
            write_json(run/'failed_response.json',{'question_id':row['question_id'],
                'raw_response_json':json.dumps(exc.raw_response,ensure_ascii=False), 'request_seconds':exc.request_seconds})
        raise
    finally:
        write_json(run/'manifest.json',m)
    if not (run/'selections.jsonl').read_bytes().startswith((source/'selections.jsonl').read_bytes()):
        raise RuntimeError('Preserved selection bytes differ')
    return run
