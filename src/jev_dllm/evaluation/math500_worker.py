"""CPU-only symbolic math worker. Parses text; never executes generated Python."""
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import sys

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

logging.disable(logging.CRITICAL)


def runtime(policy):
    installed = {name: importlib.metadata.version(name) for name in policy['packages']}
    if installed != policy['packages'] or '.'.join(platform.python_version_tuple()[:2]) != policy['python_major_minor']:
        raise ValueError('Mathematical scoring runtime differs from the pinned policy')
    # Hash installed source/data instead of assuming a version string proves identity.
    hashes = {}
    for name in installed:
        dist = importlib.metadata.distribution(name)
        entries = {}
        for item in dist.files or []:
            if '..' not in item.parts and item.suffix not in ('.pyc', '.pyo') and item.name != 'RECORD':
                path = Path(dist.locate_file(item))
                if path.is_file():
                    entries[str(item)] = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[name] = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {'python': platform.python_version(), 'packages': installed, 'installed_content_sha256': hashes}


def extract(text, policy, *, reference=False):
    if not isinstance(text, str) or len(text) > policy['max_text_characters']:
        return [], 'input_limit'
    try:
        settings = policy['prediction_extraction']
        configs = [LatexExtractionConfig(boxed_match_priority=settings['latex_boxed_priority'],
                                         try_extract_without_anchor=settings['try_extract_without_anchor'])]
        if reference:
            text = '$' + text + '$'
        else:
            configs.append(ExprExtractionConfig(try_extract_without_anchor=settings['try_extract_without_anchor']))
        values = parse(text, extraction_config=configs, extraction_mode=settings['extraction_mode'],
                       fallback_mode=settings['fallback_mode'], parsing_timeout=policy['parsing_timeout_seconds'], raise_on_error=True)
    except Exception as exc:
        return [], 'parse_' + type(exc).__name__
    return values, ('symbolic' if any(not isinstance(v, str) for v in values) else 'literal_fallback' if values else 'no_match')


def equivalent(gold, prediction, policy):
    if not gold or not prediction:
        return False, 'unextracted'
    try:
        return bool(verify(gold, prediction, **policy['verification'], raise_on_error=True)), 'checked'
    except Exception as exc:
        return False, 'verify_' + type(exc).__name__


def public_analysis(candidates, policy):
    """This phase has no reference parameter and must precede private scoring."""
    if not candidates or len({c['candidate_id'] for c in candidates}) != len(candidates):
        raise ValueError('Require unique ordered candidate IDs')
    parsed, statuses = [], []
    for c in candidates:
        if not c['completed'] or c['truncated'] or not c['text'].strip():
            values, status = [], 'invalid_candidate'
        else:
            values, status = extract(c['text'], policy)
        parsed.append(values)
        statuses.append(status)
    comparisons = []

    @lru_cache(maxsize=None)
    def same(i, j):
        forward, fs = equivalent(parsed[i], parsed[j], policy)
        backward, bs = equivalent(parsed[j], parsed[i], policy)
        comparisons.append({'left': i, 'right': j, 'forward': fs, 'backward': bs, 'equivalent': forward and backward})
        return forward and backward

    groups = []
    for i, values in enumerate(parsed):
        if not values:
            continue
        for group in groups:
            if all(same(i, j) for j in group):
                group.append(i)
                break
        else:
            groups.append([i])
    winner = max(groups, key=len)[0] if groups else 0
    return parsed, {'extraction_statuses': statuses, 'parsed_values': [[str(v) for v in p] for p in parsed],
                    'groups': groups, 'majority_candidate_id': candidates[winner]['candidate_id'],
                    'majority_support': max(map(len, groups), default=0), 'comparisons': comparisons}


def grade_question(request, policy):
    # Compute the deployed baseline entirely before reading the private reference.
    parsed, public = public_analysis(request['candidates'], policy)
    gold, gold_status = extract(request['reference'], policy, reference=True)
    if not gold:
        raise ValueError('Reference answer cannot be extracted; stop instead of inventing labels')
    records = []
    for c, prediction, status, shown in zip(request['candidates'], parsed, public['extraction_statuses'], public['parsed_values']):
        correct, verification = equivalent(gold, prediction, policy)
        records.append({'candidate_id': c['candidate_id'], 'correct': correct,
                        'extraction_status': status, 'parsed': shown, 'verification_status': verification})
    return {'question_id': request['question_id'], 'reference_status': gold_status,
            'candidates': records, 'public_selection': public}


def validate(controls, policy):
    checks = []
    for case in controls:
        gold, _ = extract(case['reference'], policy, reference=True)
        predicted, state = extract(case['prediction'], policy)
        result, verification = equivalent(gold, predicted, policy)
        checks.append({'case_id': case['case_id'], 'passed': result == case['expected_correct'],
                       'expected': case['expected_correct'], 'observed': result,
                       'extraction_status': state, 'verification_status': verification})
    # Handwritten equivalent-answer and tie cases are independent of held-out pools.
    candidate = lambda i, text, completed=True: {'candidate_id': f'fixture/c{i:02}', 'text': text, 'completed': completed, 'truncated': False}
    majority_cases = [
        ('equivalent_fraction_vote', [candidate(0,r'\boxed{2}'),candidate(1,r'\boxed{\frac{1}{2}}'),candidate(2,r'\boxed{0.5}')], 'fixture/c01'),
        ('tie_first', [candidate(0,r'\boxed{2}'),candidate(1,r'\boxed{3}')], 'fixture/c00'),
        ('invalid_does_not_vote', [candidate(0,r'\boxed{2}',False),candidate(1,r'\boxed{3}')], 'fixture/c01'),
        ('no_votes_first', [candidate(0,'No answer'),candidate(1,'')], 'fixture/c00'),
    ]
    for name, candidates, expected in majority_cases:
        _, public = public_analysis(candidates, policy)
        checks.append({'case_id': name, 'passed': public['majority_candidate_id'] == expected})
    sample = majority_cases[0][1]
    a = grade_question({'question_id':'fixture','candidates':sample,'reference':'2'},policy)
    b = grade_question({'question_id':'fixture','candidates':sample,'reference':r'\frac{1}{2}'},policy)
    checks.append({'case_id':'reference_does_not_change_vote','passed':a['public_selection']==b['public_selection']})
    return {'status': 'passed' if all(c['passed'] for c in checks) else 'failed', 'checks':checks, 'runtime':runtime(policy)}


def main():
    request = json.load(sys.stdin)
    action, policy = request['action'], request['policy']
    if action == 'runtime':
        result = runtime(policy)
    elif action == 'validate':
        result = validate(request['controls'], policy)
    elif action == 'reference-controls':
        records = []
        for item in request['references']:
            gold, state = extract(item['answer'], policy, reference=True)
            ok, _ = equivalent(gold, gold, policy)
            wrong, _ = extract(r'\boxed{\text{definitely_invalid_reference_control}}', policy)
            negative, _ = equivalent(gold, wrong, policy)
            records.append({'question_id':item['question_id'], 'reference_status':state,
                            'self_match':ok, 'negative_rejected':not negative})
        result = {'status':'passed' if all(r['self_match'] and r['negative_rejected'] for r in records) else 'failed', 'records':records}
    elif action == 'question':
        result = grade_question(request['question'], policy)
    else:
        raise ValueError('Unknown offline math worker action')
    json.dump(result, sys.stdout, allow_nan=False)
    print()


if __name__ == '__main__':
    main()
