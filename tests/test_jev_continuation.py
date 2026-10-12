"""Synthetic continuation coverage: no paid calls, no credentials, no private data."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import random
import shutil
import unittest
import test_gsm8k500 as fixtures
from jev_dllm.experiments import frozen, jev_continuation as extension
from jev_dllm.experiments.selection import load_pools
from jev_dllm.judges.typesafe import build_request, parse_response
from jev_dllm.reproducibility import fingerprint, trajectory_seed

ROOT=Path(__file__).resolve().parents[1]


def response(request, bad=False):
    keys=list(request['questions']['best_answer']['criteria'])
    return {'model':request['model'], 'answers':{'best_answer':{'type':'choice',
        'choice':keys[1] if bad else keys[0], 'probabilities':{k:float(i==0) for i,k in enumerate(keys)},
        'confidence':1.0}},'usage':{'input_tokens':100,'output_tokens':20}}


def result(request):
    raw=response(request)
    return {**parse_response(raw,request),'raw_response':raw,'request_seconds':.1,'api_cost_usd':None}


class FakeClient:
    def __init__(self, failure=None):
        self.calls=[]
        self.failure=failure

    def select(self, request):
        self.calls.append(request)
        if self.failure:
            if self.failure is KeyboardInterrupt:
                raise KeyboardInterrupt
            e=ValueError('Jev choice is inconsistent with its probabilities')
            e.raw_response=response(request,bad=True)
            e.request_seconds=.1
            raise e
        return result(request)


class JevContinuationTests(unittest.TestCase):
    def setUp(self):
        fixtures.FrozenGSM8KTests.setUp(self)
        # Reuse only synthetic archive builders, not real saved experiment answers.
        self.check=lambda:frozen.check_protocol(self.root,self.config,self.questions)
        self.receipt=lambda run:fixtures.FrozenGSM8KTests.receipt(self,run)
        archive,_=fixtures.FrozenGSM8KTests.archive(self)
        for name in extension.IMPLEMENTATION:
            p=self.root/name
            p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,p)
        questions,pools=load_pools(self.questions,archive/'candidates.jsonl')
        prompt=(self.root/self.config['prompts']['jev_final']['path']).read_text()
        requests=[]
        for q in questions:
            pool=list(pools[q.question_id])
            random.Random(trajectory_seed(self.config['runtime']['master_seed'],q.question_id,0)).shuffle(pool)
            requests.append({'question_id':q.question_id,'body':build_request(q,pool,prompt,self.config['judge']['model_version'])})
        self.source=self.root/'artifacts/runs/jev-live-fixture'
        self.source.mkdir(parents=True)
        self.save_rows(self.source/'requests.jsonl',requests)
        self.save_rows(self.source/'selections.jsonl',[{'question_id':r['question_id'],**result(r['body'])} for r in requests[:498]])
        m={'mode':'live_jev','status':'failed','model':self.config['judge']['model_version'],
            'master_seed':self.config['runtime']['master_seed'],'protocol':self.check(),
            'question_count':500,'attempted_calls':499,'successful_calls':498,
            'failed_question_id':requests[498]['question_id'],'error_type':'ValueError',
            'questions_sha256':fingerprint([asdict(q) for q in questions]),
            'pools_sha256':fingerprint({k:[asdict(c) for c in v] for k,v in pools.items()}),
            'prompt_sha256':fingerprint(prompt)}
        self.save(self.source/'manifest.json',m)
        self.save(self.source/'failed_response.json',{'question_id':requests[498]['question_id'],
            'raw_response_json':json.dumps(response(requests[498]['body'],bad=True)),'request_seconds':.1})
        self.plan=self.root/'artifacts/runs/jev-continuation-plan-fixture'
        self.plan.mkdir()
        self.save_rows(self.plan/'planned_requests.jsonl',requests[498:])
        p={'action':'one_manual_retry_then_remaining_requests','source_run':str(self.source.relative_to(self.root)),
            'generation_run':archive.name,'frozen_protocol':m['protocol'],'model':m['model'],
            'saved_accepted_selections':498,'already_attempted_requests':499,'failed_question_id':m['failed_question_id'],
            'remaining_unattempted_questions':1,'planned_additional_paid_requests':2,'total_attempts_if_successful':501,
            'explicit_retries_in_this_plan':1,'additional_automatic_retries':0,
            'planned_requests_sha256':frozen.sha256(self.plan/'planned_requests.jsonl')}
        self.save(self.plan/'plan.json',p)
        self.source_hashes()

    def save(self,path,value):
        path.write_text(json.dumps(value))

    def save_rows(self,path,rows):
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def source_hashes(self):
        p=json.loads((self.plan/'plan.json').read_text())
        p['source_files_sha256']={f.name:frozen.sha256(f) for f in self.source.iterdir()}
        self.save(self.plan/'plan.json',p)

    def execute(self,client,approved=2):
        with redirect_stdout(io.StringIO()):
            return extension.execute(self.root,self.plan,approved,client)

    def test_inspection_uses_public_inputs_only_and_no_calls(self):
        p,_,_,reqs,accepted=extension.inspect(self.root,self.plan)
        self.assertEqual((len(reqs),len(accepted),p['planned_additional_paid_requests']),(500,498,2))
        self.assertFalse((self.root/'data/private').exists())
        self.assertFalse((self.root/'artifacts/jev-continuation-claims').exists())

    def test_reuse_exact_prefix_and_send_only_suffix_with_amendment(self):
        before={p.name:p.read_bytes() for p in self.source.iterdir()}
        client=FakeClient()
        run=self.execute(client)
        self.assertEqual(client.calls,[r['body'] for r in extension.jsonl(self.plan/'planned_requests.jsonl')])
        self.assertTrue((run/'selections.jsonl').read_bytes().startswith(before['selections.jsonl']))
        self.assertEqual(len(extension.jsonl(run/'selections.jsonl')),500)
        m=json.loads((run/'manifest.json').read_text())
        self.assertEqual((m['status'],m['successful_calls'],m['attempted_calls'],m['total_attempted_calls']),('complete',500,2,501))
        self.assertEqual(m['retry_amendment']['reused_selections'],498)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.source.iterdir()})

    def test_duplicate_plan_cannot_charge_again(self):
        self.execute(FakeClient())
        client=FakeClient()
        with self.assertRaises(FileExistsError):
            self.execute(client)
        self.assertEqual(client.calls,[])

    def test_wrong_approval_or_source_change_stops_before_calls(self):
        client=FakeClient()
        with self.assertRaisesRegex(ValueError,'approval count'):
            self.execute(client,1)
        with (self.source/'selections.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError,'Original batch changed'):
            self.execute(client)
        self.assertEqual(client.calls,[])

    def test_rehashed_request_with_private_field_is_rejected(self):
        rows=extension.jsonl(self.source/'requests.jsonl')
        rows[0]['body']['state']['reference']='secret'
        self.save_rows(self.source/'requests.jsonl',rows)
        self.source_hashes()
        client=FakeClient()
        with self.assertRaisesRegex(ValueError,'public-only request'):
            self.execute(client)
        self.assertEqual(client.calls,[])

    def test_rehashed_selection_identity_mismatch_is_rejected(self):
        rows=extension.jsonl(self.source/'selections.jsonl')
        rows[0]['candidate_id']='wrong'
        self.save_rows(self.source/'selections.jsonl',rows)
        self.source_hashes()
        with self.assertRaisesRegex(ValueError,'Saved accepted selection'):
            extension.inspect(self.root,self.plan)

    def test_new_error_retains_evidence_and_never_retries(self):
        client=FakeClient(ValueError)
        with self.assertRaises(ValueError):
            self.execute(client)
        claim=json.loads((self.root/'artifacts/jev-continuation-claims'/f'{self.plan.name}.json').read_text())
        run=self.root/'artifacts/runs'/claim['run_id']
        m=json.loads((run/'manifest.json').read_text())
        self.assertEqual((m['status'],m['attempted_calls'],m['successful_calls']),('failed',1,498))
        self.assertEqual(len(client.calls),1)
        self.assertTrue((run/'failed_response.json').exists())
        self.assertEqual((run/'selections.jsonl').read_bytes(),(self.source/'selections.jsonl').read_bytes())

    def test_interrupt_retains_prefix_and_terminal_manifest(self):
        client=FakeClient(KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            self.execute(client)
        claim=json.loads((self.root/'artifacts/jev-continuation-claims'/f'{self.plan.name}.json').read_text())
        run=self.root/'artifacts/runs'/claim['run_id']
        self.assertEqual(json.loads((run/'manifest.json').read_text())['status'],'interrupted')
        self.assertEqual((run/'selections.jsonl').read_bytes(),(self.source/'selections.jsonl').read_bytes())


if __name__=='__main__':
    unittest.main()
