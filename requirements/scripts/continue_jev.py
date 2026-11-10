"""Inspect or explicitly execute an approved Jev continuation plan."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jev_dllm.experiments.jev_continuation import inspect, execute
from jev_dllm.judges.credentials import load_api_key


def main():
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--live',action='store_true')
    p.add_argument('--approved-calls',type=int)
    p.add_argument('--env-file',type=Path,default=Path('.env'))
    a=p.parse_args()
    root=a.root.resolve()
    plan=root/a.plan
    try:
        info, *_ = inspect(root,plan)
        if a.live:
            if a.approved_calls != info['planned_additional_paid_requests']:
                raise ValueError('Set the exact user-approved additional call count')
            load_api_key(root/a.env_file)
            result={'run':str(execute(root,plan,a.approved_calls))}
        else:
            result={'status':'ready_for_explicit_approval','new_calls':info['planned_additional_paid_requests'],
                    'reused_selections':info['saved_accepted_selections'],'network_calls':0}
        print(json.dumps(result,indent=2))
        return 0
    except KeyboardInterrupt:
        print('Stopped; partial outputs retained. Do not blindly repeat this plan.',file=sys.stderr)
        return 130
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as e:
        print(f'Continuation stopped: {e}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
