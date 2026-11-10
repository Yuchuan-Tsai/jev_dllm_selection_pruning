"""Validate, freeze or run offline symbolic scoring of the saved MATH500 pool."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jev_dllm.evaluation import math500 as w


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('validate','prepare','check','evaluate'))
    parser.add_argument('--math-python',type=Path,default=Path('.venv-math/bin/python'))
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--jev-plan',type=Path)
    parser.add_argument('--validation',type=Path)
    parser.add_argument('--plan',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    absolute=lambda p:(root/p).absolute() if p else None
    python=absolute(args.math_python)
    try:
        if args.action=='validate':
            out=w.validate(root,python)
        elif args.action=='prepare':
            if not all((args.run_dir,args.jev_plan,args.validation)):
                parser.error('prepare requires --run-dir, --jev-plan and --validation')
            out=w.prepare(root,python,absolute(args.run_dir),absolute(args.jev_plan),absolute(args.validation))
        else:
            if not args.plan: parser.error('Provide --plan')
            if args.action=='check':
                p,_=w.inspect(root,python,absolute(args.plan)); print(json.dumps(p,indent=2)); return 0
            out=w.evaluate(root,python,absolute(args.plan))
        print(json.dumps({'status':'complete','action':args.action,'output':str(out)},indent=2));return 0
    except KeyboardInterrupt:
        print('Stopped; partial scoring evidence retained. Saved generation and Jev responses are unchanged.',file=sys.stderr);return 130
    except (OSError,ValueError,RuntimeError,KeyError,TypeError,TimeoutError) as exc:
        print(f'MATH500 scoring stopped: {exc}',file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
