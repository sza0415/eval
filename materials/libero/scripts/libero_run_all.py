#!/usr/bin/env python3
"""Run the three frozen evaluation configurations sequentially on one GPU."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path('/ssd-public/ai_infra/public/share/users/ziangsun/OmniJev-experiments/libero10-paired-v1-seed20261010')
TOOLS = Path('/data/users/sunziang/omnijev-tools')

def status(state, **extra):
    value = {'state': state, 'pid': os.getpid(), 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra}
    temp = OUT/'status.json.tmp'
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(OUT/'status.json')
    print(json.dumps(value), flush=True)

def main():
    import fcntl
    with (OUT/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for mode in ['omnijev', 'qwen-score', 'qwen-generate']:
                metadata = OUT/f'{mode}-run.json'
                if metadata.exists() and json.loads(metadata.read_text()).get('completed_states') == 250:
                    print('ALREADY_COMPLETE', mode, flush=True)
                    continue
                status('running', stage=mode)
                with (OUT/f'{mode}.log').open('a', buffering=1) as log:
                    subprocess.run([sys.executable, str(TOOLS/'libero_eval.py'), '--mode', mode],
                                   check=True, stdout=log, stderr=subprocess.STDOUT)
            status('reporting')
            subprocess.run([sys.executable, str(TOOLS/'libero_report.py')], check=True)
            status('complete', report=str(OUT/'report.txt'), comparison=str(OUT/'comparison.json'))
        except Exception as exc:
            status('failed', error=str(exc))
            raise

if __name__ == '__main__': main()
