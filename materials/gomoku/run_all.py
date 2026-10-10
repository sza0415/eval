#!/usr/bin/env python3
"""Run the two frozen evaluations sequentially and record a persistent job status."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from evaluate import OUT, dump

OUT.mkdir(parents=True, exist_ok=True)
lock = (OUT/'run.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
status = {'pid': os.getpid(), 'status': 'running', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
dump(OUT/'job-status.json', status)
scripts = Path(__file__).parent
try:
    for name in ['qwen', 'omnijev', 'report']:
        status['phase'] = name
        dump(OUT/'job-status.json', status)
        command = [sys.executable, str(scripts/'report_comparison.py')] if name == 'report' else [sys.executable, str(scripts/'evaluate.py'), '--model', name]
        with (OUT/(name+'-run.log')).open('a', buffering=1) as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    status['status'] = 'completed'
except Exception as exc:
    status.update({'status': 'failed', 'error': repr(exc)})
    raise
finally:
    status['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    dump(OUT/'job-status.json', status)
