#!/usr/bin/env python3
"""Read-only progress for the paired LIBERO diagnostic, without loading models."""
import json
from pathlib import Path

out = Path('/ssd-public/ai_infra/public/share/users/ziangsun/OmniJev-experiments/libero10-paired-v1-seed20261010')
status = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {'state': 'not_started'}
status['progress'] = {}
for mode in ['omnijev', 'qwen-score', 'qwen-generate']:
    path = out/f'{mode}-predictions.jsonl'
    records = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    status['progress'][mode] = {'states': len(records), 'target_states': 250,
                                'questions': 2*len(records),
                                'last_uid': records[-1]['uid'] if records else None}
print(json.dumps(status, ensure_ascii=False, indent=2))
