#!/usr/bin/env python3
"""Export existing paired predictions to a self-contained, read-only HTML viewer."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
run = args.run
samples = [json.loads(x) for x in (run/'samples.jsonl').read_text().splitlines()]
summary = json.loads((run/'comparison.json').read_text())
assert hashlib.sha256((run/'samples.jsonl').read_bytes()).hexdigest() == summary['samples_sha256']
predictions = {}
reports = {}
for model in ['qwen', 'omnijev']:
    rows = [json.loads(x) for x in (run/(model+'_predictions.jsonl')).read_text().splitlines()]
    predictions[model] = {(x['id'], x['qid']): x for x in rows}
    assert len(rows) == len(predictions[model]) == len(samples)*4
    reports[model] = json.loads((run/(model+'_report.json')).read_text())
records = []
for sample in samples:
    image = run/'panels'/Path(sample['image']).name
    raw = image.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sample['image_sha256']
    answers = {}
    for model, by_key in predictions.items():
        answers[model] = {}
        for qid, gold in sample['gold'].items():
            answer = by_key[(sample['id'], qid)]
            assert answer['gold'] == gold
            answers[model][qid] = answer
    records.append({**sample, 'image': 'data:image/png;base64,'+base64.b64encode(raw).decode(), 'answers': answers})
payload = {'records': records, 'summary': summary, 'reports': reports,
           'protocol': json.loads((run/'protocol.json').read_text())}
serialized = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
template = Path(__file__).with_name('index.html').read_text()
assert template.count('__VIEWER_DATA__') == 1
output = args.output or run/'mario_viewer.html'
output.write_text(template.replace('__VIEWER_DATA__', serialized))
print(json.dumps({'path': str(output), 'states': len(records), 'questions_per_model': len(records)*4,
                  'bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))
