#!/usr/bin/env python3
"""Create a self-contained offline HTML from the verified evaluation bundle."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'results/viewer-data.json'
data = json.loads(SOURCE.read_text())
assert len(data['rows']) == 250
assert all(len(r['predictions']) == 3 for r in data['rows'])
for row in data['rows']:
    assert row['image'].startswith('data:image/png;base64,')
    assert row['future_image'].startswith('data:image/jpeg;base64,')
    for m, p in row['predictions'].items():
        assert p['uid'] == row['uid'] and p['labels'] == row['labels']
for r in data['rows']:
    for q in ['direction', 'gripper']:
        a, b = r['predictions']['qwen-score']['answers'][q], r['predictions']['qwen-generate']['answers'][q]
        assert a['choice'] == b['choice'] and a['valid'] == b['valid']
        assert b['format_ok']
raw = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
template = (HERE/'report.template.html').read_text()
assert template.count('__REPORT_DATA__') == 1
html = template.replace('__REPORT_DATA__', raw)
out = HERE/'dist'
out.mkdir(exist_ok=True)
(out/'index.html').write_text(html)
deliverable = HERE.parent/'omnijev-qwen-libero-offline.html'
deliverable.write_text(html)
metadata = {'states': len(data['rows']), 'questions': 500, 'bytes': len(html.encode()),
            'html_sha256': hashlib.sha256(html.encode()).hexdigest(),
            'source_bundle_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
(HERE/'build-manifest.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps({'file': str(deliverable), **metadata}, indent=2))
