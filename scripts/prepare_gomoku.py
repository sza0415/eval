#!/usr/bin/env python3
"""Join original predictions to every question and verify the static viewer data."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'materials/gomoku'
TARGET = ROOT/'site/gomoku'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def main():
    records = read_rows(SOURCE/'records.jsonl')
    models = {}
    for name in ['qwen', 'omnijev']:
        rows = read_rows(SOURCE/'predictions'/f'{name}.jsonl')
        models[name] = {row['id']: row for row in rows}
        assert len(rows) == len(models[name]) == 1350
    original_manifest = json.loads((SOURCE/'manifest.json').read_text())
    examples = []
    images = {}
    for record in records:
        image = TARGET/record['images'][0]
        digest = sha(image)
        assert digest == original_manifest['files_sha256'][record['images'][0]], image
        images[record['images'][0]] = {'sha256': digest, 'bytes': image.stat().st_size}
        for qid, question in record['questions'].items():
            uid = record['id']+'/'+qid
            gold = record['answers'][qid]['value']
            example = {'id': uid, 'record_id': record['id'], 'qid': qid,
                       'type': question['type'], 'image': record['images'][0],
                       'prompt': question['instructions'], 'gold': gold,
                       'side_to_move': record.get('side_to_move'),
                       'options': question.get('options', [])}
            for name in models:
                row = models[name][uid]
                answer = row['answer']
                assert row['gold'] == gold and row['qid'] == qid
                if question['type'] == 'noul':
                    assert row['prediction'] == (answer['noul'] >= .5)
                    out = {'prediction': row['prediction'], 'probability_yes': answer['noul'],
                           'correct': row['correct'], 'valid': True}
                else:
                    assert row['prediction'] == (answer['choice'] if answer['valid'] else 'abstain')
                    assert set(answer['probabilities']) == {o['key'] for o in question['options']}
                    out = {'prediction': row['prediction'], 'valid': row['valid'],
                           'forced_prediction': row['forced_prediction'], 'abstain': answer['abstain'],
                           'probabilities': answer['probabilities'], 'correct': row['correct']}
                assert row['correct'] == (row['prediction'] == gold)
                example[name] = out
            examples.append(example)
    assert len(records) == len(images) == 300 and len(examples) == 1350
    assert {e['id'] for e in examples} == set(models['qwen']) == set(models['omnijev'])
    for name in models:
        summary = json.loads((SOURCE/f'{name}-summary.json').read_text())
        for qid, task in summary['by_task'].items():
            rows = [e for e in examples if e['qid'] == qid]
            assert len(rows) == task['total']
            assert sum(e[name]['correct'] for e in rows) == task['correct']
    payload = 'window.GOMOKU_EXAMPLES='+json.dumps(examples, ensure_ascii=False, separators=(',', ':'))+';\n'
    (TARGET/'examples-data.js').write_text(payload)
    images['preview.png'] = {'sha256': sha(TARGET/'preview.png'), 'bytes': (TARGET/'preview.png').stat().st_size}
    path = ROOT/'site/import-manifest.json'
    manifest = json.loads(path.read_text())
    manifest['gomoku'] = {
        'source_sha256': sha(SOURCE/'comparison.json'), 'page_sha256': sha(TARGET/'index.html'),
        'image_count': len(images), 'images': images, 'evaluation_json_verified': True,
        'examples': len(examples), 'boards': len(records),
        'data_files': {name: {'sha256': sha(TARGET/name), 'bytes': (TARGET/name).stat().st_size}
                       for name in ['examples-data.js', 'examples.js', 'gomoku.css']},
        'source_files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                               [SOURCE/'records.jsonl', SOURCE/'predictions/qwen.jsonl', SOURCE/'predictions/omnijev.jsonl']}}
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print('Gomoku viewer verified: 300 original images, 1350 paired predictions, all task scores matched.')

if __name__ == '__main__':
    main()
