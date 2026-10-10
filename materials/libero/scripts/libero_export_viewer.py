#!/usr/bin/env python3
"""Read-only export of evaluated inputs and future reference frames for an offline report."""
import base64
import hashlib
import io
import json
import sys
from pathlib import Path
import h5py
from PIL import Image

sys.path.insert(0, '/data/users/sunziang/OmniJev')
from mso.panels import compose
ROOT = Path('/ssd-public/ai_infra/public/share/users/ziangsun')
OUT = ROOT/'OmniJev-experiments/libero10-paired-v1-seed20261010'

def jpeg(im):
    buf = io.BytesIO()
    im.convert('RGB').save(buf, format='JPEG', quality=94, subsampling=0, optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')

def main():
    rows = [json.loads(s) for s in (OUT/'test.jsonl').read_text().splitlines()]
    assert len(rows) == 250
    predictions = {m: {r['uid']: r for r in [json.loads(s) for s in (OUT/f'{m}-predictions.jsonl').read_text().splitlines()]}
                   for m in ['omnijev', 'qwen-score', 'qwen-generate']}
    export = []
    handles = {}
    try:
        for row in rows:
            assert hashlib.sha256(Path(row['image']).read_bytes()).hexdigest() == row['image_sha256']
            name = row['source_file']
            if name not in handles:
                handles[name] = h5py.File(ROOT/'datasets/LIBERO/libero_10'/name, 'r')
            g = handles[name][f"data/{row['episode']}"]
            # Future references are displayed only in the report, never passed to evaluated models.
            future_views = [Image.fromarray(g[f'obs/{sensor}'][row['future_timestep']][::-1].copy())
                            for sensor in ['agentview_rgb', 'eye_in_hand_rgb']]
            canvas = Image.new('RGB', (274, 162), (32, 32, 32))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(canvas)
            for i, im in enumerate(future_views):
                x = 6+i*134
                draw.text((x+2, 10), str(i+1), fill='white')
                canvas.paste(im, (x, 28))
            record = {k: row[k] for k in ['uid', 'task_id', 'episode', 'episode_uid', 'timestep',
                                         'future_timestep', 'task_instruction', 'questions', 'labels', 'label_evidence', 'image_sha256']}
            # Preserve original PNG input bytes for faithful inspection; JPEG future preview reduces size.
            record['image'] = 'data:image/png;base64,'+base64.b64encode(Path(row['image']).read_bytes()).decode('ascii')
            record['future_image'] = jpeg(canvas)
            record['predictions'] = {m: predictions[m][row['uid']] for m in predictions}
            export.append(record)
    finally:
        for h in handles.values(): h.close()
    bundle = {'comparison': json.loads((OUT/'comparison.json').read_text()),
              'protocol': json.loads((OUT/'protocol.json').read_text()),
              'runtime': json.loads((OUT/'runtime-evidence.json').read_text()),
              'runs': {m: json.loads((OUT/f'{m}-run.json').read_text()) for m in predictions}, 'rows': export}
    path = OUT/'viewer-data.json'
    path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(',', ':')))
    print(json.dumps({'states': len(export), 'bytes': path.stat().st_size, 'output': str(path)}), flush=True)

if __name__ == '__main__': main()
