#!/usr/bin/env python3
"""Export the completed web diagnostic, preserving original screenshots byte for byte."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from offline_operation_filter import add_operation_filter

ap = argparse.ArgumentParser()
ap.add_argument('--run', type=Path, required=True)
args = ap.parse_args()
root = args.run
inputs = [json.loads(x) for x in (root/'inputs.jsonl').read_text().splitlines()]
preds = {}
for model, name in [('omnijev','full-predictions.jsonl'),('qwen4b','qwen4b-predictions.jsonl')]:
    rows = [json.loads(x) for x in (root/name).read_text().splitlines()]
    preds[model] = {r['action_uid']: r for r in rows}
    assert len(preds[model]) == len(rows) == len(inputs) == 200
comparison = json.loads((root/'comparison-qwen4b.json').read_text())
protocol = json.loads((root/'protocol.json').read_text())
template = Path(__file__).with_name('web_viewer_template.html').read_text()

def swap(text, before, after):
    assert text.count(before) == 1, before[:80]
    return text.replace(before, after)

manifest = {'purpose':'Self-contained offline replay of existing web/webtest results', 'exports':{}}
for family in ['web','webtest']:
    selected = [r for r in inputs if r['family'] == family]
    assert len(selected) == 100
    assets = {}
    for row in selected:
        image = root/'screenshots'/Path(row['image']).name
        raw = image.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['image_sha256']
        assert raw.startswith(b'\xff\xd8\xff')
        assets[row['action_uid']] = 'data:image/jpeg;base64,'+base64.b64encode(raw).decode('ascii')
    states = {}
    for model, by_id in preds.items():
        records = []
        for inp in selected:
            p = by_id[inp['action_uid']]
            assert p['family'] == family and p['labels'] == inp['labels']
            for qid, answer in p['answers'].items():
                assert abs(sum(answer['probabilities'].values())+answer.get('abstain',0)-1) < .002
                assert p['correct'][qid] == bool(answer['valid'] and answer['choice']==inp['labels'][qid])
            records.append({**p, 'questions':inp['questions'], 'image_size':inp['image_size'],
                            'image_sha256':inp['image_sha256'], 'image_id':inp['action_uid'],
                            'candidates':inp['target_candidates'], 'action_index':inp['action_index'],
                            'website':inp['website']})
        source_summary = comparison['omnijev_summary' if model=='omnijev' else 'qwen_summary']
        for qid, metric in source_summary['families'][family]['metrics'].items():
            eligible = [r for r in records if qid in r['answers']]
            assert len(eligible) == metric['total']
            assert sum(r['correct'][qid] for r in eligible) == metric['correct']
        states[model] = {'model':'OmniJev-4B v1.1' if model=='omnijev' else 'Qwen3.5-4B 原版',
                         'method':'LoRA 与决策头的候选评分' if model=='omnijev' else
                                  '非思考模式；按候选答案字母的概率选择。概率仅在给定字母内归一化。',
                         'total':len(records), 'done':len(records), 'finished':True, 'records':records}
    payload = {'family':family, 'images':assets, 'states':states, 'comparison':comparison, 'protocol':protocol}
    html = swap(template, '<title>网页决策 · 逐步选择记录</title>', '<title>'+family.upper()+' · 离线评测记录</title>')
    html = swap(html, '<label>已完成的步骤 ', '<label>评测样本 ')
    html = swap(html, '<label><input id="live" type="checkbox" checked>跟随最新记录</label>', '<button id="play" aria-pressed="false">▶ 播放</button>')
    html = swap(html, '<button id="refresh">刷新</button>', '<button id="first">回到第一条</button>')
    html = swap(html, '只读查看；自建转换规则，非官网评测集的严格复现。', '独立离线文件，截图与两模型输出已内嵌；自建转换规则，非官网评测集的严格复现。')
    html = swap(html, 'let state={records:[]},index=0,busy=false;', 'const OFFLINE=JSON.parse(document.getElementById("offline-data").textContent);let state={records:[]},index=0,timer=null;')
    start = html.index('async function load()')
    end = html.index('function renderComparison()', start)
    html = html[:start]+'''function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ 播放';$('play').setAttribute('aria-pressed','false')}
function load(){let previous=state.records[index]?.action_uid;state=OFFLINE.states[$('model').value];state.comparison=OFFLINE.comparison;
$('model-title').textContent=OFFLINE.family.toUpperCase()+' · '+state.model+' · 离线记录';$('method').textContent=state.method;renderComparison();
$('progress').textContent='评测已完成 · '+state.total+' 个样本 · 来源 '+(OFFLINE.family==='web'?'train':'test_task')+' · 无需网络或推理服务';
$('step').replaceChildren(...state.records.map((r,i)=>{let o=document.createElement('option');o.value=String(i);o.textContent=(i+1)+' / '+state.total+' · '+r.website+' · '+r.action_uid.slice(0,8);return o}));
if(previous){let found=state.records.findIndex(r=>r.action_uid===previous);if(found>=0)index=found}draw();$('error').textContent=''}
function play(){if(timer){stop();return}if(index>=state.records.length-1)index=0;$('play').textContent='Ⅱ 暂停';$('play').setAttribute('aria-pressed','true');draw();timer=setInterval(()=>{if(index>=state.records.length-1){stop();return}index++;draw()},1800)}
''' + html[end:]
    html = swap(html, "for(const family of ['web','webtest'])", 'for(const family of [OFFLINE.family])')
    html = swap(html, "if($('image').getAttribute('src')!==r.image_url){$('image').src=r.image_url;$('frame').scrollTop=0}",
                "if($('image').dataset.imageId!==r.image_id){$('image').src=OFFLINE.images[r.image_id];$('image').dataset.imageId=r.image_id;$('frame').scrollTop=0}")
    old_tail = html.index(" $('model').onchange=", html.index('function renderComparison()'))
    html = html[:old_tail]+''' $('model').onchange=()=>{stop();load()};$('step').onchange=()=>{stop();index=Number($('step').value);draw()};$('prev').onclick=()=>{stop();index--;draw()};$('next').onclick=()=>{stop();index++;draw()};$('truth').onchange=draw;$('zoom').onchange=draw;$('play').onclick=play;$('first').onclick=()=>{stop();index=0;draw()};load();
</script></body></html>
'''
    # Hidden JSON carries each original screenshot once, shared by the two models.
    serialized = json.dumps(payload, ensure_ascii=False, separators=(',',':')).replace('<','\\u003c')
    html = swap(html, '\n<script>\n', '\n<script id="offline-data" type="application/json">'+serialized+'</script>\n<script>\n')
    html = add_operation_filter(html)
    assert 'fetch(' not in html and 'setInterval(load' not in html
    out = root/(family+'_offline.html')
    out.write_text(html)
    meta = {'path':str(out), 'samples':len(selected), 'models':list(states), 'images':len(assets),
            'original_screenshots_preserved':True, 'bytes':out.stat().st_size,
            'sha256':hashlib.sha256(out.read_bytes()).hexdigest(), 'verified_predictions':sum(len(r['answers']) for s in states.values() for r in s['records'])}
    meta['operation_filter'] = 'ground-truth labels.op; navigation and playback use filtered samples'
    meta['operation_statistics'] = 'live filtered and complete per-operation accuracy for both models; abstentions count as wrong'
    manifest['exports'][family] = meta
    print(json.dumps(meta,ensure_ascii=False),flush=True)
    del html, payload, assets, states, serialized
(root/'offline_export_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
