#!/usr/bin/env python3
"""Add gold-operation navigation to a self-contained evaluation viewer."""
import argparse
import hashlib
import json
from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new)


def add_operation_filter(html):
    start = html.index('<script id="offline-data" type="application/json">')
    end = html.index('</script>', start) + len('</script>')
    head, payload, script = html[:start], html[start:end], html[end:]
    if 'id="operation-filter"' in head:
        return add_operation_statistics(html)
    head = replace_once(head, '<label>评测样本 ',
        '<label>操作类型（标注） <select id="operation-filter" aria-label="按标注操作类型筛选"><option value="all">全部操作类型</option></select></label><label>评测样本 ')
    head = replace_once(head, '\n<main class="layout">',
        '\n<p class="muted" style="margin:0 24px 16px">按数据集标注的正确操作筛选；切换模型时保留同一批样本。上一条、下一条和播放均在筛选结果内进行。</p>\n<main class="layout">')
    head = replace_once(head, '<h2>同一批样本的结果对照</h2>',
        '<h2>完整评测集的结果对照</h2><p class="muted">下表保留完整 100 个样本的汇总，不随操作类型筛选变化。</p>')
    script = replace_once(script, 'let state={records:[]},index=0,timer=null;',
        'let state={records:[]},viewRecords=[],index=0,timer=null;')
    draw_start = script.index('function draw()')
    draw_end = script.index('function stop()', draw_start)
    draw = script[draw_start:draw_end].replace('state.records', 'viewRecords')
    draw = replace_once(draw, "$('prompt').textContent='等待该模型的首条预测记录…';",
        "$('prompt').textContent='当前操作类型没有可查看的样本。';$('play').disabled=true;$('first').disabled=true;")
    draw = replace_once(draw, "$('image').hidden=false;",
        "$('play').disabled=false;$('first').disabled=false;$('image').hidden=false;")
    script = script[:draw_start] + draw + script[draw_end:]
    load_start = script.index('function load()')
    load_end = script.index('function renderComparison()', load_start)
    script = script[:load_start] + '''function buildOperationFilter(){
const selected=$('operation-filter').value;const counts=new Map();
for(const r of state.records)counts.set(r.labels.op,(counts.get(r.labels.op)||0)+1);
const all=document.createElement('option');all.value='all';all.textContent='全部操作类型 · '+state.records.length+' 个样本';
$('operation-filter').replaceChildren(all);
const keys=[...new Set([...Object.keys(names),...counts.keys()])];
for(const key of keys){if(!counts.has(key))continue;const option=document.createElement('option');option.value=key;option.textContent=(names[key]||key)+' / '+key+' · '+counts.get(key)+' 个样本';$('operation-filter').append(option)}
$('operation-filter').value=selected==='all'||counts.has(selected)?selected:'all';
}
function applyOperationFilter(previous){
const key=$('operation-filter').value;viewRecords=state.records.filter(r=>key==='all'||r.labels.op===key);
index=Math.max(0,viewRecords.findIndex(r=>r.action_uid===previous));
const originalNumbers=new Map(state.records.map((r,i)=>[r.action_uid,i+1]));
$('step').replaceChildren(...viewRecords.map((r,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=(i+1)+' / '+viewRecords.length+(key==='all'?'':' · 原样本 '+originalNumbers.get(r.action_uid))+' · '+r.website+' · '+r.action_uid.slice(0,8);return option}));
$('step').disabled=viewRecords.length===0;
$('progress').textContent='评测已完成 · '+state.total+' 个样本 · 当前筛选 '+(key==='all'?'全部操作':(names[key]||key)+' / '+key)+'：'+viewRecords.length+' 个样本 · 来源 '+(OFFLINE.family==='web'?'train':'test_task')+' · 无需网络或推理服务';
draw();
}
function load(){const previous=viewRecords[index]?.action_uid;state=OFFLINE.states[$('model').value];state.comparison=OFFLINE.comparison;
$('model-title').textContent=OFFLINE.family.toUpperCase()+' · '+state.model+' · 离线记录';$('method').textContent=state.method;
buildOperationFilter();renderComparison();applyOperationFilter(previous);$('error').textContent='';}
function play(){if(timer){stop();return}if(!viewRecords.length)return;if(index>=viewRecords.length-1)index=0;$('play').textContent='Ⅱ 暂停';$('play').setAttribute('aria-pressed','true');draw();timer=setInterval(()=>{if(index>=viewRecords.length-1){stop();return}index++;draw()},1800)}
''' + script[load_end:]
    script = replace_once(script, "$('model').onchange=()=>{stop();load()};",
        "$('model').onchange=()=>{stop();load()};$('operation-filter').onchange=()=>{stop();applyOperationFilter(viewRecords[index]?.action_uid)};")
    return add_operation_statistics(head + payload + script)


def add_operation_statistics(html):
    start = html.index('<script id="offline-data" type="application/json">')
    end = html.index('</script>', start) + len('</script>')
    head, payload, script = html[:start], html[start:end], html[end:]
    if 'id="operation-stats-panel"' in head:
        return html
    head = replace_once(head, '</style></head>', '''
#operation-stats-panel{margin:0 24px 20px}
#operation-stats-panel th:first-child{width:24%}
#operation-stats-panel table{margin-bottom:12px}
#operation-stats-panel .selected-group{background:color-mix(in srgb,var(--blue) 9%,var(--panel))}
@media(max-width:820px){#operation-stats-panel{margin:0 12px 16px}#operation-stats-panel{overflow-x:auto}#operation-stats-panel table{min-width:600px}}
</style></head>''')
    head = replace_once(head, '\n<main class="layout">', '''
<section class="panel" id="operation-stats-panel">
<h2>当前筛选的准确率</h2><p id="operation-stats-scope" class="status" aria-live="polite"></p>
<p class="muted">按标注的正确操作类型分组，弃权计错。括号为正确数 / 可评测数；联合准确率要求操作与元素同时正确，未检查输入文本或下拉选择值。</p>
<table><thead><tr><th>模型</th><th>操作准确率</th><th>元素准确率</th><th>联合准确率</th><th>弃权数<br>操作 / 元素</th></tr></thead><tbody id="operation-stats"></tbody></table>
<details><summary>展开全部操作类型的统计</summary><p class="muted">本表展示完整评测集按操作类型的分组；当前筛选类型会高亮。小样本类别需结合正确数判断。</p>
<table><thead><tr><th>操作类型 · 模型</th><th>操作准确率</th><th>元素准确率</th><th>联合准确率</th><th>弃权数<br>操作 / 元素</th></tr></thead><tbody id="operation-breakdown"></tbody></table></details>
</section>
<main class="layout">''')
    functions = '''function operationMetrics(records){
const targets=records.filter(r=>r.answers.target);return {
op:{correct:records.filter(r=>r.correct.op).length,total:records.length},
target:{correct:targets.filter(r=>r.correct.target).length,total:targets.length},
joint:{correct:targets.filter(r=>r.correct.op&&r.correct.target).length,total:targets.length},
abstainOp:records.filter(r=>!r.answers.op.valid).length,abstainTarget:targets.filter(r=>!r.answers.target.valid).length};
}
function metricText(metric){return metric.total?(100*metric.correct/metric.total).toFixed(2)+'% ('+metric.correct+'/'+metric.total+')':'暂无可评测样本 (0/0)'}
function appendOperationStats(body,label,metrics,highlight){
const row=document.createElement('tr');if(highlight)row.classList.add('selected-group');const title=document.createElement('td');title.textContent=label;row.append(title);
for(const key of ['op','target','joint']){const cell=document.createElement('td');cell.className='number';cell.textContent=metricText(metrics[key]);row.append(cell)}
const abstain=document.createElement('td');abstain.className='number';abstain.textContent=metrics.abstainOp+' / '+metrics.abstainTarget;row.append(abstain);body.append(row);
}
function renderOperationStats(){
const selected=$('operation-filter').value;const subset=operationMetrics(viewRecords);
$('operation-stats-scope').textContent=OFFLINE.family.toUpperCase()+' · '+(selected==='all'?'全部操作类型':(names[selected]||selected)+' / '+selected)+' · 操作题 '+subset.op.total+' 道，元素题及联合指标 '+subset.target.total+' 道';
$('operation-stats').replaceChildren();$('operation-breakdown').replaceChildren();
const models=['omnijev','qwen4b'];
for(const model of models){const source=OFFLINE.states[model];const rows=source.records.filter(r=>selected==='all'||r.labels.op===selected);appendOperationStats($('operation-stats'),source.model,operationMetrics(rows),false)}
const counts=new Set(state.records.map(r=>r.labels.op));const operations=[...new Set([...Object.keys(names),...counts])].filter(key=>counts.has(key));
for(const operation of operations){for(const model of models){const source=OFFLINE.states[model];appendOperationStats($('operation-breakdown'),(names[operation]||operation)+' · '+source.model,operationMetrics(source.records.filter(r=>r.labels.op===operation)),selected===operation)}}
}
'''
    script = replace_once(script, 'function buildOperationFilter(){', functions + 'function buildOperationFilter(){')
    script = replace_once(script, 'draw();\n}\nfunction load()', 'draw();renderOperationStats();\n}\nfunction load()')
    return head + payload + script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.run / 'offline_export_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for family in ('web', 'webtest'):
        path = args.run / (family + '_offline.html')
        before = path.read_text()
        after = add_operation_filter(before)
        data_start = before.index('<script id="offline-data"')
        data_end = before.index('</script>', data_start) + len('</script>')
        payload = before[data_start:data_end]
        assert after.count(payload) == 1, 'Embedded evaluation data changed'
        if before != after:
            temporary = path.with_suffix('.html.tmp')
            temporary.write_text(after)
            temporary.replace(path)
        entry = manifest['exports'][family]
        entry['bytes'] = path.stat().st_size
        entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry['operation_filter'] = 'ground-truth labels.op; navigation and playback use filtered samples'
        entry['operation_statistics'] = 'live filtered and complete per-operation accuracy for both models; abstentions count as wrong'
        print(json.dumps({'family': family, 'bytes': entry['bytes'], 'sha256': entry['sha256']}, ensure_ascii=False))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
