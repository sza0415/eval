#!/usr/bin/env python3
"""Convert local standalone reports into static pages, preserving image bytes.

Run when refreshing imported reports. Deployment uses the already tracked site/
and does not require the ignored original standalone reports.
"""
import base64
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    'libero': ROOT/'materials/libero/omnijev-qwen-libero-offline.html',
    'mario': ROOT/'materials/mario/mario_viewer.html',
    'web': ROOT/'materials/web/web_offline.html',
    'webtest': ROOT/'materials/web/webtest_offline.html',
}
DATA_URI = re.compile(r'data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)')

def customize_libero(html):
    """Put the question viewer first and navigate within the filtered question set."""
    html = html.replace('<a href="#overview">结果概览</a><a href="#tasks">逐任务成绩</a><a href="#samples">逐题查看</a>',
                        '<a href="#samples">逐题查看</a><a href="#overview">结果概览</a><a href="#tasks">逐任务成绩</a>')
    samples = re.search(r'<section id="samples">.*?</section>', html, re.S).group()
    html = html.replace(samples, '', 1).replace('<section id="overview">', samples+'\n<section id="overview">', 1)
    numbers = re.search(r'<div class="numbers">.*?(?=\n)', html).group()
    html = html.replace(numbers, '', 1).replace('<section id="overview">', '<section id="overview">'+numbers, 1)
    html = html.replace('原始输入 PNG 已嵌入', '按当前筛选顺序浏览 · 原始输入 PNG 已归档')
    html = html.replace('<article class="detail" id="sampleDetail" aria-label="选中样本详情"></article>',
        '<div class="question-viewer"><nav class="question-navigation" aria-label="逐题切换">'
        '<button class="button" id="prevQuestion" type="button">← 上一题</button>'
        '<span id="questionPosition" role="status" aria-live="polite" aria-atomic="true"></span>'
        '<button class="button button-primary" id="nextQuestion" type="button">下一题 →</button>'
        '</nav><article class="detail" id="sampleDetail" aria-label="选中样本详情"></article></div>')
    html = html.replace('page=0;if(!filtered.some(i=>i.key===selected))selected=filtered[0]?.key??null;',
        'if(!filtered.some(i=>i.key===selected))selected=filtered[0]?.key??null;page=Math.max(0,Math.floor(filtered.findIndex(i=>i.key===selected)/PAGE_SIZE));')
    html = html.replace('function renderDetail(){', 'function renderDetail(){updateQuestionNavigation();', 1)
    html = html.replace("function probs(answer,color){", """function updateQuestionNavigation(){
const index=filtered.findIndex(i=>i.key===selected);
el('questionPosition').textContent=index<0?'没有匹配的题目':`第 ${index+1} / ${filtered.length} 题`;
el('prevQuestion').disabled=index<=0;
el('nextQuestion').disabled=index<0||index>=filtered.length-1;
}
function moveQuestion(step){
const index=filtered.findIndex(i=>i.key===selected),next=index+step;
if(index<0||next<0||next>=filtered.length)return;
selected=filtered[next].key;page=Math.floor(next/PAGE_SIZE);renderList();renderDetail();
}
function probs(answer,color){""", 1)
    html = html.replace("el('prevPage').addEventListener", "el('prevQuestion').addEventListener('click',()=>moveQuestion(-1));\nel('nextQuestion').addEventListener('click',()=>moveQuestion(1));\nel('prevPage').addEventListener", 1)
    html = html.replace('</style>', """
.hero{padding:28px 0 24px}.numbers{margin-bottom:24px}.question-viewer{min-width:0}
.question-navigation{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;padding:12px;background:var(--paper);border:1px solid var(--line);border-radius:8px}
#questionPosition{font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums;text-align:center}
@media(max-width:650px){.question-viewer{order:-1}.question-navigation{gap:8px;padding:10px}.question-navigation .button{padding:8px 10px}#questionPosition{font-size:12px}}
@media print{.question-navigation{display:none}}
</style>""", 1)
    return html

def convert(name, source):
    original = source.read_text()
    target = ROOT/'site'/name
    assets = target/'assets'
    assets.mkdir(parents=True, exist_ok=True)
    records = {}

    def extract(match):
        raw = base64.b64decode(match.group(2), validate=True)
        digest = hashlib.sha256(raw).hexdigest()
        suffix = {'png': '.png', 'jpeg': '.jpg', 'webp': '.webp'}[match.group(1)]
        relative = 'assets/'+digest+suffix
        file = target/relative
        if file.exists():
            assert file.read_bytes() == raw
        else:
            file.write_bytes(raw)
        records[relative] = {'sha256': digest, 'bytes': len(raw)}
        return relative

    html = DATA_URI.sub(extract, original)
    html = html.replace("img-src data:", "img-src 'self' data:")
    # Keep all evaluation scripts and payloads intact except image locations.
    html = html.replace('<body>', '<body><div style="padding:12px 24px;background:#fff;border-bottom:1px solid #d9e2e7"><a href="../index.html" style="color:#23534b;font:600 14px system-ui;text-decoration:none">← 返回评测场景</a></div>', 1)
    html = html.replace('截图与两模型输出已内嵌', '截图与两模型输出已归档')
    html = html.replace('独立离线文件，', '静态评测页面，')
    html = html.replace('离线报告 · 全部完成', '评测报告 · 全部完成')
    html = html.replace('内嵌的数据、原始图片与未来参考图片', '归档的数据、原始图片与未来参考图片')
    html = html.replace('全部逻辑在本地运行，不进行网络请求。', '报告展示已归档的评测结果，图片从本站加载。')
    if name == 'libero':
        html = customize_libero(html)
    (target/'index.html').write_text(html)
    # Confirm the JSON evaluation payload is unchanged apart from image paths.
    def payloads(text):
        return [json.loads(x) for x in re.findall(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', text, re.S)]
    before, after = payloads(original), payloads(html)
    if before:
        expected = payloads(DATA_URI.sub(lambda m: 'assets/'+hashlib.sha256(base64.b64decode(m.group(2))).hexdigest()+{'png':'.png','jpeg':'.jpg','webp':'.webp'}[m.group(1)], original))
        assert expected == after
    return {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'page_sha256': hashlib.sha256(html.encode()).hexdigest(),
            'image_count': len(records), 'images': records,
            'evaluation_json_verified': bool(before)}

def main():
    path = ROOT/'site/import-manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update({name: convert(name, source) for name, source in REPORTS.items()})
    (ROOT/'site/import-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({name: {'images': m['image_count'], 'payload_verified': m['evaluation_json_verified']} for name, m in manifest.items()}, indent=2))

if __name__ == '__main__':
    main()
