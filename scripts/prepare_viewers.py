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
    manifest = {name: convert(name, source) for name, source in REPORTS.items()}
    (ROOT/'site/import-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({name: {'images': m['image_count'], 'payload_verified': m['evaluation_json_verified']} for name, m in manifest.items()}, indent=2))

if __name__ == '__main__':
    main()
