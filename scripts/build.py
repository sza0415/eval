#!/usr/bin/env python3
"""Validate and package the versioned static website. No inference or dependencies."""
import hashlib
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/'site'
DIST = ROOT/'dist'

class ResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        self.paths.extend(value for key, value in attrs if key in ('src', 'href') and value)

manifest = json.loads((SITE/'import-manifest.json').read_text())
for name, report in manifest.items():
    page = SITE/name/'index.html'
    assert hashlib.sha256(page.read_bytes()).hexdigest() == report['page_sha256'], page
    for relative, meta in report['images'].items():
        image = SITE/name/relative
        assert image.stat().st_size == meta['bytes'], image
        assert hashlib.sha256(image.read_bytes()).hexdigest() == meta['sha256'], image

for page in SITE.rglob('*.html'):
    html = page.read_text()
    assert '<html' in html and '</html>' in html, page
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=["\x27]https?://', html), page
    parser = ResourceParser()
    parser.feed(html)
    for path in parser.paths:
        if re.match(r'(?:https?:|data:|blob:|mailto:|#)', path):
            continue
        clean = path.split('#')[0].split('?')[0]
        if clean:
            resolved = (page.parent/clean).resolve()
            assert resolved.is_relative_to(SITE.resolve()), (page, path)
            assert resolved.exists(), (page, path)

if DIST.exists():
    shutil.rmtree(DIST)
shutil.copytree(SITE, DIST)
files = [p for p in DIST.rglob('*') if p.is_file()]
assert all(p.stat().st_size < 100_000_000 for p in files)
print(f'Validated {len(manifest)} imported reports and {sum(m["image_count"] for m in manifest.values())} original images.')
print(f'Static site: {len(files)} files, {sum(p.stat().st_size for p in files):,} bytes -> {DIST}')
