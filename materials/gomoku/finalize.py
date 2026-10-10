#!/usr/bin/env python3
"""Document and package a validated set; never modify its boards or gold answers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--logs', default='')
    a = ap.parse_args()
    root = Path(a.root).resolve()
    records = [json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    validation = json.loads((root/'validation.json').read_text())
    assert validation['status'] == 'passed' and len(records) == 300
    with (root/'questions.jsonl').open('w') as flat, (root/'omnijev_single_question_inputs.jsonl').open('w') as api:
        n = 0
        for rec in records:
            for qid, question in rec['questions'].items():
                example_id = rec['id']+'/'+qid
                flat.write(json.dumps({'id': example_id, 'record_id': rec['id'], 'qid': qid,
                                       'image': rec['images'][0], 'question': question,
                                       'answer': rec['answers'][qid]}, ensure_ascii=False)+'\n')
                api.write(json.dumps({'id': example_id, 'record_id': rec['id'],
                                      'state': {'images': [str(root/p) for p in rec['images']]},
                                      'questions': {qid: question}}, ensure_ascii=False)+'\n')
                n += 1
    assert n == 1350
    # Check every flattened input against its independently validated parent record.
    lookup = {r['id']: r for r in records}
    rows = [json.loads(s) for s in (root/'questions.jsonl').read_text().splitlines()]
    inputs = [json.loads(s) for s in (root/'omnijev_single_question_inputs.jsonl').read_text().splitlines()]
    for row, inp in zip(rows, inputs):
        rec = lookup[row['record_id']]
        assert row['question'] == rec['questions'][row['qid']]
        assert row['answer'] == rec['answers'][row['qid']]
        assert inp['questions'] == {row['qid']: row['question']}
        assert not {'answer', 'answers', 'metadata', 'board', 'moves'} & set(inp)
    stats = json.loads((root/'statistics.json').read_text())
    baselines = {'binary_majority_accuracy_percent': {}, 'fixed_option_position_accuracy_percent': {}}
    for qid, counts in stats['binary_labels'].items():
        baselines['binary_majority_accuracy_percent'][qid] = 100*max(counts.values())/sum(counts.values())
    for qid, counts in stats['choice_gold_indices'].items():
        baselines['fixed_option_position_accuracy_percent'][qid] = 100*max(counts.values())/sum(counts.values())
    (root/'baselines.json').write_text(json.dumps(baselines, indent=2)+'\n')
    if a.logs:
        (root/'logs').mkdir(exist_ok=True)
        for source, name in [('gomoku-generation.log', 'generation.log'), ('gomoku-validation.log', 'validation.log')]:
            p = Path(a.logs)/source
            if p.exists(): shutil.copy2(p, root/'logs'/name)
    for p in Path(__file__).parent.glob('*.py'):
        if p.name in ['generate.py', 'validate.py', 'finalize.py']:
            shutil.copy2(p, root/'provenance'/p.name)
    readme = f'''# Gomoku Eval v1 — 独立合成五子棋评测集

数据目录：`{root}`
生成日期：2026-10-10；随机种子：20261010。
这是一套新生成的独立评测数据，不是 OmniJev 原评测集，分数不能直接对应其 70.60%。
已完成数据生成和标签校验，没有运行模型推理。

## 规模与规则

- 300 个 15×15 棋盘，PNG 尺寸 720×760；3 种清晰棋盘配色。
- 150 个已获胜局面：黑胜 75、白胜 75；150 个未结束局面。
- 正负局面的黑白棋子数量分布完全一致，避免靠数量猜“五连”。
- 黑先白后交替落子；每个棋盘提供完整可达落子历史，首次获胜立即结束。
- 自由五子棋：横、竖、两条斜向，连续至少五子即获胜；长连也获胜，无禁手、交换或特殊开局规则。
- 棋盘互不重复，旋转和镜像也不重复。整套数据为 test，不提供训练集。
- 非终局包括双方都有一步获胜点 38 个、仅当前方有 37 个、仅对方有 37 个、双方都没有 38 个。

## 题型与标签

| qid | 精确定义 | 题数 | 答案 |
| --- | --- | ---: | --- |
| has_five | 当前棋盘是否已有任一方五子或更多连线 | 300 | 是 150 / 否 150 |
| black_count | 图中黑子数量；四个候选整数 | 300 | 正确候选的 key |
| white_count | 图中白子数量；四个候选整数 | 300 | 正确候选的 key |
| can_win_now | 当前方下一手是否能形成五子或更多连线 | 150 | 是 75 / 否 75 |
| opponent_has_immediate_win | 在当前棋盘上，假设对方落一子，是否能形成五子或更多连线 | 150 | 是 75 / 否 75 |
| winning_move | 四个候选空位中，哪个能让当前方立即形成五连 | 75 | 正确坐标 key |
| blocking_move | 四个候选空位中，哪个能消除对方全部一步五连完成点 | 75 | 正确坐标 key |

共 1,350 题。末两项每题只有一个正确候选，不代表长期“最佳落子”。
阻挡题只判断对方的几何五连威胁，即使己方也能获胜，也按题目指定的阻挡条件回答。
所有二分类题固定定义为 **label=1 / value=true / label_text=yes**，**label=0 / value=false / label_text=no**。
这是本数据集自己的明确编码，不能拿来推断原 OmniJev gm_win 的编码。
选择题的 `gold_index` 是四个候选中的从 0 开始的位置；`value` 是候选 key。
计数题四个正确选项位置各 75 次；落子题四个位置分别出现 19、19、19、18 次。
OmniJev 仍可返回 abstain；本数据每道选择题都有一个正确候选，弃答计为错误。

## 文件

- `images/`：300 张纯棋盘图片，无答案、计数、输赢或最后一步标记。
- `records.jsonl`：300 行，包含棋盘、落子历史、问题、答案与元数据。
- `questions.jsonl`：1,350 行，每行一题与其答案，便于按 qid 分别评测。
- `omnijev_inputs.jsonl`：300 个按棋盘分组的 MSO1 输入，只有图片路径和问题，无标签。
- `omnijev_single_question_inputs.jsonl`：1,350 个单题 MSO1 输入，无标签；推荐按题型独立评测。
- `statistics.json`、`baselines.json`：标签分布与简单基线。
- `validation.json`：独立验证报告；`preview.png`：六种代表局面的带标签预览。
- `manifest.json`、`SHA256SUMS`：配置、来源脚本和文件校验值。
- `provenance/`：生成、独立验证和打包脚本；`logs/`：运行日志。

棋盘矩阵：0=空位，1=黑子，2=白子。坐标列 A–O 从左到右，行 1–15 从上到下。
落子历史中的 row/column 从 1 开始。这个棋盘编码与二分类标签编码属于不同字段。
`records.jsonl` 和 `questions.jsonl` 的图片路径相对数据目录，便于迁移；
两个 MSO1 输入文件使用当前服务器的绝对路径，迁移时需重新生成这两个输入视图。

## 读取示例（不加载模型）

```python
import json
from pathlib import Path
root = Path("{root}")
record = json.loads((root / "records.jsonl").open().readline())
image_path = root / record["images"][0]
questions = record["questions"]
gold = record["answers"]   # 仅供评分，不能传给模型
```

后续推理可以从单题输入文件取 `state` 和 `questions`，传给 `m.system_one(state, questions)`。
模型初始化仍使用已下载的 OmniJev-4B-v1.1 和 Qwen3.5-4B。

## 验证与评分

```bash
/data/users/sunziang/venvs/omnijev/bin/python \\
  /data/users/sunziang/omnijev-tools/gomoku-eval-v1/validate.py --root {root}
cd {root}
sha256sum -c SHA256SUMS
```

生成器用局部相邻棋子计数；验证器用独立的 NumPy 长度五滑动窗口，覆盖四个方向和长连。
逐步验证了全部 9,402 步落子，核对全部问题、所有候选、所有图片和文件校验值。
每个二分类题型的多数类基线均为 50%；计数题固定选同一位置为 25%；落子题最高为 25.33%。
建议按 qid 报告准确率，二分类额外报告两类召回率与混淆矩阵；不要只给整体平均值。

这里的轨迹是合法合成落子顺序，不是专家对局，也不评测长远局势、真实摄像头棋盘或闭环胜率。
若未来用于训练/验证拆分，要按 game_id 分组，并将同一 metadata.pair_id 的两个样本放在同一组。
生成脚本拒绝覆盖非空目录；复现时应指定新的空目录。
'''
    (root/'README.md').write_text(readme)
    manifest = json.loads((root/'manifest.json').read_text())
    manifest.update({'created_date': '2026-10-10', 'grouped_inputs': 300, 'single_question_inputs': 1350,
                     'validation_status': 'passed', 'model_inference_run': False})
    manifest['files_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(root.rglob('*')) if p.is_file()
                              and p.name not in ['manifest.json', 'SHA256SUMS']}
    (root/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    checksums = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'}
    (root/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in checksums.items()))
    for name, digest in checksums.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    print(json.dumps({'packaged_files': len(checksums), 'all_sha256_verified': True,
                      'boards': len(records), 'questions': n, 'bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}))


if __name__ == '__main__':
    main()
