#!/usr/bin/env python3
"""Recompute scores from raw predictions and produce paired task comparisons."""
from collections import Counter
import csv
import json
from pathlib import Path
from evaluate import DATA, OUT, summary, dump

NAMES = {'has_five': '已有五连', 'black_count': '黑子计数', 'white_count': '白子计数',
         'can_win_now': '己方一步获胜', 'opponent_has_immediate_win': '对方一步获胜威胁',
         'winning_move': '选择获胜落点', 'blocking_move': '选择阻挡落点'}

def main():
    gold = {r['id']: r for r in [json.loads(s) for s in (DATA/'questions.jsonl').read_text().splitlines()]}
    models = {}
    for name in ['qwen', 'omnijev']:
        rows = [json.loads(s) for s in (OUT/(name+'-predictions.jsonl')).read_text().splitlines()]
        indexed = {r['id']: r for r in rows}
        assert len(rows) == len(indexed) == len(gold) == 1350 and set(indexed) == set(gold)
        for uid, p in indexed.items():
            expected = gold[uid]['answer']['value']
            assert p['gold'] == expected and p['qid'] == gold[uid]['qid']
            answer = p['answer']
            forced = answer['noul'] >= 0.5 if p['question_type'] == 'noul' else answer['choice']
            valid = True if p['question_type'] == 'noul' else bool(answer['valid'])
            assert p['prediction'] == (forced if valid else 'abstain')
            assert p['correct'] == bool(valid and forced == expected)
            assert p['forced_correct'] == bool(forced == expected)
        recalculated = summary(name, rows, len(gold))
        saved = json.loads((OUT/(name+'-summary.json')).read_text())
        assert saved['by_task'] == recalculated['by_task'] and saved['complete']
        models[name] = indexed
    protocols = {n: json.loads((OUT/(n+'-protocol.json')).read_text()) for n in models}
    for key in ['questions_sha256', 'inputs_sha256', 'max_pixels', 'code_commit']:
        assert protocols['qwen'][key] == protocols['omnijev'][key]
    counts = Counter(r['qid'] for r in gold.values())
    comparison = {'scope': 'Custom balanced synthetic Gomoku evaluation; not official OmniJev result reproduction.',
                  'boards': 300, 'questions_per_model': 1350, 'all_predictions_independently_rescored': True,
                  'protocols': protocols, 'tasks': {}}
    table = []
    for qid, count in counts.items():
        pairs = [(models['qwen'][uid], models['omnijev'][uid]) for uid, r in gold.items() if r['qid'] == qid]
        qcorrect = sum(q['correct'] for q, o in pairs)
        ocorrect = sum(o['correct'] for q, o in pairs)
        task = {'n': count, 'qwen_correct': qcorrect, 'omnijev_correct': ocorrect,
                'qwen_accuracy_percent': 100*qcorrect/count, 'omnijev_accuracy_percent': 100*ocorrect/count,
                'omnijev_minus_qwen_percentage_points': 100*(ocorrect-qcorrect)/count,
                'both_correct': sum(q['correct'] and o['correct'] for q, o in pairs),
                'qwen_only_correct': sum(q['correct'] and not o['correct'] for q, o in pairs),
                'omnijev_only_correct': sum(not q['correct'] and o['correct'] for q, o in pairs),
                'both_wrong': sum(not q['correct'] and not o['correct'] for q, o in pairs)}
        for name in models:
            rows = [r for r in models[name].values() if r['qid'] == qid]
            task[name+'_forced_choice_accuracy_percent'] = 100*sum(r['forced_correct'] for r in rows)/count
            task[name+'_abstentions'] = sum(not r['valid'] for r in rows)
        comparison['tasks'][qid] = task
        table.append({'task': qid, **task})
    dump(OUT/'comparison.json', comparison)
    with (OUT/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    lines = ['# 五子棋：Qwen3.5-4B 与 OmniJev-4B-v1.1', '',
             '相同 300 张图片、每个模型 1,350 题；这是一套独立合成评测集，不能直接对应官方分数。', '',
             '| 任务 | 题数 | Qwen (%) | OmniJev (%) | OmniJev 差值（百分点） |',
             '|---|---:|---:|---:|---:|']
    for qid, task in comparison['tasks'].items():
        lines.append(f"| {NAMES[qid]} | {task['n']} | {task['qwen_accuracy_percent']:.2f} | {task['omnijev_accuracy_percent']:.2f} | {task['omnijev_minus_qwen_percentage_points']:+.2f} |")
    lines += ['', '主表将弃答计为错误。选择题另列强制选择准确率：', '',
              '| 任务 | Qwen 强制选择 (%) | OmniJev 强制选择 (%) | Qwen 弃答 | OmniJev 弃答 |',
              '|---|---:|---:|---:|---:|']
    for qid, task in comparison['tasks'].items():
        if gold[next(uid for uid, r in gold.items() if r['qid'] == qid)]['question']['type'] == 'choice':
            lines.append(f"| {NAMES[qid]} | {task['qwen_forced_choice_accuracy_percent']:.2f} | {task['omnijev_forced_choice_accuracy_percent']:.2f} | {task['qwen_abstentions']} | {task['omnijev_abstentions']} |")
    lines += ['', 'Qwen 使用原始权重，关闭 thinking，取候选答案字母的下一 token 概率；二分类为 A=yes/B=no，选择题为 A–D 加 E=以上皆非。该结果不代表开启思考链的 Qwen 表现。', '',
              'OmniJev 使用仓库 MSO1 接口、LoRA、决策头及发布的校准参数。每个问题的分支相互隔离，只共享图像前缀。图像预算相同（768×28×28）；全部 300 张图片的两个 processor 像素张量已核对一致。', '',
              '为避免此 CUDA 栈的 Conv3d 性能问题，两个模型均将覆盖整个 patch 的卷积替换为数学等价的矩阵乘法，权重不变。8 个实际 patch 的 FP32 误差低于 1e-4。BF16 累加舍入可能导致概率差异；与原卷积的 6 道试运行题预测相同，最大概率差约 0.0312，详见 patch_output_parity.json。', '',
              '二分类采用 P(yes)≥0.5；选择题以含弃答的最高概率为主指标，弃答计错。所有结果由原始逐题预测重新评分并核对。二分类混淆矩阵及两类召回率见各模型 summary。', '',
              '结果为单张图像的计数和局部几何规则判断，不代表专家棋力、长远最优落子或对局胜率。运行时间包含不同推理路径，不用于宣称架构速度优势。', '',
              f'服务器结果目录：`{OUT}`', '',
              '文件：qwen/omnijev-predictions.jsonl（逐题预测）、*-summary.json（分项指标）、*-protocol.json（协议与版本）、comparison.json/csv（对照结果）。', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in comparison.items() if k != 'protocols'}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
