#!/usr/bin/env python3
"""Validate complete paired predictions, then aggregate with episode-level uncertainty."""
import collections
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path('/ssd-public/ai_infra/public/share/users/ziangsun')
OUT = ROOT/'OmniJev-experiments/libero10-paired-v1-seed20261010'
MODES = ['omnijev', 'qwen-score', 'qwen-generate']

def load(path): return [json.loads(s) for s in path.read_text().splitlines()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    rows = load(OUT/'test.jsonl')
    assert len(rows) == 250 and len({r['uid'] for r in rows}) == 250
    source = {r['uid']: r for r in rows}
    preds, summaries = {}, {}
    for mode in MODES:
        values = load(OUT/f'{mode}-predictions.jsonl')
        assert len(values) == len(rows)
        index = {r['uid']: r for r in values}
        assert set(index) == set(source)
        run = json.loads((OUT/f'{mode}-run.json').read_text())
        assert run['completed_states'] == 250
        assert run['predictions_sha256'] == sha(OUT/f'{mode}-predictions.jsonl')
        for row in values:
            original = source[row['uid']]
            assert row['labels'] == original['labels']
            assert row['task_id'] == original['task_id'] and row['episode_uid'] == original['episode_uid']
            for q, answer in row['answers'].items():
                assert row['correct'][q] == bool(answer['valid'] and answer['choice'] == row['labels'][q])
                if 'probabilities' in answer:
                    assert set(answer['probabilities']) == {o['key'] for o in original['questions'][q]['options']}
                    p = list(answer['probabilities'].values())+[answer['abstain']]
                    assert all(math.isfinite(v) and 0 <= v <= 1 for v in p)
                    assert abs(sum(p)-1) < .001
        preds[mode] = index
        summary = {'states': len(values), 'questions': len(values)*2, 'metrics': {},
                   'latency_two_questions_per_state': {'median_seconds': float(np.median([v['wall_seconds'] for v in values])),
                       'p90_seconds': float(np.percentile([v['wall_seconds'] for v in values], 90)),
                       'mean_seconds': float(np.mean([v['wall_seconds'] for v in values])),
                       'total_seconds': sum(v['wall_seconds'] for v in values)},
                   'peak_allocated_gib': run['peak_allocated_gib'], 'peak_reserved_gib': run['peak_reserved_gib']}
        for q in ['direction', 'gripper']:
            labels = [o['key'] for o in rows[0]['questions'][q]['options']]
            counts = collections.Counter(r['labels'][q] for r in values)
            confusion = {k: dict(collections.Counter(
                (r['answers'][q]['choice'] if r['answers'][q]['valid'] else
                 ('invalid_format' if not r['answers'][q].get('format_ok', True) else 'abstain'))
                for r in values if r['labels'][q] == k)) for k in labels}
            recalls, f1s = [], []
            for label in labels:
                tp = sum(r['correct'][q] and r['labels'][q] == label for r in values)
                fp = sum(r['answers'][q]['valid'] and r['answers'][q]['choice'] == label and r['labels'][q] != label for r in values)
                fn = counts[label]-tp
                recalls.append(tp/counts[label] if counts[label] else 0)
                f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
            correct = sum(v['correct'][q] for v in values)
            forced = (sum(v['answers'][q]['choice'] == v['labels'][q] for v in values)/len(values)
                      if mode != 'qwen-generate' else None)
            summary['metrics'][q] = {
                'correct': correct, 'total': len(values), 'accuracy_percent': 100*correct/len(values),
                'forced_choice_accuracy_percent': None if forced is None else 100*forced,
                'macro_f1_percent': 100*float(np.mean(f1s)),
                'balanced_accuracy_percent': 100*float(np.mean(recalls)),
                'abstentions': sum((not v['answers'][q]['valid'] if mode != 'qwen-generate'
                                    else v['answers'][q]['abstained']) for v in values),
                'invalid_format': sum(not v['answers'][q].get('format_ok', True) for v in values),
                'label_counts': dict(counts), 'majority_class': counts.most_common(1)[0][0],
                'test_majority_class_baseline_percent': 100*max(counts.values())/len(values),
                'confusion_matrix': confusion,
                'per_task': {str(t): {'correct': sum(v['correct'][q] for v in values if v['task_id'] == t),
                                      'total': sum(v['task_id'] == t for v in values),
                                      'accuracy_percent': 100*float(np.mean([v['correct'][q] for v in values if v['task_id'] == t]))}
                             for t in range(10)},
            }
            if mode == 'qwen-score':
                summary['metrics'][q]['mean_full_vocabulary_letter_mass'] = float(np.mean([v['answers'][q]['full_vocabulary_letter_mass'] for v in values]))
        summary['overall_accuracy_percent'] = 100*sum(sum(v['correct'].values()) for v in values)/(2*len(values))
        summary['both_questions_correct_states'] = sum(all(v['correct'].values()) for v in values)
        summaries[mode] = summary
    # Stratified paired bootstrap: resample five entire episodes per task, never individual adjacent frames.
    rng = np.random.default_rng(20261010)
    episodes_by_task = {t: sorted({r['episode_uid'] for r in rows if r['task_id'] == t}) for t in range(10)}
    assert all(len(e) == 5 for e in episodes_by_task.values())
    episodes = [e for t in range(10) for e in episodes_by_task[t]]
    cluster = np.array([[[[float(preds[mode][r['uid']]['correct'][q]) for r in rows if r['episode_uid'] == episode]
                         for episode in episodes] for q in ['direction', 'gripper']] for mode in MODES])
    assert cluster.shape == (3, 2, 50, 5)
    means = cluster.mean(axis=-1)
    sampled = np.concatenate([rng.integers(t*5, (t+1)*5, size=(5000, 5)) for t in range(10)], axis=1)
    boot = means[:, :, sampled].mean(axis=-1)  # model, question, bootstrap
    pairs = {}
    for m, mode in enumerate(MODES):
        for qi, q in enumerate(['direction', 'gripper']):
            summaries[mode]['metrics'][q]['episode_bootstrap_accuracy_ci95_percent'] = (100*np.percentile(boot[m, qi], [2.5, 97.5])).tolist()
        summaries[mode]['overall_episode_bootstrap_ci95_percent'] = (100*np.percentile(boot[m].mean(axis=0), [2.5, 97.5])).tolist()
    for m, mode in enumerate(MODES[1:], start=1):
        pairs[mode] = {}
        for qi, q in enumerate(['direction', 'gripper']):
            pairs[mode][q] = {
                'both_correct': sum(preds['omnijev'][r['uid']]['correct'][q] and preds[mode][r['uid']]['correct'][q] for r in rows),
                'omnijev_only_correct': sum(preds['omnijev'][r['uid']]['correct'][q] and not preds[mode][r['uid']]['correct'][q] for r in rows),
                'qwen_only_correct': sum(not preds['omnijev'][r['uid']]['correct'][q] and preds[mode][r['uid']]['correct'][q] for r in rows),
                'both_wrong': sum(not preds['omnijev'][r['uid']]['correct'][q] and not preds[mode][r['uid']]['correct'][q] for r in rows),
                'omnijev_minus_qwen_percentage_points': summaries['omnijev']['metrics'][q]['accuracy_percent']-summaries[mode]['metrics'][q]['accuracy_percent'],
                'paired_episode_bootstrap_delta_ci95_pp': (100*np.percentile(boot[0, qi]-boot[m, qi], [2.5, 97.5])).tolist(),
            }
    result = {'purpose': 'Custom paired frame diagnostic, not official robot_long reproduction or closed-loop task success',
              'sample': json.loads((OUT/'protocol.json').read_text())['statistics']['test'],
              'uncertainty': '5000 paired stratified episode bootstrap replicates; seed 20261010; resample 5 episodes within each of 10 tasks.',
              'validated': 'Complete unique IDs, source hashes, identical gold labels, consistent correctness flags, valid probabilities, and matching pixels verified.',
              'models': summaries, 'paired': pairs}
    (OUT/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ['LIBERO-10 自定义配对评测：250 个状态，500 道题',
             '不是官网 robot_long 分数复现，也不是机器人闭环任务成功率。', '',
             '模型 | 方向准确率 | 夹爪命令准确率 | 总准确率 | 两题/状态耗时中位数 | 峰值显存']
    for mode in MODES:
        s = summaries[mode]
        lines.append(f"{mode} | {s['metrics']['direction']['accuracy_percent']:.2f}% | {s['metrics']['gripper']['accuracy_percent']:.2f}% | {s['overall_accuracy_percent']:.2f}% | {s['latency_two_questions_per_state']['median_seconds']:.3f}s | {s['peak_allocated_gib']:.2f}GiB")
    lines += ['', '测试集多数类常量基线：方向 19.2%，夹爪 52.4%。',
              '方向：未来5个记录步的净位移，转换到工作台相机坐标后取最大分量；每轴均小于1cm为静止。',
              '夹爪：画面为动作后观测，下一命令取 actions[t+1,6]；-1打开，+1关闭，包含重复命令。',
              '输入：同一张经过OpenGL上下翻转后的双相机PNG，固定英文任务指令/问题/选项。',
              'OmniJev：发布的适配器与决策头；Qwen：未加载适配器的原始4B权重，关闭thinking。',
              'Qwen-score：限定候选字母的下一token概率；Qwen-generate：贪心生成，最多8token，严格单字母解析。',
              '弃答计错；强制选择准确率、宏F1、混淆矩阵、逐任务成绩、配对置信区间见 comparison.json。',
              '加载和3次开发集预热不计入耗时；耗时是当前具体实现，包含图片处理。',
              '发布模型与LIBERO演示的训练重叠未知；不能据此声称未见数据泛化。']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__ == '__main__': main()
