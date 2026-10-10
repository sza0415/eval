# Mario：Qwen3.5-4B 与 OmniJev-4B 的同题测试

这是新建的离线对照实验，不是官网 `mario` 行的原样复现，也不测游戏通关率。

## 官网分数如何理解

七列依次是 Base 0.8B、SFT 0.8B、OmniJev 0.8B、Base 2B、OmniJev 2B、Base 4B、OmniJev 4B。
Mario 分数为 27.43、40.14、53.88、52.13、54.69、46.78、56.05%。

OmniJev-4B 的 56.05% 是 412/735 道题答对：

| 题型 | 答对数 / 题数 | 准确率 |
|---|---:|---:|
| `mario_act` 动作选择 | 84/193 | 43.52% |
| `mario_jump` 跳跃判断 | 116/193 | 60.10% |
| `mario_run` 奔跑判断 | 143/193 | 74.09% |
| `mario_hold` 保持动作判断 | 69/156 | 44.23% |

原始 Qwen3.5-4B 报告是 729 道 test 题，动作选择 197 道，其他判断合计 532 道。
原始 Qwen 使用 `A1_raw` 候选答案概率；OmniJev 使用决策头。
历史基座只读第一张静态图，v1.1 OmniJev 将多张图组成带编号面板。
题目、输入和抽样不一致，所以官网的 +9.27 个百分点不是严格控制变量对照。
原始输入题目、转换器、735 题 ID 与划分规则在本次检查的公开仓库与发布资产中没有找到。

来源：

- https://github.com/tinnel123666888/OmniJev#evaluation-protocol-and-limitations
- https://github.com/tinnel123666888/OmniJev/blob/main/docs/dataset_scores_v11.md
- https://github.com/tinnel123666888/OmniJev/blob/main/docs/results_v11.json

## 本次固定协议

数据：`OpenGenGAME/super-mario-bros-rl-1-1`，版本 `3efa9014cf60b0a68de05b95f7cb76807a342575`。
151 个片段，每个 17 张截图；HF imagefolder 的 label 是片段编号，真实动作来自 `actions.txt`。
数字按数值排序，避免把 `10.png` 排到 `2.png` 之前。

在全部 2,265 个有效时点中，不看目标标签，以 seed=20261009 均匀无放回抽取 200 个时点，覆盖 110 个片段。
每个时点看图片 t-1 和 t，目标为 `actions.txt[t+1]`；图片 t+1 不输入模型。
数据采集器与真实模拟器 step 的因果对齐没有公开说明，因此准确描述是“预测下一条动作标注”。
不能由这个实验声称模型能控制游戏。

两个模型输入相同的 530×290 PNG 编号面板，前一帧在左、当前帧在右；同一份题目，max_pixels=200704。
每个时点四道题，共 800 道：

- 动作：七个 SIMPLE_MOVEMENT 动作加 abstain。
- 跳跃：下一条动作是否包含 A。
- 奔跑：下一条动作是否包含 B。
- 保持动作：下一条动作是否等于当前动作；此题明确给出当前动作文字。其他三题不提供动作标签。

动作映射来自数据作者的 loader 注释：

| 编号 | 按键 |
|---:|---|
| 0 | NOOP |
| 1 | right |
| 2 | right + A |
| 3 | right + B |
| 4 | right + A + B |
| 5 | A |
| 6 | left |

来源：https://github.com/opengengame/super-mario-bros-1-1/blob/main/datasets/action_video.py

Qwen 使用官方聊天模板，关闭 thinking，读取 A–H 这些单 token 选项字母的下一 token logits，softmax 后取最大值。
判断题使用 A=是、B=否；动作题按上述固定顺序排列，H=abstain。
这是一种明确可复核的候选打分方式；不声称与官网未公开的 `A1_raw` 代码一致。
四题分别组成提示并在同一 batch 计算，不生成文本。只在候选 token 内归一化的概率不等于全词表置信度。

OmniJev 使用 `MSO1.system_one`，四题 packed；运行时禁用温度和 noul bias，得到 raw 结果。
动作 argmax 包含 abstain。权重文件不修改。

当前 Blackwell/PyTorch 栈上的 Conv3d 图片分块投影很慢。评测脚本把该层改用数学等价的 flattened linear 运算：
输入分块空间尺寸恰好等于卷积 kernel，只有一个空间输出；权重不变。
两个模型都使用这条路径，并在首次输入上检查 CPU float32 卷积和矩阵乘法一致性。
BF16 的不同计算后端可能存在舍入差异；报告记录此设置。

## 如何运行

代码：`/data/users/sunziang/omnijev-tools/mario_compare.py`
数据和结果：`/ssd-public/ai_infra/public/share/users/ziangsun/OmniJev-experiments/mario_compare_20261009/`

```bash
source /data/users/sunziang/omnijev-tools/activate.sh
python /data/users/sunziang/omnijev-tools/mario_compare.py build
python /data/users/sunziang/omnijev-tools/mario_compare.py qwen
python /data/users/sunziang/omnijev-tools/mario_compare.py omnijev
```

`build` 会重建固定样本和面板。已有正式结果后应保留该样本清单；更换协议应使用新的 `--out` 目录。

输出：`samples.jsonl`、`protocol.json`、`panels/`、两个模型的 `*_predictions.jsonl` 与 `*_report.json`。
每个题型报告 accuracy、在实际出现类别上的 balanced accuracy、标签与预测分布、混淆矩阵、Brier、NLL、10-bin ECE。
混合准确率按 800 道题计分。混合多数类基线分别为每种题型选择其最多的标签，是同一评测集上的描述性基线，不是独立训练的模型。
延迟包括图片处理与各自 API 调用方式，Qwen 四个独立提示、OmniJev packed；不适合解释为严格等计算量的速度对比。

## 解释边界

这批公开轨迹可能曾进入 OmniJev 训练；无法证明与训练片段分离。
同一片段内样本相关，不能把 800 道题当成 800 个独立游戏回合。
未运行模拟器，没有测通关率、累计奖励、存活时长。
新结果应与官网历史结果分开展示。

## 本次已完成的结果

两个模型均完成同一份 200 个样本、800 道题；已独立核对样本 ID、gold、预测 argmax、面板校验值和准确率计算。

| 题型 | Qwen3.5-4B | OmniJev-4B v1.1 | 多数类基线 |
|---|---:|---:|---:|
| 动作 | 8.00% | 19.50% | 46.00% |
| 跳跃 | 49.50% | 48.50% | 51.50% |
| 奔跑 | 72.00% | 72.50% | 72.50% |
| 保持动作 | 61.00% | 71.50% | 71.50% |
| 混合 | 47.625%（381/800） | 53.00%（424/800） | 60.375% |

OmniJev 比 Qwen 高 5.375 个百分点，但两个模型均低于混合多数类基线。
OmniJev 的跳跃、奔跑、保持动作题分别 200/200 次回答“是”；较高的奔跑和保持准确率来自标签分布，不能说明这些判断具备有效区分能力。
动作 balanced accuracy：Qwen 13.33%，OmniJev 16.49%；始终预测 `right_B` 的 balanced accuracy 为 20%（按这批数据实际出现的五类计算）。

数据、模型版本、脚本和样本校验值记录于 `comparison.json`。逐题输出保留在两个 `*_predictions.jsonl` 中。

## 结果网页

`mario_viewer.html` 是独立 HTML，包含全部 200 个样本的截图和两模型逐题概率，可直接在浏览器打开。
支持按题型、预测差异和正确性筛选、搜索片段 ID、翻页、自动播放、隐藏答案以及导出当前预测 JSON。
页面回放已有输出，不调用模型。

生成代码位于 `/data/users/sunziang/omnijev-tools/mario-viewer/`，HTML 位于上述结果目录。
服务器只读查看服务监听 `127.0.0.1:8767`。从其他电脑访问时可转发端口：

```bash
ssh -L 8767:127.0.0.1:8767 sunziang@117.50.172.69
```

然后打开 `http://127.0.0.1:8767/`。如果本机已有同端口的预览，可改用 `-L 8768:127.0.0.1:8767` 并打开本机 8768。
