LIBERO-10 自定义配对评测 v1

服务器：sunziang@117.50.172.69
代码：/data/users/sunziang/omnijev-tools/libero_*.py
虚拟环境：/data/users/sunziang/venvs/omnijev
数据：/ssd-public/ai_infra/public/share/users/ziangsun/datasets/LIBERO/libero_10
结果：/ssd-public/ai_infra/public/share/users/ziangsun/OmniJev-experiments/libero10-paired-v1-seed20261010

查看进度（不加载模型）：
  python /data/users/sunziang/omnijev-tools/libero_status.py

查看日志：
  tail -n 10 /ssd-public/ai_infra/public/share/users/ziangsun/OmniJev-experiments/libero10-paired-v1-seed20261010/omnijev.log
  后续阶段将文件名替换为 qwen-score.log 或 qwen-generate.log。

中断后的恢复：
  source /data/users/sunziang/omnijev-tools/activate.sh
  python /data/users/sunziang/omnijev-tools/libero_run_all.py
  已完成的模式会跳过；未完成的模式按 uid 恢复。文件锁防止重复启动。
  题集与 runner 哈希发生变化时会拒绝混用结果。重新设计实验需另建输出目录。

单独运行一种模式（避免与现有作业同时运行）：
  source /data/users/sunziang/omnijev-tools/activate.sh
  python /data/users/sunziang/omnijev-tools/libero_eval.py --mode omnijev
  支持模式：omnijev、qwen-score、qwen-generate。

仅重新汇总：
  source /data/users/sunziang/omnijev-tools/activate.sh
  python /data/users/sunziang/omnijev-tools/libero_report.py

关键文件：
  protocol.json：标签规则、随机种子、轨迹划分、数据版本、题集哈希。
  dev.jsonl：20 条开发轨迹，100 个状态、200 道题，用于图片/坐标/符号核查。
  test.jsonl：50 条测试轨迹，250 个状态、500 道题。
  images/*-panels.png：两个模型共用的双相机输入；不含未来帧或标准答案。
  audit-contact-sheet.png：10 个任务的开发集图片检查，含未来帧，仅用于标签审计。
  *-predictions.jsonl：每个状态的标签、模型输出、正确性、耗时和 token 数。
  *-run.json：模型/环境/输入/脚本哈希、预热耗时、显存与完成状态。
  comparison.json：准确率、宏F1、多数类基线、混淆矩阵、配对结果、轨迹 bootstrap 置信区间。
  report.txt：中文摘要。只有三种模式完整完成且通过配对一致性检查后才会生成。

解释边界：
  本评测不是官网 robot_long 八类题的复现，不测闭环机器人成功率。
  方向题预测未来五个记录步的净位移，使用固定工作台相机坐标。
  夹爪题预测 actions[t+1,6] 的打开/关闭命令，包括重复命令。
  发布模型与这些演示的训练重叠未知。
  Qwen 候选概率是限定字母集合内的条件概率，不能直接当成校准后的正确概率。
  三种配置共用画面、问题语义、候选答案及 BF16 精度；接口实现与内部提示格式不同。
  耗时反映当前具体实现，包含双题的图片处理，不是隔离的架构速度结论。
