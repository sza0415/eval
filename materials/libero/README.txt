LIBERO-10 · OmniJev / Qwen3.5 配对评测

此文件夹统一保存本次 LIBERO 评测的本地材料。
评测版本：libero10-paired-v1-seed20261010（2026-10-10）。

打开报告
  双击 omnijev-qwen-libero-offline.html。
  页面包含全部 500 道测试题、250 个状态的图片、成绩和逐题模型回答。
  数据、图片和脚本均内嵌在 HTML 中，无需联网或启动服务器。

目录说明
  omnijev-qwen-libero-offline.html  最终离线网页，日常查看用这个文件
  results/                       三种模型配置的预测、成绩、协议和运行记录
  results/viewer-data.json        构建离线网页所用的完整数据及嵌入式图片
  scripts/                       服务器端评测与导出脚本的本地备份
  docs/                          评测说明、协议、依赖版本及标签审计图
  report-source/                 网页模板、构建脚本与构建校验记录
  report-source/dist/index.html   构建输出，与最终离线网页内容相同
  exports/                       网页筛选结果的 JSON 导出示例

重新构建网页（仅使用本地结果，不执行模型推理）
  在本文件夹运行：python3 report-source/build.py
  会更新最外层的离线网页及 report-source/dist/index.html。

服务器位置
  scripts/ 中的评测脚本保留原有服务器绝对路径；此处是归档备份。
  代码：/data/users/sunziang/omnijev-tools/
  数据：/ssd-public/ai_infra/public/share/users/ziangsun/datasets/LIBERO/libero_10/
  结果：/ssd-public/ai_infra/public/share/users/ziangsun/OmniJev-experiments/libero10-paired-v1-seed20261010/
  完整评测命令及限制见 docs/libero-evaluation-README.txt。

整理记录
  原先散落于 OmniJev运行记录/ 的 LIBERO 文件均已移入此文件夹。
  网页检查时下载的 libero10-filtered-questions.json 已归入 exports/。
  此次整理未重新执行评测；服务器上的原始数据与结果路径未调整。
