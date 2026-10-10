# OmniJev / Qwen3.5 评测场景

统一展示 LIBERO、Mario、WEB、WEBTEST 的真实对照评测记录，以及五子棋数据集准备进度。
每个模型共完成 1,686 道题；五子棋尚未运行推理，不包含在该总数中。

## 目录

- `site/`：可直接托管的静态网页，`site/index.html` 为统一入口。
- `site/libero/`：500 题机器人评测；含 Qwen 打分与文本生成配置。
- `site/mario/`：800 题游戏动作评测。
- `site/web/`、`site/webtest/`：195 / 191 题网页操作及候选元素评测。
- `site/gomoku/`：数据集说明、审计预览、统计和校验报告；未运行推理。
- `materials/`：按场景归档的原始结果、协议、依赖记录和复现脚本。
- `scripts/prepare_viewers.py`：把原始离线 HTML 转成静态网页并拆出原图。
- `scripts/build.py`：校验已跟踪的网页和图片，并输出 `dist/`。

截图拆分后保持原始字节，图片以 SHA256 命名；不压缩或更改模型输入。
`site/import-manifest.json` 保留来源报告校验值、全部图片校验值及数据验证记录。
逐题预测、概率、标签和协议在场景网页内保留；仅将图像数据 URI 改为相对资源路径。

## 本地查看和部署

Python 3.9+，无需安装依赖，不调用模型。

```sh
python3 scripts/build.py
python3 -m http.server 8769 --bind 127.0.0.1 --directory dist
```

打开 `http://127.0.0.1:8769/`。也可以直接打开 `site/index.html`；保持 `site/` 内的目录结构。
任意静态托管服务均可使用 `dist/`，没有 API、数据库或 GPU 运行需求。
所有导航与图片路径均为相对路径，支持仓库子路径部署。

GitHub Pages 使用 `.github/workflows/pages.yml`：

1. 在仓库 Settings → Pages 将 Source 设为 GitHub Actions。
2. 推送到 `main` 后自动校验并发布 `dist/`。
3. Actions 成功后通过该工作流返回的网址访问。

## 更新评测记录

`materials/` 中较大的原始单文件离线报告和包含同一组图片的 LIBERO 数据包保留在本地，并由 `.gitignore` 排除。
远端托管使用已提交的 `site/` 文件，部署时无需原始离线报告。
更新本地报告后运行：

```sh
python3 scripts/prepare_viewers.py
python3 scripts/build.py
```

原始服务器路径记录在各场景的说明和协议中。材料里的服务器脚本是归档备份，仍使用既有服务器路径。
此仓库整理与部署不重新执行模型评测。

## 结果解释

这些实验不是官网历史榜单的原样复现，未评测闭环成功率。
公开轨迹与模型训练数据的重合情况未知。
WEB 使用 Mind2Web train，WEBTEST 使用 test_task；元素题有 oracle 正样本候选。
不同场景的首页百分比对应不同指标，不能据此直接排序模型的整体能力。
页面中的网页截图仅用于评测分析，与原网站无隶属关系。
