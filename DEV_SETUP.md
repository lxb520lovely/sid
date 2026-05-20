# Local VSCode Setup

这个项目已经配置好 VSCode 的 Python 解释器、调试配置和常用任务。

## 解释器

项目现在使用本地虚拟环境：

```bash
/Users/xiaobin.li/Documents/sid/.venv/bin/python
```

这个 `.venv` 是用 Codex bundled Python 创建的，并开启了 `--system-site-packages`，所以它能直接继承原解释器里已经安装好的依赖：

```text
numpy
python-docx
pillow
matplotlib
seaborn
scikit-learn
scipy
```

如果 VSCode 没自动识别解释器：

1. 打开命令面板：`Cmd+Shift+P`
2. 选择 `Python: Select Interpreter`
3. 选择或粘贴 `.venv/bin/python`

## 安装新依赖

以后在当前项目里安装新库，用 `.venv` 的 pip：

```bash
.venv/bin/python -m pip install 包名
```

例如：

```bash
.venv/bin/python -m pip install numpy
```

安装 requirements：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

新装的包会进入项目目录下的 `.venv`，不会污染系统 Python。

## 推荐插件

VSCode 会提示安装这些推荐插件：

```text
ms-python.python
ms-python.debugpy
ms-python.vscode-pylance
```

至少安装 `Python` 插件后，Run/Debug 才会最顺。

## 直接运行脚本

在 VSCode Terminal 里运行：

```bash
.venv/bin/python build_rqopq_sid.py \
  --input shopee_tittle_emb.npy \
  --output-dir rqopq_sid_out_debug \
  --code-dim 32 \
  --rq-clusters 512 \
  --rq-levels 3 \
  --opq-subspaces 2 \
  --opq-clusters 256
```

评估：

```bash
.venv/bin/python evaluate_sid_quality.py \
  --input shopee_tittle_emb.npy \
  --sid-dir rqopq_sid_out \
  --output-dir rqopq_sid_out/eval_debug \
  --max-scatter-points 1500
```

## 在 VSCode 里 Debug

1. 打开左侧 `Run and Debug`
2. 选择一个配置：
   - `Build RQ-OPQ SID (default)`
   - `Evaluate SID Quality`
   - `Build RQ-KMeans SID`
   - `Create DOCX Report`
   - `Python: Current File`
3. 在代码左侧行号处打断点
4. 按 `F5`

默认 debug 输出会写到 `*_debug` 目录，避免覆盖已经生成好的正式结果。

## 用 Tasks 运行

命令面板 `Cmd+Shift+P` -> `Tasks: Run Task`，选择：

```text
Build RQ-OPQ SID
Evaluate SID Quality
Create DOCX Report
```

## 重建 venv

如果 `.venv` 被删掉，可以用下面命令重建：

```bash
/Users/xiaobin.li/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

`umap-learn` 暂时没有放进 requirements，因为当前 macOS + Python 3.12 环境下它依赖的 `llvmlite` 编译失败。评估脚本已经自动 fallback 到 t-SNE/PCA，不影响当前可视化。
