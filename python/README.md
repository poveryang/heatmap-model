# Python：热力图模型训练

PyTorch Lightning 训练、评估、推理与 ONNX 导出。

## 目录结构

```text
python/
  train.py / eval.py / infer.py   入口脚本
  configs/                        实验 YAML（hmap-v1、hmap-smoke 等）
  hmap/                           核心包（dataset / model / utils）
  scripts/
    export_onnx.py                导出 ONNX
    install_mqbench.sh            安装 MQBench（GitHub，非 PyPI）
    prepare_dataset.py            labelme → train/test 数据集
    quant/mqbench_quant.py        MQBench 量化导出
    dev/profile_model.py          模型 FLOPs 分析
    dev/data_misc.py              旧数据集辅助工具
  runs/                           训练输出（gitignore）
  requirements.txt
```

## 环境

推荐使用 Conda 环境 `hmap`（Python 3.10）：

```bash
conda create -n hmap python=3.10 -y
conda activate hmap
pip install -r python/requirements.txt
bash python/scripts/install_mqbench.sh
```

或使用 venv：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
bash python/scripts/install_mqbench.sh
```

`mqbench` 不在 PyPI 上，需通过 `install_mqbench.sh` 从 GitHub 安装。脚本默认拉取
[ModelTC/MQBench](https://github.com/ModelTC/MQBench) 的 `main` 分支，可用
`--ref` 指定 tag 或 commit。

## 训练

```bash
python python/train.py --exp hmap-smoke
```

每次训练会在 `python/runs/` 下创建以 **日期时间 + 配置名** 命名的目录，例如：

```text
python/runs/20250521_153045_hmap-smoke/
  checkpoints/          模型 checkpoint
  metrics.csv           训练指标
  hmap_*.png            验证集 object heatmap 可视化（heatmap/manifest.txt）
  qroi_*.png            验证集 Q_roi 可视化（固定 val batch，GT ROI）
```

可选参数：

```bash
python python/train.py --exp hmap-v2 --pretrained /path/to/base.ckpt
python python/train.py --exp hmap-v2 --resume /path/to/last.ckpt
python python/train.py --exp hmap-barcode-qroi-v2 --wandb
python python/train.py --exp hmap-barcode-qroi-v2 --no-wandb
```

### 后台稳定训练（推荐）

使用 tmux 启动，Mac 离线后任务仍继续，日志写入 run 目录：

```bash
bash python/scripts/train_daemon.sh hmap-barcode-qroi-v2
bash python/scripts/train_daemon.sh hmap-barcode-qroi-v2 --wandb

# 查看实时日志
tail -f python/runs/<run_name>/train.log

# 重新 attach 到 tmux
tmux attach -t hmap-hmap-barcode-qroi-v2
```

### 本地维护代码，GPU 服务器训练

推荐规则：

1. 本地只负责改代码、改 YAML、提交 Git。
2. GPU 服务器只负责同步同一 Git 分支并运行训练。
3. 数据集、checkpoint、`python/runs/` 留在服务器，不进 Git。
4. 每次训练目录会保存 `run.env`、`config.yaml`、`train.log`，方便回看本次配置与 commit。

从 Windows 本地一键提交远端训练：

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote_train.ps1 start hmap-barcode-qroi-v3
```

脚本默认会：

- 推送当前 Git 分支到 `origin`
- SSH 到 `yjunj@10.80.31.40`
- 在 `/home/yjunj/projects/heatmap-model` 更新同一分支
- 用 `python/scripts/train_daemon.sh` 在 tmux 后台启动训练

常用操作：

```powershell
# 启动并额外传训练参数
powershell -ExecutionPolicy Bypass -File tools/remote_train.ps1 start hmap-barcode-qroi-v3 -- --wandb

# 查看远端训练状态和最近日志路径
powershell -ExecutionPolicy Bypass -File tools/remote_train.ps1 status hmap-barcode-qroi-v3

# 跟随最新一次训练日志
powershell -ExecutionPolicy Bypass -File tools/remote_train.ps1 tail hmap-barcode-qroi-v3 -Follow

# 停止这个实验名对应的 tmux 训练
powershell -ExecutionPolicy Bypass -File tools/remote_train.ps1 stop hmap-barcode-qroi-v3
```

如果服务器路径、环境名不同，可覆盖：

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote_train.ps1 start hmap-barcode-qroi-v3 `
  -Remote yjunj@10.80.31.40 `
  -RemoteRepo /home/yjunj/projects/heatmap-model `
  -CondaEnv hmap
```

第一次使用前，服务器上需确认：

```bash
cd /home/yjunj/projects/heatmap-model
git status
conda activate hmap
pip install -r python/requirements.txt
bash python/scripts/install_mqbench.sh
wandb login  # 如需 W&B
```

### Weights & Biases

1. 安装并登录（一次性）：

```bash
pip install wandb
wandb login
```

2. 在 YAML 中启用（见 `hmap-barcode-qroi-v2.yaml` 的 `wandb:` 段），或用 CLI 覆盖。

`entity` 和 `name` **可不写**：登录后默认使用你的 W&B 账号；只有上传到团队项目时才需要设置 `entity`。

训练指标与 `hmap_*.png` / `qroi_*.png` 会自动同步到 W&B 项目。

> 注意：`wandb login` 需在**训练所在的远程机器**上执行，Mac 本地登录不会同步到服务器。

### 自定义验证可视化样本

heatmap 与 q_roi **均通过 manifest.txt 指定**，图像必须来自 `test/` 集（与 Q_roi 标签坐标一致，不使用 `img_aug`）。

```text
{root_dir}/viz/
  heatmap/manifest.txt    # 1-8 行
  qroi/manifest.txt       # 1-4 行
```

| 类型 | 文件 | 数量 | 要求 |
|------|------|------|------|
| heatmap | `viz/heatmap/manifest.txt` | 1–8 | 每行一条路径，须存在于 `test/test.txt` |
| qroi | `viz/qroi/manifest.txt` | 1–4 | 同上，且含 ROI / Q_roi 标注 |

`manifest.txt` 示例（路径与 `test/test.txt` 分号前一致）：

```text
large_code/00110.png
large_code/00326.png
```

数量上限与网格列数由 YAML 控制：

```yaml
data:
  viz:
    heatmap_max_images: 8
    qroi_max_images: 4
    grid_cols: 2
```

多图以 **网格** 排列（q_roi 顶部另有汇总信息栏）。manifest 缺失或为空时训练启动会报错。

**当前默认行为：**

| 文件 | 数据来源 |
|------|----------|
| `hmap_{epoch}.png` | `viz/heatmap/manifest.txt` |
| `qroi_{epoch}.png` | `viz/qroi/manifest.txt` |

### Geometry + Q_roi 实验

`hmap-geo-qroi-smoke` 会输出 3 个 object heatmap 通道和 6 个 geometry 通道，并在训练时用
GT rotated ROI feature 回归 `Q_roi`。模型采用 CSP 风格 backbone、PAN/FPN neck、
object/geometry decoupled dense head：

```bash
python python/train.py --exp hmap-geo-qroi-smoke
```

`data.quality_path` 可指向 SDK 分支生成的 `roi_quality.jsonl`。每行至少包含：

```json
{"image_id": "train/foo.png", "roi_id": 0, "q_roi": 0.75, "quality_mask": true}
```

`quality_mask=false` 或缺失质量标签的 ROI 不参与 `Q_roi` loss，但仍作为 object heatmap
正样本参与训练。

#### 验证可视化

每个 validation epoch 结束（global rank 0）会在 run 目录写入两类图片：

| 文件 | 数据来源 | 内容 |
|------|----------|------|
| `hmap_{epoch}.png` | `viz/heatmap/manifest.txt` | 3 通道 object heatmap 与输入叠加（网格排列） |
| `qroi_{epoch}.png` | `viz/qroi/manifest.txt` | GT rotated ROI 框、汇总栏 + 网格面板 |

`qroi_*.png` 使用与训练相同的 letterbox resize 和 ROI 坐标；`quality_mask=false` 的 ROI
以灰色虚线框标注 `MASKED`，且不计入汇总 MAE。若 batch 内无有效 `Q_roi`，图片会标注
`No valid Q_roi`。

## 评估与推理

```bash
python python/eval.py --ckpt python/runs/.../checkpoints/model.ckpt
python python/infer.py --ckpt /path/to/model.ckpt --image /path/to/image.png
```

## 数据准备

将 labelme 标注的 png/json 转为训练目录结构：

```bash
python python/scripts/prepare_dataset.py \
  --source /path/to/labelme_root \
  --dest /path/to/dataset_root
```

生成的 `train/`、`test/` 目录在 YAML 配置的 `data.root_dir` 中引用。

## 导出 ONNX

```bash
python python/scripts/export_onnx.py \
  --exp hmap-v2 \
  --ckpt python/runs/.../checkpoints/model.ckpt \
  --out cpp/artifacts/onnx/model.onnx
```

导出的 ONNX 供 [cpp/README.md](../cpp/README.md) 中的 Tengine 转换链路使用。

## 开发工具

```bash
# 模型计算量分析（需 thop）
python python/scripts/dev/profile_model.py

# MQBench 量化导出（需先运行 install_mqbench.sh）
python python/scripts/quant/mqbench_quant.py
```
