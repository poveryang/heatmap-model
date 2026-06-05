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
  hmap_*.png            验证集可视化
```

可选参数：

```bash
python python/train.py --exp hmap-v2 --pretrained /path/to/base.ckpt
python python/train.py --exp hmap-v2 --resume /path/to/last.ckpt
```

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
