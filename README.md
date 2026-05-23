# heatmap-model

热力图模型 monorepo：Python 负责训练与 ONNX 导出，C++ 负责 Tengine 端侧推理与部署。

| 子项目 | 说明 |
| --- | --- |
| [python/](python/README.md) | PyTorch Lightning 训练、评估、推理、ONNX 导出 |
| [cpp/](cpp/README.md) | Tengine 模型转换、交叉编译、板端 NPU 推理 |

## 快速开始

**训练**（见 [python/README.md](python/README.md)）：

```bash
conda activate hmap   # pip install -r python/requirements.txt && bash python/scripts/install_mqbench.sh
python python/train.py --exp hmap-smoke
```

**imx8plus 部署**（见 [cpp/README.md](cpp/README.md)）：

```bash
cp cpp/board.env.example cpp/board.env   # 编辑板端 IP/密码

bash cpp/scripts/deploy/pipeline.sh \
  --onnx /path/to/model.onnx \
  --calib /path/to/calib_images/ \
  --image /path/to/test.png \
  --config cpp/board.env
```

## 生成物目录

| 路径 | 内容 | gitignore |
| --- | --- | --- |
| `python/runs/` | 训练日志、checkpoint、可视化 | 是 |
| `cpp/artifacts/` | tmfile、板端推理输出 | 是 |
| `cpp/build/` | 交叉编译产物 | 是 |
