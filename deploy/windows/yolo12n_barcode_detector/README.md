# YOLO12n 条码检测模型 Windows 转换包

这个目录用于把检测模型传到 Windows 上做模型转换和 uint8 量化。它独立于 `cpp/README.md`
里的 HMAP 热力图模型链路；不要复用热力图模型的输入形状、均值或方差参数。

## 目录内容

```text
deploy/windows/yolo12n_barcode_detector/
  yolo12n_barcode_detector.onnx   # 需要传到 Windows 的检测模型
  classes.txt                     # 类别顺序：bar, qr, dm
  model_config.yaml               # 输入输出、预处理、量化参数
  manifest.json                   # 模型来源、哈希、评估指标
  collect_calibration_images.py   # 从验证集收集代表性校准图片
  calibration/
    .gitignore                    # 校准图片本地生成，不提交到 git
```

## 需要传到 Windows 的文件

将整个 `deploy/windows/yolo12n_barcode_detector/` 目录复制到 Windows。正式量化前，先在
Linux 本机生成校准图片：

```bash
/home/yjunj/miniforge3/envs/hmap/bin/python \
  deploy/windows/yolo12n_barcode_detector/collect_calibration_images.py \
  --count 512
```

执行后会生成：

```text
calibration/images/
calibration/calibration_manifest.txt
```

然后再把整个目录复制到 Windows。校准图片建议 256 到 512 张，尽量覆盖 `bar`、`qr`、`dm`
和不同光照、尺寸、角度。

## 模型规格

| 项目 | 值 |
| --- | --- |
| 任务 | 目标检测 |
| 输入名 | `images` |
| 输入形状 | `1,3,640,640` |
| 输入颜色 | RGB |
| 输入预处理 | letterbox 到 640x640，像素除以 255，不做 mean/std |
| 输出名 | `output0` |
| 输出形状 | `1,7,8400` |
| 输出含义 | `4 bbox + 3 class scores` |
| 后处理 | 置信度阈值 0.25，NMS IoU 0.45 |
| 类别 | `bar`, `qr`, `dm` |

YOLO 输出不包含内置 NMS，板端推理代码需要单独做解码、阈值过滤和 NMS。

## Tengine 转换参考

如果使用仓库里的 Tengine 脚本转换，参考 `cpp/README.md` 的流程，但参数要按检测模型覆盖：

```bash
bash cpp/scripts/convert/onnx_to_tmfile.sh \
  --onnx deploy/windows/yolo12n_barcode_detector/yolo12n_barcode_detector.onnx \
  --out cpp/artifacts/tmfile/yolo12n_barcode_detector-fp32.tmfile

bash cpp/scripts/convert/quantize_uint8.sh \
  --fp32 cpp/artifacts/tmfile/yolo12n_barcode_detector-fp32.tmfile \
  --calib deploy/windows/yolo12n_barcode_detector/calibration/images \
  --out cpp/artifacts/tmfile/yolo12n_barcode_detector-uint8.tmfile \
  --shape 1,3,640,640 \
  --mean 0,0,0 \
  --scale 0.0039215686,0.0039215686,0.0039215686
```

关键差异：

- 检测模型输入是 `1,3,640,640`，不是 HMAP 的 `1,400,640`。
- 检测模型预处理是 `x / 255`，因此量化参数是 `mean=0`、`scale=1/255`。
- 不要使用 HMAP 热力图参数 `mean=110.3895`、`scale=0.01669463`。

## Windows 转换工具填写建议

如果使用 Windows 上的厂商 GUI 或命令行工具，按下面填写：

```text
model: yolo12n_barcode_detector.onnx
input_name: images
input_shape: 1,3,640,640
input_format: NCHW
color_order: RGB
normalize: x / 255.0
mean: 0,0,0
scale: 0.0039215686,0.0039215686,0.0039215686
calibration_dir: calibration/images
output_name: output0
output_shape: 1,7,8400
```

转换完成后建议产物命名：

```text
yolo12n_barcode_detector-fp32.tmfile
yolo12n_barcode_detector-uint8.tmfile
```

