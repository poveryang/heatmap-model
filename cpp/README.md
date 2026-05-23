# C++：Tengine 端侧推理与部署

基于 Tengine 的热力图模型端侧推理，当前主平台为 **imx8plus**（TIM-VX NPU）。

## 目录结构

```text
cpp/
  include/ src/ test/             HeatMapGenerator 与 HMAP-TEST
  thirdparty/tengine/             Tengine 头文件与预编译 .so（lib 需自行放置）
  scripts/
    convert/
      onnx_to_tmfile.sh           ONNX → fp32 tmfile
      quantize_uint8.sh           fp32 → uint8 量化
    build/
      cross_build.sh              Docker 交叉编译
    deploy/
      pipeline.sh                 一键部署链路
      run_on_board.sh             板端 SSH 推理
    dev/
      validate_local.sh           本地 x86 验证
  profiles/                       各平台 Docker 转换参数
  board.env.example               板端 SSH 配置样例
  artifacts/                      模型与推理输出（gitignore）
  build/                          编译产物（gitignore）
```

## 依赖

- Docker 镜像：
  - `model-convert-tool:imx8plus` — ONNX 转 tmfile 与 uint8 量化
  - `compiler:imx8plus` — C++ 交叉编译
- `cpp/thirdparty/tengine/lib/{aarch64,x86}/` 需自行放置预编译 Tengine 库
- 板端 SSH 访问（配置见 `board.env.example`）

## 一键部署

```bash
cp cpp/board.env.example cpp/board.env   # 编辑 BOARD_HOST、BOARD_PASSWORD 等

bash cpp/scripts/deploy/pipeline.sh \
  --onnx /path/to/model.onnx \
  --calib /path/to/calib_images/ \
  --image /path/to/test.png \
  --config cpp/board.env
```

流程：`ONNX → fp32 tmfile → uint8 tmfile → cross build → 板端 NPU 推理`

中间产物默认写入 `cpp/artifacts/tmfile/`，板端输出写入 `cpp/artifacts/imx8plus/`。

## 分步执行

```bash
# 1. ONNX → fp32 tmfile
bash cpp/scripts/convert/onnx_to_tmfile.sh \
  --onnx /path/to/model.onnx \
  --out cpp/artifacts/tmfile/model-fp32.tmfile

# 2. fp32 → uint8 量化
bash cpp/scripts/convert/quantize_uint8.sh \
  --fp32 cpp/artifacts/tmfile/model-fp32.tmfile \
  --calib /path/to/calib_images/ \
  --out cpp/artifacts/tmfile/model-uint8.tmfile

# 3. 交叉编译
bash cpp/scripts/build/cross_build.sh

# 4. 部署到板端
HEATMAP_BOARD_CONFIG=cpp/board.env \
  bash cpp/scripts/deploy/run_on_board.sh \
  --context timvx \
  --precision uint8 \
  --model cpp/artifacts/tmfile/model-uint8.tmfile \
  --image /path/to/test.png \
  --out-dir cpp/artifacts/imx8plus \
  --min-hot 1
```

## 板端配置

`cpp/board.env`（已 gitignore）中的关键字段：

| 字段 | 说明 |
| --- | --- |
| `BOARD_HOST` / `BOARD_SSH_PORT` | 板端 IP 与 SSH 端口 |
| `CONTEXT=timvx` | NPU（TIM-VX）；`cpu` 为纯 CPU 回退 |
| `PRECISION=uint8` | 量化精度 |
| `TENGINE_RUNTIME_MODE=minimal` | 仅上传 `libtengine-lite.so` |

## 本地 x86 验证

```bash
bash cpp/scripts/dev/validate_local.sh \
  --model /path/to/model-uint8.tmfile \
  --image /path/to/test.png
```

## 量化参数与第三方量化表

当前 C++ uint8 路径直接把 raw uint8 图像送入 Tengine，因此 Tengine
输入量化参数必须等价于 FP32/PyTorch 预处理：
`(x / 255 - 0.4329) / 0.2349`。对应参数为：

```bash
--mean 110.3895,110.3895,110.3895 \
--scale 0.01669463,0.01669463,0.01669463
```

这组参数已经写入 `cpp/profiles/imx8plus.env` 和
`cpp/scripts/convert/quantize_uint8.sh` 的默认值。不要再使用
`--scale 0.2349,0.2349,0.2349`，那会让 uint8 输入反量化到错误范围。

如果必须使用第三方导出的量化表，目前兼容性最明确的候选是
**MQBench + Tengine 导入表**：

1. MQBench 使用 `BackendType.Tengine_u8` 做 PTQ 校准并导出 `*_for_tengine.scale`。
2. `convert_tool` 将 MQBench 导出的 `*_for_tengine.onnx` 转成 fp32 tmfile。
3. `quant_tool_uint8 -f` 导入 MQBench scale 表并生成 uint8 tmfile。
4. 每个方案用独立目录保存模型、scale、metadata 和板端推理结果。

```bash
# 1. 导出 MQBench Tengine scale 表
conda run -n hmap python python/scripts/quant/mqbench_quant.py \
  --ckpt cpp/artifacts/ckpt/hmap-v2-epoch=499-val_loss=3.828e-04.ckpt \
  --exp hmap-v2 \
  --calib cpp/artifacts/calib_from_test \
  --out-dir cpp/artifacts/quant_eval/mqbench-symw-512 \
  --model-name hmap_mqbench_symw_512 \
  --observer default \
  --weight-qscheme symmetric \
  --calib-limit 512

# 2. MQBench ONNX -> fp32 tmfile
bash cpp/scripts/convert/onnx_to_tmfile.sh \
  --onnx cpp/artifacts/quant_eval/mqbench-symw-512/hmap_mqbench_symw_512_for_tengine.onnx \
  --out cpp/artifacts/quant_eval/mqbench-symw-512/model-fp32.tmfile

# 3. 导入 MQBench scale 表生成 uint8 tmfile
bash cpp/scripts/convert/quantize_uint8.sh \
  --fp32 cpp/artifacts/quant_eval/mqbench-symw-512/model-fp32.tmfile \
  --calib cpp/artifacts/quant_eval/calib-512-even \
  --scale-file cpp/artifacts/quant_eval/mqbench-symw-512/hmap_mqbench_symw_512_for_tengine.scale \
  --out cpp/artifacts/quant_eval/mqbench-symw-512/model-uint8.tmfile

# 4. 板端批量推理，结果单独落盘
HEATMAP_BOARD_CONFIG=cpp/board.env \
  bash cpp/scripts/deploy/run_on_board.sh \
  --context timvx \
  --precision uint8 \
  --tengine-runtime none \
  --model cpp/artifacts/quant_eval/mqbench-symw-512/model-uint8.tmfile \
  --input-dir python/infer-datasets/focus \
  --out-dir cpp/artifacts/quant_eval/mqbench-symw-512/focus-results
```

`mqbench_quant.py` 默认会把输入量化参数强制改成部署预处理一致的形式：
`scale = 1 / (std * 255)`，`zero_point = round(mean * 255)`。这与 C++ uint8
路径直接传 raw uint8 图像匹配；如果保持 MQBench 观测到的输入分布，需显式传
`--no-force-input-qparams`。

`mqbench_quant.py` 还默认使用 `--weight-qscheme symmetric`。不要用 MQBench
`Tengine_u8` 原始 asymmetric weight 默认值；本模型上会把负权重裁到正范围，
导出的 ONNX 本身就会发散，表现为粗块状白/灰热图。

本轮实测结果保存在 `cpp/artifacts/quant_eval/`：

| 方案 | 结果目录 | 结论 |
| --- | --- | --- |
| FP32 基线 | `fp32-tengine-cpu/focus-results-clean` | Tengine CPU FP32 参考输出 |
| 厂商 KL + 修正输入参数 | `vendor-kl-normalized/focus-results-clean` | 简单、稳定，接近 FP32 |
| 厂商 KL 原参数 | `vendor-kl-current/focus-results-clean` | 与 FP32 差距明显，不推荐 |
| MQBench Tengine 原始 weight 默认值 | `mqbench-ema-512/focus-results-clean` | 导出 ONNX 已发散，不推荐 |
| MQBench symmetric weight scale 导入 | `mqbench-symw-512/focus-results-clean` | 本轮最接近 FP32 的第三方量化表方案 |

## 注意事项

- `cpp/src/hmap_generator.cpp` 已兼容 Tengine 输出 NCHW/NHWC。
- `--tengine-runtime minimal` 时 Vivante/OpenVX 走板端原生环境。
- imx8plus 日志中可能出现 TIM-VX EVIS shader 编译告警，当前 uint8 + timvx 推理正常。
- 如果要复查厂商直量化的输入归一化，当前 FP32/PyTorch 预处理为
  `(x / 255 - 0.4329) / 0.2349`，对应 Tengine raw 输入参数约为
  `--mean 110.3895,110.3895,110.3895 --scale 0.01669463,0.01669463,0.01669463`。

---

## 附录 A：构建 Docker 镜像

### model-convert-tool:imx8plus

```bash
cat > /tmp/Dockerfile.imx8plus-convert <<'EOF'
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git libopencv-dev \
    libprotobuf-dev protobuf-compiler && rm -rf /var/lib/apt/lists/*
ARG TENGINE_REPO=https://github.com/OAID/Tengine.git
ARG TENGINE_REF=tengine-lite
WORKDIR /opt
RUN git clone --depth 1 --branch "${TENGINE_REF}" "${TENGINE_REPO}" tengine
WORKDIR /opt/tengine/build
RUN cmake \
    -DTENGINE_BUILD_CONVERT_TOOL=ON \
    -DTENGINE_BUILD_QUANT_TOOL=ON \
    -DCMAKE_INSTALL_PREFIX=/opt/tengine/build/install \
    .. && make -j"$(nproc)" && make install
ENV PATH="/opt/tengine/build/install/bin:${PATH}"
WORKDIR /workspace
EOF

docker build --platform linux/amd64 \
  -f /tmp/Dockerfile.imx8plus-convert \
  -t model-convert-tool:imx8plus .
```

### compiler:imx8plus

C++ 交叉编译镜像，需自行准备（含 aarch64 工具链与 OpenCV）。

### 其他平台（预留）

| 平台 | 本地镜像 | Profile |
| --- | --- | --- |
| NovaIC | `model-convert-tool:novaic` | `cpp/profiles/novaic.env` |
| Allwinner | `model-convert-tool:allwinner` | `cpp/profiles/allwinner.env` |

---

## 附录 B：待办与已知问题

### imx8plus 端侧推理

- [ ] 排查 TIM-VX EVIS shader 编译告警（`cl_viv_vx_ext.h` / resize kernel）。
  - 当前 `uint8 + timvx` 与 `uint8 + cpu` 输出接近，但日志仍可能出现 shader 编译错误。
  - 需确认相关算子是否 fallback 到 CPU，或是否需调整板端 Vivante/OpenVX 运行库。

### 其他平台

- [ ] 接入 NovaIC 转换工具：确认镜像内 CLI、IO 格式与量化流程。
- [ ] 接入 Allwinner 转换工具：同上。
