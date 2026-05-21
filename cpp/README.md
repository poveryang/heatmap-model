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

## 注意事项

- `cpp/src/hmap_generator.cpp` 已兼容 Tengine 输出 NCHW/NHWC。
- `--tengine-runtime minimal` 时 Vivante/OpenVX 走板端原生环境。
- imx8plus 日志中可能出现 TIM-VX EVIS shader 编译告警，当前 uint8 + timvx 推理正常。

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
