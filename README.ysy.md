# GenZI 本地入口说明

这份文档是 YSynthetic 本地工作流入口。原始论文说明保留在 `README.md`；这里主要记录如何用 `uv` 管理 GenZI 的基础 Python 环境，以及哪些 CUDA / 源码编译依赖需要单独处理。

## 1. 基础环境

安装 `uv` 后，在仓库根目录执行：

```bash
uv python install 3.8
uv sync --frozen
```

`.python-version` 固定为 Python 3.8，`uv.lock` 用于复现基础依赖。默认 PyTorch 版本为 `torch==2.0.1`，并通过 PyTorch CUDA 11.7 wheel index 安装。

## 2. 常用入口

运行 Sketchfab generation：

```bash
uv run python -m genzi.generation run_cfg=config/sketchfab_gen.yml
```

运行 Sketchfab evaluation：

```bash
uv run python -m genzi.evaluation run_cfg=config/sketchfab_eval.yml
```

运行 PROX-S generation / evaluation：

```bash
uv run python -m genzi.generation run_cfg=config/proxs_gen.yml
uv run python -m genzi.evaluation run_cfg=config/proxs_eval.yml
```

查询 Houdini 导出的 GenZI dense SDF：

```bash
uv run python gen_sdf/probe_genzi_sdf.py --sdf gen_sdf/export/scene.json --point 0 0 0
```

Houdini 侧 dense SDF 导出脚本位于：

```text
gen_sdf/export_genzi_dense_sdf.py
gen_sdf/export_genzi_dense_sdf_callback.py
gen_sdf/export_genzi_dense_sdf_hython.py
```

## 3. 运行环境变量

运行 GenZI 前建议设置：

```bash
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYOPENGL_PLATFORM=egl
```

Windows PowerShell 对应写法：

```powershell
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:WANDB_MODE = "offline"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYOPENGL_PLATFORM = "egl"
```

## 4. 特殊依赖

下面这些依赖和 CUDA、平台 wheel、源码编译或第三方仓库绑定较深，不放进基础 `uv sync` 里一键锁定。基础环境同步后，再按需要单独安装。

SMPL-X / VPoser：

```bash
uv pip install --no-deps smplx==0.1.28 git+https://github.com/nghorbani/human_body_prior.git
```

NVDiffRast：

```bash
uv pip install git+https://github.com/NVlabs/nvdiffrast.git
```

xFormers：

```bash
uv pip install xformers==0.0.20
```

`xformers` 当前不作为基础依赖安装；代码没有直接 import 它，通常只在 diffusion 后端优化时才需要。

Open3D：

```bash
uv pip install open3d==0.10.0.0
```

如果 `open3d==0.10.0.0` 在当前平台没有可用 wheel，优先参考原 README 的 conda 安装方式：

```bash
conda install -y open3d-admin::open3d=0.10.0.0
```

PyTorch3D：

```text
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
```

AlphaPose：

```text
https://github.com/MVIG-SJTU/AlphaPose
```

如果 AlphaPose 编译报错，可优先尝试原 README 建议的版本：

```bash
uv pip install Cython==0.29.35 setuptools==65.7.0
```

torch-mesh-isect：

```text
https://github.com/vchoutas/torch-mesh-isect
```

这些依赖安装后可用下面的命令做 import 检查：

```bash
uv run python -c "import smplx; import human_body_prior; import nvdiffrast; import open3d; import pytorch3d; import alphapose; import mesh_intersection; print('special deps ok')"
```

## 5. 基础验证

检查 PyTorch 版本：

```bash
uv run python -c "import torch; print(torch.__version__)"
```

检查 SDF probe CLI：

```bash
uv run python gen_sdf/probe_genzi_sdf.py --help
```

如果 `uv sync --frozen` 失败，先检查当前平台是否支持 `torch==2.0.1/cu117` 对应的 wheel。`tensorflow`、`torchaudio`、`SharedArray` 和 `xformers` 没有在当前代码里直接使用，已不放入基础 uv 环境；如后续确认某条路径需要，再按需单独安装。
