# GenZI 本地入口说明

这份文档是 YSynthetic 本地工作流入口。原始论文说明保留在 `README.md`；这里主要记录如何用 `uv` 管理 GenZI 的基础 Python 环境，以及哪些 CUDA / 源码编译依赖需要单独处理。

## 1. 基础环境

安装 `uv` 后，在仓库根目录执行：

```bash
uv python install 3.8
uv sync --frozen
```

`.python-version` 固定为 Python 3.8，`uv.lock` 用于复现基础依赖。默认 PyTorch 版本为 `torch==2.0.1`，并通过 PyTorch CUDA 11.7 wheel index 安装。

`uv sync` 默认会把当前 `.venv` 精确同步到本次选择的依赖集合：缺少的包会安装，不在集合里的包会被移除。这个行为适合创建干净基础环境，但如果当前环境里已经通过脚本或源码编译安装了 `nvdiffrast`、`mesh_intersection`、AlphaPose 等特殊依赖，普通 `uv sync --frozen` 可能会把它们卸载。

常用同步方式：
```bash
# 干净复现基础环境；可能移除额外安装的特殊依赖
uv sync --frozen

# 日常补齐基础/dev 依赖，同时保留当前 .venv 里的额外包
uv sync --frozen --inexact

# 安装 generation 依赖组，并保留脚本/源码安装的特殊依赖
uv sync --frozen --group generation --inexact
```

如果不确定当前命令会改动哪些包，先用 `--dry-run` 查看计划：
```bash
uv sync --frozen --dry-run
uv sync --frozen --group generation --inexact --dry-run
```

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

基础环境同步后，generation 推荐先安装专用依赖组：

```bash
uv sync --frozen --group generation --inexact
```

这个 group 会安装：

```text
smplx==0.1.28
human_body_prior @ 4c246d8a83ce16d3cff9c79dcf04d81fa440a6bc
open3d==0.10.0.0  # 仅 Windows/Linux x86_64 类平台
pywinpty==2.0.13  # 仅 Windows，用于避开 2.0.14 sdist 元数据问题
```

`human_body_prior` 固定到旧 commit，是因为当前上游默认说明已经面向更新 Python 版本，而 GenZI 当前环境固定为 Python 3.8。

Windows 上 `open3d==0.10.0.0` 会经由 `notebook -> jupyter-server -> terminado` 间接拉入 `pywinpty`。`pywinpty==2.0.14` 在 Python 3.8 的锁定结果里只有 sdist，且其 `pyproject.toml` 缺少 `project.version`，`uv` 会拒绝构建；因此本项目在 Windows 下显式固定 `pywinpty==2.0.13`，该版本有 `cp38` 的 `win_amd64` wheel。

如果 `open3d==0.10.0.0` 在当前平台没有可用 wheel，使用 conda fallback：

```bash
conda install -y open3d-admin::open3d=0.10.0.0
```

下面这些依赖仍然需要单独处理，因为它们依赖 CUDA/source build、外部仓库布局或额外权重文件。

NVDiffRast：

```bash
uv run python tools/install_nvdiffrast.py
```

该脚本会先安装 `setuptools` / `wheel` / `ninja`，再按上游方式用 `--no-build-isolation` 构建 CUDA extension。若 `nvcc` 不在 `PATH` 中，可传 `--cuda-home <CUDA Toolkit 根目录>`；Windows 上需要 MSVC `cl.exe`，遇到 CUDA 11.x 与较新 MSVC 的版本检查冲突时可追加 `--allow-unsupported-msvc`。详见 [NVDiffRast Windows 构建说明](docs/nvdiffrast-windows-build.md)。

AlphaPose：

```bash
uv run python tools/install_alphapose.py --target external/AlphaPose
```

AlphaPose 上游通常没有稳定 release；辅助脚本默认 checkout `master`，需要复现实验时可以传 `--rev <commit>` 固定版本。脚本会通过 `uv pip` 安装旧式 `setup.py` 需要的构建辅助包，并在 Windows 上预装 `pycocotools==2.0.7` 和 `cython-bbox==0.1.5`，避免 AlphaPose 内部调用 `python -m pip`。脚本不会下载 pose checkpoint 或 detector 权重，这些文件仍需按 AlphaPose 文档手动准备。

torch-mesh-isect：

```bash
uv run python tools/install_mesh_intersection.py --target external/torch-mesh-isect
```

该脚本会 clone 固定 revision 的 `https://github.com/vchoutas/torch-mesh-isect.git`，并用当前 `uv` 环境构建安装 `mesh_intersection` CUDA extension。运行前请先准备 CUDA Toolkit / `nvcc`，并设置 `CUDA_SAMPLES_INC`，或传入 `--cuda-samples-inc <path>`；该路径必须包含 CUDA Samples 里的 `helper_math.h`。当前默认 generation 配置启用了 `loss.self_intersect_weights`，所以需要该扩展；如果只想先跳过这个依赖，可以把 self-intersection 权重全部设为 0，代码会跳过 `mesh_intersection`。详见 [mesh_intersection 构建说明](docs/mesh-intersection-windows-build.md)。

xFormers：

```text
xformers 当前不作为必需依赖安装；代码没有直接 import 它，通常只在 diffusion 后端优化时才需要。
```

PyTorch3D：

```text
当前 GenZI 已使用本地 torch rotation helper 替代 PyTorch3D 的两个 rotation 函数，不再需要单独安装 PyTorch3D。
```

如果手动安装 AlphaPose，构建辅助包至少需要包含：

```bash
uv pip install Cython==0.29.35 pytest-runner setuptools==65.7.0 wheel
uv pip install --no-deps cython-bbox==0.1.5
uv pip install pycocotools==2.0.7  # Windows
```

推荐用 doctor 脚本做完整检查，不再用单行 import 命令：

```powershell
uv run python tools/check_special_deps.py --run-cfg config/sketchfab_gen.yml
uv run python tools/check_special_deps.py --run-cfg config/proxs_gen.yml
```

## 5. 模型与权重文件准备

安装 Python 包并不会自动准备所有模型文件。当前 generation 配置还需要下面这些本地模型、权重或资产文件。

需要手动准备的模型 / 资产：

| 用途 | 当前配置期待路径 | 说明 |
| --- | --- | --- |
| SMPL-X model | `./data/smpl-x/models_smplx_v1_1` | `smplx` 只提供 Python API，不会下载 SMPL-X 模型文件。请按 SMPL-X 官方许可下载后放到该目录。 |
| VPoser checkpoint | `./data/smpl-x/vposer_V02_05` | `human_body_prior` 只提供 VPoser 代码，不会自动下载 checkpoint。请准备兼容 GenZI 的 VPoser V02_05 目录。 |
| SMPL-X UV template | `./data/smpl-x/smplx_uv_template.txt` | 用于保存带 UV 的 `optim_human.obj`。 |
| SMPL-X texture | `./data/smpl-x/smplx_texture_f_alb_1024.png` | 用于保存带贴图的 SMPL-X mesh。 |
| AlphaPose pose checkpoint | AlphaPose 安装目录下的 `pretrained_models/noface_fast50_dcn_combined_256x192.pth` | 路径来自 `config/alphapose.yml` 的 `checkpoint` 字段，代码会按 AlphaPose 安装目录拼接。 |
| AlphaPose detector 权重 | 由 `config/alphapose.yml` 的 `detector: "yolo"` 和 AlphaPose 自身 detector 配置决定 | 请按 AlphaPose 官方安装说明准备对应 detector 权重；本仓库不自动下载。 |

运行时会自动下载或读取缓存的模型：

| 用途 | 配置值 | 下载方式 |
| --- | --- | --- |
| Stable Diffusion inpainting | `sd2-community/stable-diffusion-2-inpainting` | `diffusers` 首次运行 `from_pretrained` 时从 Hugging Face 下载，或读取本地 cache；这是当前可访问的 SD2 inpainting mirror。 |
| CLIP | `openai/clip-vit-base-patch32` | `transformers` 首次运行 `from_pretrained` 时从 Hugging Face 下载，或读取本地 cache。 |

只需要安装库、不需要额外模型文件的特殊依赖：

```text
nvdiffrast
open3d
mesh_intersection / torch-mesh-isect
```

`pytorch3d` 当前已不再需要安装；`xformers` 是可选 diffusion 优化依赖，不属于必需项。

推荐在运行 generation 前使用检查脚本一次性确认依赖和模型文件：

```powershell
uv run python tools/check_special_deps.py --run-cfg config/sketchfab_gen.yml
uv run python tools/check_special_deps.py --run-cfg config/proxs_gen.yml
```

脚本会区分 `OK`、`缺依赖`、`缺模型文件` 和 `可选项`。如果只想查看诊断结果、不希望缺失项导致非零退出码，可追加 `--no-fail`。

## 6. 基础验证

检查 PyTorch 版本和 CUDA 设备是否可用：

```bash
uv run python -c "import sys, torch; cuda_ok = torch.cuda.is_available(); device = torch.cuda.get_device_name(0) if cuda_ok else 'none'; print('torch:', torch.__version__); print('cuda available:', cuda_ok); print('cuda device:', device); sys.exit(0 if cuda_ok else 1)"
```

检查 SDF probe CLI：

```bash
uv run python gen_sdf/probe_genzi_sdf.py --help
```

检查开发工具依赖：
```bash
uv run --frozen python -m pytest tests
uv run --frozen ruff --version
```

如果 `uv sync --frozen` 失败，先检查当前平台是否支持 `torch==2.0.1/cu117` 对应的 wheel。`tensorflow`、`torchaudio`、`SharedArray` 和 `xformers` 没有在当前代码里直接使用，已不放入基础 uv 环境；如后续确认某条路径需要，再按需单独安装。
