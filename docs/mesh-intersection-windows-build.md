# mesh_intersection / torch-mesh-isect 构建说明

本文记录为 GenZI 构建 `mesh_intersection` 的环境要求和常见报错。`mesh_intersection` 来自上游 `torch-mesh-isect`，用于 `loss.self_intersect_weights > 0` 时计算 SMPL-X human self-intersection penalty。

当前项目固定使用：

```text
Python 3.8
torch==2.0.1+cu117
CUDA Toolkit 11.7
```

`mesh_intersection` 是 CUDA/source build 扩展，安装时会调用本机 `nvcc` 和 MSVC `cl.exe` 编译。它比 NVDiffRast 多一个前置条件：上游源码需要 `helper_math.h`，所以必须提供 `CUDA_SAMPLES_INC`，或通过安装脚本的 `--cuda-samples-inc` 指定包含该文件的 CUDA Samples include 目录。

## 快速检查

在安装前确认 PyTorch、CUDA Toolkit、MSVC 和 CUDA Samples include 都可用：

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
where.exe nvcc
nvcc --version
where.exe cl
Test-Path "$env:CUDA_SAMPLES_INC\helper_math.h"
```

期望结果：

```text
torch 2.0.1+cu117 / CUDA 11.7
nvcc from CUDA\v11.7
cl.exe from Visual Studio C++ tools
helper_math.h exists
```

如果 `CUDA_SAMPLES_INC` 没有设置，可以从 NVIDIA CUDA Samples 准备该目录。若 CUDA Samples clone 在 `D:\deps\cuda-samples`，通常传入：

```powershell
--cuda-samples-inc D:\deps\cuda-samples\Common
```

## 推荐安装命令

在 x64 Native Tools Command Prompt / Developer PowerShell 中运行：

```powershell
uv run python tools/install_mesh_intersection.py --target external/torch-mesh-isect --cuda-samples-inc D:\deps\cuda-samples\Common
```

也可以先设置环境变量，再省略参数：

```powershell
$env:CUDA_SAMPLES_INC = "D:\deps\cuda-samples\Common"
uv run python tools/install_mesh_intersection.py --target external/torch-mesh-isect
```

安装脚本会 clone 固定 revision 的 `https://github.com/vchoutas/torch-mesh-isect.git`，临时为构建子进程设置 `CUDA_SAMPLES_INC`，并用当前 `uv` 环境执行：

```powershell
uv pip install --python <current-python> --no-build-isolation <checkout>
```

安装后验证：

```powershell
uv run python -c "import mesh_intersection.bvh_search_tree; import mesh_intersection.loss; print('mesh_intersection import ok')"
```

## 常见报错

### 未设置 CUDA_SAMPLES_INC

报错：

```text
未设置 CUDA_SAMPLES_INC
```

处理方式是传入 `--cuda-samples-inc <path>`，或设置 `CUDA_SAMPLES_INC`。该路径必须直接包含 `helper_math.h`，不是 CUDA Toolkit 根目录。

### 找不到 helper_math.h

报错：

```text
CUDA Samples include 目录无效
```

说明传入的目录不对。请确认：

```powershell
Test-Path "<path>\helper_math.h"
```

返回 `True` 后再重新运行安装脚本。

### 找不到 nvcc

报错：

```text
未找到 nvcc
```

说明当前 shell 没有找到完整 CUDA Toolkit。请安装 CUDA Toolkit 11.7，并确认 `nvcc` 在 `PATH` 中，或设置：

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
$env:CUDA_PATH = $env:CUDA_HOME
$env:Path = "$env:CUDA_HOME\bin;$env:CUDA_HOME\libnvvp;$env:Path"
```

### 找不到 cl.exe

报错：

```text
未找到 MSVC cl.exe
```

请安装 Visual Studio Build Tools 的 C++ build tools 和 Windows SDK，然后使用 x64 Native Tools Command Prompt / Developer PowerShell，或先调用 `vcvars64.bat`。

### 不想启用 self-intersection loss

如果只是想先跑通 generation，可以把配置里的 `loss.self_intersect_weights` 全部设为 0。这样 `genzi.loss.HSILoss` 会跳过 `mesh_intersection` lazy import，不需要安装该扩展，但最终人体自交约束会被关闭。

## 构建产物和运行环境

`nvcc`、MSVC `cl.exe`、Windows SDK 和 CUDA Samples include 主要是构建机需要。编译完成后，运行环境通常只需要：

```text
Python 3.8
torch==2.0.1+cu117
已安装的 mesh_intersection / bvh_cuda
可用的 NVIDIA driver
必要的 VC++ runtime
```

关键是编译产物、Python ABI、PyTorch CUDA ABI 保持一致。不要为了运行环境改装最新 CUDA Toolkit。
