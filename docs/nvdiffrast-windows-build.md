# NVDiffRast Windows 构建说明

本文记录在 Windows 上为 GenZI 构建 `nvdiffrast` 的环境要求和常见报错。当前项目固定使用：

```text
Python 3.8
torch==2.0.1+cu117
CUDA Toolkit 11.7
```

`nvdiffrast` 是 CUDA/source build 扩展，安装时会调用本机 `nvcc` 和 MSVC `cl.exe` 编译。仅安装 NVIDIA driver 或 conda 的 `pytorch-cuda=11.7` runtime 不够；构建机需要完整 CUDA Toolkit 和 C++ 编译工具链。

## 快速检查

在安装前确认 PyTorch、CUDA Toolkit 和 MSVC 都指向预期版本：

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
where.exe nvcc
nvcc --version
where.exe cl
```

期望结果：

```text
torch 2.0.1+cu117 / CUDA 11.7
nvcc from CUDA\v11.7
cl.exe from Visual Studio C++ tools
```

如果 `where.exe cl` 找不到编译器，请安装 Visual Studio Build Tools 的 C++ build tools 和 Windows SDK，并从 x64 Native Tools Command Prompt / Developer PowerShell 中运行安装命令。

## 推荐安装命令

在 x64 Native Tools Command Prompt / Developer PowerShell 中运行：

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
uv run python tools/install_nvdiffrast.py --cuda-home $env:CUDA_HOME
```

安装脚本会临时为构建子进程设置 `CUDA_HOME`、`CUDA_PATH`、`PATH` 和 `DISTUTILS_USE_SDK`，并会先安装 `setuptools` / `wheel` / `ninja`。如果遇到较新 MSVC 与 CUDA 11.x 的版本检查冲突，可改用：

```powershell
uv run python tools/install_nvdiffrast.py --cuda-home $env:CUDA_HOME --allow-unsupported-msvc
```

安装后验证：

```powershell
uv run python -c "import nvdiffrast.torch as dr; print('nvdiffrast import ok', dr.__name__)"
```

## 常见报错

### CUDA 版本不匹配

报错：

```text
The detected CUDA version (...) mismatches the version that was used to compile PyTorch (11.7)
```

原因是当前 `nvcc` 不是 CUDA Toolkit 11.7。例如机器上同时安装了 CUDA 13.x/12.x，且这些版本排在 `PATH` 前面。

处理方式：

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
$env:CUDA_PATH = $env:CUDA_HOME
$env:Path = "$env:CUDA_HOME\bin;$env:CUDA_HOME\libnvvp;$env:Path"
```

重新检查：

```powershell
where.exe nvcc
nvcc --version
```

### 找不到 cl.exe

报错或警告：

```text
Error checking compiler version for cl
where.exe cl 找不到结果
```

说明当前 shell 没有加载 MSVC 编译环境。请安装 Visual Studio Build Tools 的 C++ build tools 和 Windows SDK，然后使用 x64 Native Tools Command Prompt / Developer PowerShell，或先调用 `vcvars64.bat`。

### DISTUTILS_USE_SDK 未设置

报错：

```text
VC environment is activated but DISTUTILS_USE_SDK is not set
```

处理方式：

```powershell
$env:DISTUTILS_USE_SDK = "1"
```

### MSVC 版本过新

报错：

```text
unsupported Microsoft Visual Studio version
STL1002: Unexpected compiler version, expected CUDA 12.4 or newer
```

这是 CUDA Toolkit 11.7 与较新的 VS 2022/MSVC toolset 的版本检查冲突。可临时加入：

```powershell
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
```

更稳的做法是使用 CUDA 11.7 官方支持范围内的旧 MSVC toolset。使用上述绕过参数时，应在安装后至少做一次 import 验证。

## 构建产物和运行环境

`nvcc`、MSVC `cl.exe`、Windows SDK 主要是构建机需要。编译完成后，运行环境通常只需要：

```text
Python 3.8
torch==2.0.1+cu117
已编译好的 nvdiffrast wheel / 已安装的 nvdiffrast
可用的 NVIDIA driver
必要的 VC++ runtime
```

不要为了运行环境改装最新 CUDA Toolkit。关键是编译产物、Python ABI、PyTorch CUDA ABI 保持一致。

如需给其他机器复用，可以在构建机先打 wheel：

```powershell
uv pip wheel git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation -w wheelhouse
```

然后在运行机安装生成的 wheel：

```powershell
uv pip install wheelhouse\nvdiffrast-*.whl
```
