"""Install the external NVDiffRast dependency for GenZI."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


NVDIFFRAST_URL = "https://github.com/NVlabs/nvdiffrast.git"
DEFAULT_REV = "253ac4fcea7de5f396371124af597e6cc957bfae"
BUILD_HELPERS = ["setuptools", "wheel", "ninja"]
WINDOWS_NVCC_COMPAT_FLAGS = (
    "-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="安装 GenZI 使用的 NVDiffRast CUDA 扩展。"
    )
    parser.add_argument(
        "--rev",
        default=DEFAULT_REV,
        help="NVDiffRast git revision，默认固定到已验证的上游 HEAD。",
    )
    parser.add_argument(
        "--cuda-home",
        default=None,
        help=(
            "CUDA Toolkit 根目录；未提供时优先读取 CUDA_HOME / CUDA_PATH，"
            "再尝试从 PATH 中的 nvcc 推断。"
        ),
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="安装 NVDiffRast 时向 uv pip install 传递 --force-reinstall。",
    )
    parser.add_argument(
        "--skip-build-helpers",
        action="store_true",
        help="跳过 setuptools / wheel / ninja 安装步骤。",
    )
    parser.add_argument(
        "--allow-unsupported-msvc",
        action="store_true",
        help=(
            "Windows 上临时设置 NVCC_PREPEND_FLAGS，绕过 CUDA 11.x 与较新 MSVC "
            "版本检查。仅在遇到 unsupported compiler 报错时使用。"
        ),
    )
    return parser.parse_args(argv)


def run(
    cmd: list[str],
    env: Mapping[str, str] | None = None,
) -> None:
    print(f"[*] {' '.join(cmd)}")
    subprocess.run(cmd, env=dict(env) if env is not None else None, check=True)


def uv_command() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("未找到 uv。请先安装 uv，或确认 uv 在 PATH 中。")
    return uv


def _nvcc_executable() -> str:
    return "nvcc.exe" if platform.system() == "Windows" else "nvcc"


def _cuda_home_from_nvcc(nvcc_path: str | None) -> Path | None:
    if nvcc_path is None:
        return None
    nvcc = Path(nvcc_path).resolve()
    if nvcc.parent.name.lower() != "bin":
        return None
    return nvcc.parent.parent


def _validate_cuda_home(cuda_home: Path) -> Path:
    resolved = cuda_home.expanduser().resolve()
    nvcc = resolved / "bin" / _nvcc_executable()
    if not nvcc.is_file():
        raise RuntimeError(
            f"CUDA Toolkit 目录无效: {resolved}。未找到 {nvcc}。"
        )
    return resolved


def resolve_cuda_home(
    cli_value: str | None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    env = os.environ if environ is None else environ
    if cli_value:
        return _validate_cuda_home(Path(cli_value))

    for key in ("CUDA_HOME", "CUDA_PATH"):
        value = env.get(key)
        if value:
            return _validate_cuda_home(Path(value))

    inferred = _cuda_home_from_nvcc(shutil.which("nvcc"))
    if inferred is not None:
        return _validate_cuda_home(inferred)

    raise RuntimeError(
        "未找到 CUDA Toolkit。请传入 --cuda-home，或设置 CUDA_HOME / CUDA_PATH，"
        "并确认 nvcc 可用。"
    )


def check_msvc_compiler() -> None:
    if platform.system() != "Windows":
        return
    if shutil.which("cl.exe") is None:
        raise RuntimeError(
            "未找到 MSVC cl.exe。Windows 上请先打开 x64 Native Tools Command Prompt，"
            "或配置 Visual Studio C++ Build Tools 环境变量后再运行。"
        )
    print("[*] MSVC cl.exe: OK")


def check_torch() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 环境无法导入 torch。请先运行 `uv sync --frozen`。"
        ) from exc

    torch_cuda = getattr(torch.version, "cuda", None)
    if torch_cuda is None:
        raise RuntimeError("当前 torch 不是 CUDA build，无法构建 NVDiffRast。")
    print(f"[*] torch: {torch.__version__}, CUDA: {torch_cuda}")


def build_install_env(
    cuda_home: Path,
    *,
    allow_unsupported_msvc: bool = False,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    cuda_home = cuda_home.resolve()
    cuda_bin = cuda_home / "bin"
    cuda_libnvvp = cuda_home / "libnvvp"

    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    env["PATH"] = os.pathsep.join(
        [str(cuda_bin), str(cuda_libnvvp), env.get("PATH", "")]
    )
    if platform.system() == "Windows":
        env.setdefault("DISTUTILS_USE_SDK", "1")
        if allow_unsupported_msvc:
            existing = env.get("NVCC_PREPEND_FLAGS")
            env["NVCC_PREPEND_FLAGS"] = (
                f"{WINDOWS_NVCC_COMPAT_FLAGS} {existing}".strip()
                if existing
                else WINDOWS_NVCC_COMPAT_FLAGS
            )
    return env


def install_build_helpers(env: Mapping[str, str]) -> None:
    run(
        [
            uv_command(),
            "pip",
            "install",
            "--python",
            sys.executable,
            *BUILD_HELPERS,
        ],
        env=env,
    )


def install_nvdiffrast(
    rev: str,
    env: Mapping[str, str],
    *,
    force_reinstall: bool = False,
) -> None:
    package_spec = f"git+{NVDIFFRAST_URL}@{rev}"
    cmd = [
        uv_command(),
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-build-isolation",
    ]
    if force_reinstall:
        cmd.append("--force-reinstall")
    cmd.append(package_spec)
    run(cmd, env=env)


def verify_imports() -> None:
    module = importlib.import_module("nvdiffrast.torch")
    print(f"[*] import OK: {module.__name__}")


def print_next_steps(cuda_home: Path) -> None:
    print()
    print("[完成] NVDiffRast 已准备。")
    print(f"CUDA_HOME: {cuda_home.resolve()}")
    print()
    print("推荐检查:")
    print("uv run python tools/check_special_deps.py --run-cfg config/sketchfab_gen.yml")


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)

    print("GenZI NVDiffRast 安装辅助脚本")
    print(f"上游仓库: {NVDIFFRAST_URL}")
    print(f"默认 revision: {DEFAULT_REV}")
    print()

    try:
        uv_command()
        cuda_home = resolve_cuda_home(args.cuda_home)
        check_torch()
        check_msvc_compiler()

        env = build_install_env(
            cuda_home,
            allow_unsupported_msvc=args.allow_unsupported_msvc,
        )
        if args.skip_build_helpers:
            print("[*] --skip-build-helpers 已启用，跳过 build helper 安装。")
        else:
            install_build_helpers(env)
        install_nvdiffrast(
            args.rev,
            env,
            force_reinstall=args.force_reinstall,
        )
        verify_imports()
        print_next_steps(cuda_home)
    except subprocess.CalledProcessError as exc:
        print(f"[错误] 命令执行失败，退出码 {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:  # noqa: BLE001 - setup helper should surface all failures.
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
