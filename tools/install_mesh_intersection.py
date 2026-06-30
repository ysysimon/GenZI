"""Install the external torch-mesh-isect dependency for GenZI."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
MESH_INTERSECTION_URL = "https://github.com/vchoutas/torch-mesh-isect.git"
DEFAULT_REV = "85b30177821a1527e3fe62fcf8ce65262d7c1879"
DEFAULT_TARGET = REPO_ROOT / "external" / "torch-mesh-isect"
HELPER_MATH_HEADER = "helper_math.h"


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="克隆并安装 GenZI 使用的 torch-mesh-isect CUDA 扩展。"
    )
    parser.add_argument(
        "--target",
        default=str(DEFAULT_TARGET),
        help="torch-mesh-isect checkout 目录，默认: external/torch-mesh-isect",
    )
    parser.add_argument(
        "--rev",
        default=DEFAULT_REV,
        help="torch-mesh-isect git revision，默认固定到已验证的上游 HEAD。",
    )
    parser.add_argument(
        "--cuda-samples-inc",
        default=None,
        help=(
            "包含 helper_math.h 的 CUDA Samples include 目录；"
            "未提供时读取 CUDA_SAMPLES_INC 环境变量。"
        ),
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="只克隆/切换 revision，不运行 CUDA extension 构建安装。",
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="安装时向 uv pip install 传递 --force-reinstall。",
    )
    return parser.parse_args(argv)


def run(
    cmd: list[str],
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    display = " ".join(cmd)
    location = f" cwd={cwd}" if cwd is not None else ""
    print(f"[*] {display}{location}")
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        check=True,
    )


def uv_command() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("未找到 uv。请先安装 uv，或确认 uv 在 PATH 中。")
    return uv


def _resolve_existing_file(path: Path, filename: str) -> Path:
    resolved = path.expanduser().resolve()
    expected = resolved / filename
    if not expected.is_file():
        raise RuntimeError(
            f"CUDA Samples include 目录无效: {resolved}。"
            f"请确认该目录下存在 {filename}。"
        )
    return resolved


def resolve_cuda_samples_inc(
    cli_value: str | None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    env = os.environ if environ is None else environ
    value = cli_value or env.get("CUDA_SAMPLES_INC")
    if not value:
        raise RuntimeError(
            "未设置 CUDA_SAMPLES_INC。请传入 --cuda-samples-inc，"
            f"或设置 CUDA_SAMPLES_INC 指向包含 {HELPER_MATH_HEADER} 的目录。"
        )
    return _resolve_existing_file(Path(value), HELPER_MATH_HEADER)


def find_nvcc(environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return nvcc

    executable = "nvcc.exe" if platform.system() == "Windows" else "nvcc"
    for key in ("CUDA_HOME", "CUDA_PATH"):
        root = env.get(key)
        if not root:
            continue
        candidate = Path(root) / "bin" / executable
        if candidate.is_file():
            return str(candidate)
    return None


def check_cuda_toolkit(environ: Mapping[str, str] | None = None) -> None:
    nvcc = find_nvcc(environ)
    if nvcc is None:
        raise RuntimeError(
            "未找到 nvcc。torch-mesh-isect 需要 CUDA Toolkit 编译 CUDA extension；"
            "请安装与当前 PyTorch CUDA 版本兼容的 CUDA Toolkit，并确认 nvcc 在 PATH "
            "中，或设置 CUDA_HOME / CUDA_PATH。"
        )
    print(f"[*] nvcc: {nvcc}")


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
        raise RuntimeError(
            "当前 torch 不是 CUDA build，无法构建 torch-mesh-isect CUDA extension。"
        )
    print(f"[*] torch: {torch.__version__}, CUDA: {torch_cuda}")


def ensure_checkout(target: Path, rev: str) -> None:
    target = target.resolve()
    if target.exists():
        if not (target / ".git").is_dir():
            raise RuntimeError(
                f"{target} 已存在但不是 git checkout。"
                "请换一个 --target，或手动清理该目录。"
            )
        run(["git", "-C", str(target), "fetch", "--all", "--tags"])
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", MESH_INTERSECTION_URL, str(target)])

    run(["git", "-C", str(target), "checkout", rev])


def install_mesh_intersection(
    target: Path,
    cuda_samples_inc: Path,
    *,
    force_reinstall: bool = False,
) -> None:
    env = os.environ.copy()
    env["CUDA_SAMPLES_INC"] = str(cuda_samples_inc.resolve())

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
    cmd.append(str(target.resolve()))
    run(cmd, env=env)


def verify_imports() -> None:
    for module_name in ("mesh_intersection.bvh_search_tree", "mesh_intersection.loss"):
        importlib.import_module(module_name)
        print(f"[*] import OK: {module_name}")


def print_next_steps(target: Path, cuda_samples_inc: Path | None) -> None:
    print()
    print("[完成] torch-mesh-isect 已准备。")
    print(f"checkout 目录: {target.resolve()}")
    if cuda_samples_inc is not None:
        print(f"CUDA_SAMPLES_INC: {cuda_samples_inc.resolve()}")
    print()
    print("推荐检查:")
    print("uv run python tools/check_special_deps.py --run-cfg config/sketchfab_gen.yml")


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)
    target = Path(args.target)
    if not target.is_absolute():
        target = REPO_ROOT / target

    print("GenZI torch-mesh-isect 安装辅助脚本")
    print(f"上游仓库: {MESH_INTERSECTION_URL}")
    print(f"默认 revision: {DEFAULT_REV}")
    print()

    try:
        uv_command()
        if args.skip_install:
            cuda_samples_inc = None
        else:
            cuda_samples_inc = resolve_cuda_samples_inc(args.cuda_samples_inc)
            check_torch()
            check_cuda_toolkit()
            check_msvc_compiler()

        ensure_checkout(target, args.rev)
        if args.skip_install:
            print("[*] --skip-install 已启用，跳过 CUDA extension 构建安装。")
        else:
            assert cuda_samples_inc is not None
            install_mesh_intersection(
                target,
                cuda_samples_inc,
                force_reinstall=args.force_reinstall,
            )
            verify_imports()

        print_next_steps(target, cuda_samples_inc)
    except subprocess.CalledProcessError as exc:
        print(f"[错误] 命令执行失败，退出码 {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:  # noqa: BLE001 - setup helper should surface all failures.
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
