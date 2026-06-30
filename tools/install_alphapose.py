"""Install or prepare an external AlphaPose checkout for GenZI.

The script intentionally does not download model weights.  AlphaPose pose and
detector checkpoints still need to be prepared according to their licenses and
upstream documentation.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHAPOSE_URL = "https://github.com/MVIG-SJTU/AlphaPose.git"
DEFAULT_TARGET = REPO_ROOT / "external" / "AlphaPose"


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="克隆并安装 GenZI 使用的 AlphaPose 外部依赖。"
    )
    parser.add_argument(
        "--target",
        default=str(DEFAULT_TARGET),
        help="AlphaPose checkout 目录，默认: external/AlphaPose",
    )
    parser.add_argument(
        "--rev",
        default="master",
        help="AlphaPose git revision，默认: master。建议在可复现环境中传入固定 commit。",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="只克隆/切换 revision，不运行 setup.py build develop。",
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    display = " ".join(cmd)
    location = f" cwd={cwd}" if cwd is not None else ""
    print(f"[*] {display}{location}")
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def uv_command() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("未找到 uv。请先安装 uv，或确保 uv 在 PATH 中。")
    return uv


def ensure_checkout(target: Path, rev: str) -> None:
    target = target.resolve()
    if target.exists():
        if not (target / ".git").is_dir():
            raise RuntimeError(
                f"{target} 已存在但不是 git checkout。请换一个 --target，或手动清理该目录。"
            )
        run(["git", "-C", str(target), "fetch", "--all", "--tags"])
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", ALPHAPOSE_URL, str(target)])

    run(["git", "-C", str(target), "checkout", rev])


def install_alphapose(target: Path) -> None:
    print("[*] 安装前会固定 AlphaPose 常见兼容构建依赖。")
    build_packages = [
        "Cython==0.29.35",
        "pytest-runner",
        "setuptools==65.7.0",
        "wheel",
    ]
    run([uv_command(), "pip", "install", "--python", sys.executable, *build_packages])

    run(
        [
            uv_command(),
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "cython-bbox==0.1.5",
        ]
    )
    if platform.system() == "Windows":
        run([uv_command(), "pip", "install", "--python", sys.executable, "pycocotools==2.0.7"])
    run([sys.executable, "setup.py", "build", "develop"], cwd=target)


def print_next_steps(target: Path) -> None:
    target = target.resolve()
    print()
    print("[完成] AlphaPose checkout 已准备。")
    print(f"AlphaPose 目录: {target}")
    print()
    print("后续还需要手动准备模型/权重文件：")
    print(
        "- pose checkpoint: "
        f"{target / 'pretrained_models' / 'noface_fast50_dcn_combined_256x192.pth'}"
    )
    print(
        "- pose config: "
        f"{target / 'configs' / 'halpe_68_noface' / 'resnet' / '256x192_res50_lr1e-3_2x-dcn-combined.yaml'}"
    )
    print("- detector 权重: 由 config/alphapose.yml 的 detector: yolo 和 AlphaPose 自身配置决定。")
    print()
    print("如果运行时找不到 detector 包，请确认 editable/develop 安装成功，或临时设置：")
    print(f"PowerShell: $env:PYTHONPATH = \"{target};$env:PYTHONPATH\"")
    print(f"bash: export PYTHONPATH=\"{target}:$PYTHONPATH\"")
    print()
    print("推荐检查：")
    print("uv run python tools/check_special_deps.py --run-cfg config/sketchfab_gen.yml")


def main() -> int:
    configure_stdio()
    args = parse_args()
    target = Path(args.target)
    if not target.is_absolute():
        target = REPO_ROOT / target

    print("GenZI AlphaPose 安装辅助脚本")
    print("AlphaPose 上游通常没有稳定 release；默认使用 master，可用 --rev 固定 commit。")
    print("本脚本不会下载 pose checkpoint 或 detector 权重。")
    print()

    ensure_checkout(target, args.rev)
    if args.skip_install:
        print("[*] --skip-install 已启用，跳过 setup.py build develop。")
    else:
        install_alphapose(target)

    print_next_steps(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
