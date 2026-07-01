"""Check GenZI special dependencies and model assets.

This script intentionally does not import genzi.generation.  It checks each
special dependency independently so one missing package does not hide the rest
of the setup issues.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
NVDIFFRAST_INSTALL_COMMAND = "uv run python tools/install_nvdiffrast.py"
NVDIFFRAST_INSTALL_HINT = (
    f"请运行 `{NVDIFFRAST_INSTALL_COMMAND}`。该脚本会安装 setuptools / wheel / "
    "ninja，并用 --no-build-isolation 构建 NVDiffRast。若 nvcc 不在 PATH 中，"
    "请传入 --cuda-home 或设置 CUDA_HOME / CUDA_PATH。"
)
MESH_INTERSECTION_INSTALL_COMMAND = (
    "uv run python tools/install_mesh_intersection.py "
    "--target external/torch-mesh-isect"
)
MESH_INTERSECTION_INSTALL_HINT = (
    f"请运行 `{MESH_INTERSECTION_INSTALL_COMMAND}`。"
    "该 CUDA extension 需要 CUDA Toolkit / nvcc；"
    "CUDA_SAMPLES_INC 或 --cuda-samples-inc 必须指向包含 helper_math.h 的 "
    "CUDA Samples include 目录。详见 docs/mesh-intersection-windows-build.md。"
)


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    required: bool = True


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 GenZI 特殊依赖、模型权重和配置资产是否准备好。"
    )
    parser.add_argument(
        "--run-cfg",
        default="config/sketchfab_gen.yml",
        help="generation 配置文件路径，默认: config/sketchfab_gen.yml",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="即使发现缺失项也返回 0，适合只想查看诊断结果的场景。",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "缺少 PyYAML，无法读取配置。请先运行 `uv sync --frozen`。"
        ) from exc

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} 不是有效的 YAML mapping。")
    return data


def get_nested(data: dict[str, Any], dotted_key: str) -> Any:
    value: Any = data
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def resolve_templates(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        resolved = value
        for key, replacement in context.items():
            if isinstance(replacement, (str, int, float)):
                resolved = resolved.replace("${" + key + "}", str(replacement))
        return resolved
    if isinstance(value, dict):
        return {key: resolve_templates(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_templates(item, context) for item in value]
    return value


def as_repo_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def check_import(label: str, module_name: str, hint: str) -> CheckResult:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - setup diagnostics should catch all.
        first_line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return CheckResult(
            name=label,
            status="缺依赖",
            detail=f"{module_name} 导入失败: {first_line}。{hint}",
        )
    return CheckResult(name=label, status="OK", detail=f"已成功导入 {module_name}。")


def check_optional_import(label: str, module_name: str, hint: str) -> CheckResult:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - setup diagnostics should catch all.
        first_line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return CheckResult(
            name=label,
            status="可选项",
            detail=f"{module_name} 当前不可用: {first_line}。{hint}",
            required=False,
        )
    return CheckResult(
        name=label,
        status="可选项",
        detail=f"已成功导入 {module_name}；当前 GenZI 代码没有直接使用它。",
        required=False,
    )


def check_path(
    label: str,
    path: Path | None,
    hint: str,
    expect_dir: bool,
    required: bool = True,
) -> CheckResult:
    missing_status = "缺模型文件" if required else "可选项"
    if path is None:
        return CheckResult(
            name=label,
            status=missing_status,
            detail=f"配置中未找到路径。{hint}",
            required=required,
        )
    if not path.exists():
        return CheckResult(
            name=label,
            status=missing_status,
            detail=f"未找到 {path}。{hint}",
            required=required,
        )
    if expect_dir and not path.is_dir():
        return CheckResult(
            name=label,
            status=missing_status,
            detail=f"{path} 存在但不是目录。{hint}",
            required=required,
        )
    if not expect_dir and not path.is_file():
        return CheckResult(
            name=label,
            status=missing_status,
            detail=f"{path} 存在但不是文件。{hint}",
            required=required,
        )
    kind = "目录" if expect_dir else "文件"
    return CheckResult(
        name=label,
        status="OK",
        detail=f"已找到 {kind}: {path}",
        required=required,
    )


def get_alphapose_root() -> Path | None:
    try:
        module = importlib.import_module("alphapose")
    except Exception:
        return None
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    return Path(module_file).resolve().parent.parent


def check_alphapose_assets(alpha_cfg_path: Path | None) -> list[CheckResult]:
    results: list[CheckResult] = []
    if alpha_cfg_path is None:
        return [
            CheckResult(
                name="AlphaPose 配置",
                status="缺模型文件",
                detail="run_cfg 中没有 pose2d.args_path，无法定位 AlphaPose 配置。",
            )
        ]

    results.append(
        check_path(
            "AlphaPose 本地配置",
            alpha_cfg_path,
            "该文件来自 run_cfg 的 pose2d.args_path。",
            expect_dir=False,
        )
    )
    if not alpha_cfg_path.exists():
        return results

    alpha_cfg = load_yaml(alpha_cfg_path)
    alpha_root = get_alphapose_root()
    checkpoint = alpha_cfg.get("checkpoint")
    cfg_file = alpha_cfg.get("cfg")
    detector = alpha_cfg.get("detector", "")

    if alpha_root is None:
        results.append(
            CheckResult(
                name="AlphaPose checkpoint",
                status="缺依赖",
                detail=(
                    "尚未安装或无法导入 alphapose，无法定位其安装目录。"
                    "安装 AlphaPose 后，请确认 checkpoint 位于其安装目录下的 "
                    f"{checkpoint}。"
                ),
            )
        )
        results.append(
            CheckResult(
                name="AlphaPose detector 权重",
                status="缺依赖",
                detail=(
                    f"当前 detector={detector!r}。请先安装 AlphaPose，再按其 detector "
                    "文档准备对应权重。"
                ),
            )
        )
        return results

    results.append(
        check_path(
            "AlphaPose checkpoint",
            alpha_root / str(checkpoint) if checkpoint else None,
            "该路径相对 AlphaPose 安装目录。",
            expect_dir=False,
        )
    )
    results.append(
        check_path(
            "AlphaPose 模型配置",
            alpha_root / str(cfg_file) if cfg_file else None,
            "该路径相对 AlphaPose 安装目录。",
            expect_dir=False,
        )
    )
    results.append(
        CheckResult(
            name="AlphaPose detector 权重",
            status="可选项",
            detail=(
                f"当前 detector={detector!r}。detector 权重路径由 AlphaPose 自己的 "
                "detector 配置决定，本脚本不硬编码；请按 AlphaPose 文档确认已准备。"
            ),
            required=False,
        )
    )
    return results


def check_auto_downloads(cfg: dict[str, Any]) -> list[CheckResult]:
    ldm_path = get_nested(cfg, "vlm.ldm_inpaint_path")
    clip_path = get_nested(cfg, "vlm.clip_path")
    results = [
        CheckResult(
            name="Stable Diffusion inpainting",
            status="可选项",
            detail=(
                f"配置值: {ldm_path!r}。首次运行会由 diffusers 从 Hugging Face "
                "下载或读取本地 cache；本脚本不强制检查 cache。"
            ),
            required=False,
        ),
        CheckResult(
            name="CLIP",
            status="可选项",
            detail=(
                f"配置值: {clip_path!r}。首次运行会由 transformers 从 Hugging Face "
                "下载或读取本地 cache；本脚本不强制检查 cache。"
            ),
            required=False,
        ),
    ]
    return results


def print_install_guidance() -> None:
    print("\n[推荐安装顺序]")
    print("1. 基础环境: uv sync --frozen")
    print("2. generation Python group: uv sync --group generation")
    print(f"3. NVDiffRast: {NVDIFFRAST_INSTALL_COMMAND}")
    print("4. AlphaPose: uv run python tools/install_alphapose.py --target external/AlphaPose")
    print(f"5. torch-mesh-isect: {MESH_INTERSECTION_INSTALL_COMMAND}")


def print_section(title: str, results: Iterable[CheckResult]) -> None:
    print(f"\n[{title}]")
    for result in results:
        print(f"{result.status} - {result.name}: {result.detail}")


def main() -> int:
    configure_stdio()
    args = parse_args()
    run_cfg_path = as_repo_path(args.run_cfg)
    if run_cfg_path is None or not run_cfg_path.is_file():
        print(f"缺模型文件 - run_cfg: 未找到配置文件 {args.run_cfg}", file=sys.stderr)
        return 1

    raw_cfg = load_yaml(run_cfg_path)
    cfg = resolve_templates(raw_cfg, raw_cfg)

    python_results = [
        check_import(
            "smplx",
            "smplx",
            "请运行 `uv sync --group generation`。",
        ),
        check_import(
            "human_body_prior / VPoser",
            "human_body_prior.models.vposer_model",
            (
                "请运行 `uv sync --group generation`；该 group 会固定到兼容 "
                "Python 3.8 的 human_body_prior commit。"
            ),
        ),
        check_import(
            "nvdiffrast",
            "nvdiffrast.torch",
            NVDIFFRAST_INSTALL_HINT,
        ),
        check_import(
            "open3d",
            "open3d",
            "请运行 `uv sync --group generation`，无 wheel 时使用 conda fallback。",
        ),
        check_import(
            "alphapose",
            "alphapose",
            "请按 AlphaPose 官方仓库安装，并准备其 checkpoint 和 detector 权重。",
        ),
        check_import(
            "AlphaPose cython_bbox",
            "cython_bbox",
            "请运行 `uv run python tools/install_alphapose.py --target external/AlphaPose`。",
        ),
        check_import(
            "AlphaPose pycocotools",
            "pycocotools.mask",
            "请运行 `uv run python tools/install_alphapose.py --target external/AlphaPose`。",
        ),
        check_import(
            "mesh_intersection",
            "mesh_intersection.bvh_search_tree",
            MESH_INTERSECTION_INSTALL_HINT,
        ),
        check_optional_import(
            "xformers",
            "xformers",
            "当前代码没有直接 import xformers，通常只作为 diffusion 后端优化可选项。",
        ),
    ]

    asset_results = [
        check_path(
            "SMPL-X model directory",
            as_repo_path(get_nested(cfg, "smplx.model_path")),
            "请手动下载 SMPL-X 模型并放到配置指定目录。",
            expect_dir=True,
        ),
        check_path(
            "VPoser checkpoint directory",
            as_repo_path(get_nested(cfg, "vposer.ckpt_path")),
            "请手动准备 VPoser checkpoint 目录。",
            expect_dir=True,
        ),
        check_path(
            "SMPL-X UV template",
            as_repo_path(get_nested(cfg, "smplx.uv_path")),
            "仅用于导出 textured OBJ；缺失时 generation 会跳过 optim_human.obj，继续保留 optim_human.ply。",
            expect_dir=False,
            required=False,
        ),
        check_path(
            "SMPL-X texture",
            as_repo_path(get_nested(cfg, "smplx.tex_path")),
            "仅用于 textured OBJ 贴图；缺失时会导出无贴图 OBJ 或跳过贴图复制。",
            expect_dir=False,
            required=False,
        ),
    ]

    alpha_cfg_path = as_repo_path(get_nested(cfg, "pose2d.args_path"))
    asset_results.extend(check_alphapose_assets(alpha_cfg_path))
    auto_results = check_auto_downloads(cfg)

    print("GenZI 特殊依赖与模型文件检查")
    print(f"run_cfg: {run_cfg_path}")
    print(f"repo_root: {REPO_ROOT}")
    print_install_guidance()
    print_section("Python 依赖", python_results)
    print_section("模型/资产文件", asset_results)
    print_section("运行时自动下载模型", auto_results)

    required_failures = [
        result
        for result in python_results + asset_results
        if result.required and result.status != "OK"
    ]
    print("\n[汇总]")
    if required_failures:
        print(f"发现 {len(required_failures)} 个必需项未准备好。")
        if args.no_fail:
            print("--no-fail 已启用，返回码保持 0。")
            return 0
        return 1

    print("必需依赖和本地模型/资产文件已准备好。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
