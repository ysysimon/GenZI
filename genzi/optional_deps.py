"""Helpers for optional GenZI runtime dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType


def import_optional_dependency(
    module_name: str,
    *,
    package_name: str | None = None,
    purpose: str,
    install_hint: str,
) -> ModuleType:
    """Import an optional dependency and raise a setup-oriented error."""

    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - optional deps can fail during import.
        label = package_name or module_name
        first_line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        raise ImportError(
            "\n".join(
                [
                    f"缺少或无法加载 GenZI 可选依赖: {label}",
                    f"用途: {purpose}",
                    f"原始错误: {first_line}",
                    "准备方式:",
                    install_hint,
                    "也可以先运行: uv run python tools/check_special_deps.py --run-cfg config/sketchfab_gen.yml",
                ]
            )
        ) from exc
