"""Houdini button callback entrypoint for GenZI dense SDF export.

Use this one-line Python callback on the Houdini button:

    import os, hou; p = hou.expandString(kwargs["node"].parm("genzi_callback_script").evalAsString()); p = os.path.join(p, "export_genzi_dense_sdf_callback.py") if os.path.isdir(p) else p; exec(compile(open(p, encoding="utf-8").read(), p, "exec"), {"kwargs": kwargs, "__file__": p, "__name__": "__houdini_callback__"})

The callback reads exporter parameters from the button node:

    genzi_callback_script
        Type: String/File path
        Path to this callback script, for example
        "$HIP/export_genzi_dense_sdf_callback.py". It may also be the script
        directory, for example "$HIP"; in that case the one-line callback
        appends "export_genzi_dense_sdf_callback.py" automatically.

    genzi_out_json
        Type: String/File path
        Output GenZI metadata path, for example "$HIP/export/scene.json".
        The paired SDF array is written next to it as "scene_sdf.npy".

    genzi_volume_name
        Type: String
        Dense Houdini Volume primitive name to export. Default is "sdf".

    genzi_input_index
        Type: Integer
        Geometry source. Use -1 for the current node's cooked geometry, or
        0/1/2... to export from a connected input node.

    genzi_flip_sign
        Type: Toggle/Boolean
        Multiply SDF values by -1 before export. Enable this if the generated
        field has inside positive and outside negative.

    genzi_unit_scale
        Type: Float
        Scale both SDF values and JSON min/max. Use 1.0 for meters, or 0.01
        when converting centimeter Houdini units to GenZI meters.

    genzi_layout
        Type: Menu/String
        Voxel memory layout. Supported values are "auto", "x_fastest", and
        "raw_xyz". Keep "auto" unless layout validation fails.
        "auto" tries both supported layouts and compares several sampled voxel
        values against Houdini's volume.voxel((ix, iy, iz)) API.
        "x_fastest" treats Houdini's flat voxel buffer as z/y/x storage with x
        as the fastest-changing axis, then exports sdf[ix, iy, iz].
        "raw_xyz" reshapes the flat voxel buffer directly as sdf[ix, iy, iz].

    genzi_overwrite
        Type: Toggle/Boolean
        Whether existing .json and _sdf.npy outputs may be overwritten.

This entrypoint has no UI dependency. It prints to Houdini's Python output
and raises exceptions on failure.
"""

from __future__ import annotations

import os
import sys
from typing import Any

try:
    import hou
except ImportError as exc:  # pragma: no cover - only happens outside Houdini.
    hou = None
    _HOU_IMPORT_ERROR = exc
else:
    _HOU_IMPORT_ERROR = None


def _require_hou() -> Any:
    if hou is None:
        raise RuntimeError(
            "This callback must run inside Houdini or hython. "
            "Could not import hou: {}".format(_HOU_IMPORT_ERROR)
        )
    return hou


def _eval_string_parm(node: Any, name: str, default: str = "") -> str:
    parm = node.parm(name)
    if parm is None:
        return default
    value = parm.evalAsString()
    return value if value else default


def _callback_script_path(node: Any) -> str:
    h = _require_hou()
    path = _eval_string_parm(node, "genzi_callback_script", "")
    if path:
        path = os.path.normpath(h.expandString(path))
        if os.path.isdir(path):
            return os.path.join(path, "export_genzi_dense_sdf_callback.py")
        return path

    module_file = globals().get("__file__")
    if module_file:
        return os.path.normpath(os.path.abspath(str(module_file)))

    return ""


def _script_dir(node: Any) -> str:
    h = _require_hou()
    callback_script = _callback_script_path(node)
    if callback_script:
        return os.path.dirname(callback_script)

    return os.path.normpath(h.expandString("$HIP"))


def _ensure_script_dir(callback_kwargs: dict[str, Any]) -> str:
    if "node" not in callback_kwargs:
        raise RuntimeError("Button callback kwargs does not contain 'node'.")

    script_dir = _script_dir(callback_kwargs["node"])
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    return script_dir


def run(kwargs: dict[str, Any], prefix: str = "genzi_") -> dict[str, Any]:
    _ensure_script_dir(kwargs)
    from export_genzi_dense_sdf import config_from_node, export_from_node

    node = kwargs["node"]
    config = config_from_node(node, prefix=prefix)
    return export_from_node(node, config)


if "kwargs" in globals() and globals().get("__name__") != "export_genzi_dense_sdf_callback":
    run(globals()["kwargs"])
