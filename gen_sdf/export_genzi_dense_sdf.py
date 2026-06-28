"""Shared GenZI dense SDF export logic for Houdini and hython entrypoints.

This module intentionally has no button-callback or command-line entrypoint.
Use one of the thin wrappers instead:

    export_genzi_dense_sdf_callback.py
    export_genzi_dense_sdf_hython.py

The exported files are:

    scene.json
    scene_sdf.npy

The JSON schema matches the current GenZI loader:

    {"dim": N, "min": [x, y, z], "max": [x, y, z]}

The NPY layout is sdf[ix, iy, iz], where ix/iy/iz correspond to world XYZ.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import hou
except ImportError as exc:  # pragma: no cover - only happens outside Houdini.
    hou = None
    _HOU_IMPORT_ERROR = exc
else:
    _HOU_IMPORT_ERROR = None


@dataclass
class ExportConfig:
    out_json: str
    volume_name: str = "sdf"
    input_index: int = -1
    flip_sign: bool = False
    unit_scale: float = 1.0
    layout: str = "auto"
    overwrite: bool = True


def _require_hou() -> Any:
    if hou is None:
        raise RuntimeError(
            "This exporter must run inside Houdini or hython. "
            "Could not import hou: {}".format(_HOU_IMPORT_ERROR)
        )
    return hou


def _eval_parm(node: Any, name: str, default: Any) -> Any:
    parm = node.parm(name)
    if parm is None:
        return default

    if isinstance(default, bool):
        return bool(parm.eval())
    if isinstance(default, int):
        return int(parm.eval())
    if isinstance(default, float):
        return float(parm.eval())
    if isinstance(default, str):
        return parm.evalAsString()

    return parm.eval()


def config_from_node(node: Any, prefix: str = "genzi_") -> ExportConfig:
    """Read optional exporter parameters from a Houdini node.

    Suggested parameters:

        genzi_out_json      string/file path
        genzi_volume_name   string, default "sdf"
        genzi_input_index   int, default -1
        genzi_flip_sign     toggle, default off
        genzi_unit_scale    float, default 1.0
        genzi_layout        menu token: auto / x_fastest / raw_xyz
        genzi_overwrite     toggle, default on

    Layout tokens:

        auto
            Try x_fastest and raw_xyz, then compare sampled values against
            Houdini's volume.voxel((ix, iy, iz)) API.
        x_fastest
            Treat Houdini's flat voxel buffer as z/y/x storage with x as the
            fastest-changing axis, then export sdf[ix, iy, iz].
        raw_xyz
            Reshape Houdini's flat voxel buffer directly as sdf[ix, iy, iz].
    """

    h = _require_hou()

    out_json = _eval_parm(node, prefix + "out_json", "$HIP/export/scene.json")
    out_json = h.expandString(str(out_json))

    return ExportConfig(
        out_json=out_json,
        volume_name=_eval_parm(node, prefix + "volume_name", "sdf"),
        input_index=_eval_parm(node, prefix + "input_index", -1),
        flip_sign=_eval_parm(node, prefix + "flip_sign", False),
        unit_scale=_eval_parm(node, prefix + "unit_scale", 1.0),
        layout=_eval_parm(node, prefix + "layout", "auto"),
        overwrite=_eval_parm(node, prefix + "overwrite", True),
    )


def _prim_name(prim: Any) -> str:
    attrib = prim.geometry().findPrimAttrib("name")
    if attrib is None:
        return ""
    try:
        return str(prim.attribValue(attrib))
    except Exception:
        return ""


def _get_geometry(node: Any, input_index: int) -> Any:
    if input_index >= 0:
        inputs = node.inputs()
        if input_index >= len(inputs) or inputs[input_index] is None:
            raise RuntimeError("Input {} does not exist on node {}.".format(input_index, node.path()))
        return inputs[input_index].geometry()

    return node.geometry()


def _find_sdf_volume(geo: Any, volume_name: str) -> Any:
    h = _require_hou()

    volumes = []
    vdb_names = []

    for prim in geo.prims():
        name = _prim_name(prim)

        if isinstance(prim, h.Volume):
            volumes.append((name, prim))
            if name == volume_name:
                return prim

        if hasattr(h, "VDB") and isinstance(prim, h.VDB):
            vdb_names.append(name)

    if len(volumes) == 1:
        name, prim = volumes[0]
        print("No volume named {!r}; using the only Houdini Volume: {!r}".format(volume_name, name))
        return prim

    if vdb_names and not volumes:
        raise RuntimeError(
            "Found VDB primitives {}, but this exporter needs a dense Houdini Volume. "
            "Convert or resample the VDB to a dense volume before exporting GenZI SDF.".format(vdb_names)
        )

    raise RuntimeError(
        "Could not find dense Houdini Volume named {!r}. Available dense volumes: {}".format(
            volume_name, [name for name, _ in volumes]
        )
    )


def _read_flat_voxels(volume_prim: Any, expected_count: int) -> np.ndarray:
    try:
        raw = volume_prim.allVoxelsAsString()
        if isinstance(raw, str):
            raw = raw.encode("latin1")
        flat = np.frombuffer(raw, dtype=np.float32)
        if flat.size == expected_count:
            return flat.copy()
    except Exception as exc:
        print("allVoxelsAsString failed, falling back to allVoxels(): {}".format(exc))

    flat = np.asarray(volume_prim.allVoxels(), dtype=np.float32)
    if flat.size != expected_count:
        raise RuntimeError("Voxel count mismatch: got {}, expected {}".format(flat.size, expected_count))
    return flat


def _voxel_value(volume_prim: Any, ix: int, iy: int, iz: int) -> float:
    return float(volume_prim.voxel((int(ix), int(iy), int(iz))))


def _build_layout(flat: np.ndarray, nx: int, ny: int, nz: int, layout: str) -> np.ndarray:
    if layout == "x_fastest":
        return flat.reshape((nz, ny, nx)).transpose(2, 1, 0)
    if layout == "raw_xyz":
        return flat.reshape((nx, ny, nz))
    raise RuntimeError("Unknown voxel layout {!r}. Use auto, x_fastest, or raw_xyz.".format(layout))


def _choose_layout(volume_prim: Any, flat: np.ndarray, nx: int, ny: int, nz: int, layout: str) -> np.ndarray:
    if layout != "auto":
        sdf = _build_layout(flat, nx, ny, nz, layout)
        print("Using explicit voxel layout: {}".format(layout))
        return np.ascontiguousarray(sdf.astype(np.float32))

    sample_indices = [
        (0, 0, 0),
        (nx - 1, 0, 0),
        (0, ny - 1, 0),
        (0, 0, nz - 1),
        (nx // 2, ny // 2, nz // 2),
        (min(nx - 1, nx // 3), min(ny - 1, ny // 5), min(nz - 1, nz // 7)),
    ]

    best_name = None
    best_sdf = None
    best_error = None

    for name in ("x_fastest", "raw_xyz"):
        sdf = _build_layout(flat, nx, ny, nz, name)
        error = 0.0

        for ix, iy, iz in sample_indices:
            expected = _voxel_value(volume_prim, ix, iy, iz)
            actual = float(sdf[ix, iy, iz])
            error += abs(expected - actual)

        if best_error is None or error < best_error:
            best_name = name
            best_sdf = sdf
            best_error = error

    if best_sdf is None or best_error is None or best_error > 1e-4:
        raise RuntimeError("Could not infer Houdini voxel layout safely. Best error: {}".format(best_error))

    print("Using inferred voxel layout: {} (error {:.8f})".format(best_name, best_error))
    return np.ascontiguousarray(best_sdf.astype(np.float32))


def _orient_array_to_world_xyz(sdf: np.ndarray, volume_prim: Any, nx: int, ny: int, nz: int) -> np.ndarray:
    p000 = np.asarray(volume_prim.indexToPos((0, 0, 0)), dtype=np.float64)
    axis_names = ("x", "y", "z")
    dims = (nx, ny, nz)

    for expected_axis, offset in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        if dims[expected_axis] <= 1:
            continue

        p1 = np.asarray(volume_prim.indexToPos(offset), dtype=np.float64)
        vec = p1 - p000
        dominant_axis = int(np.argmax(np.abs(vec)))

        if dominant_axis != expected_axis:
            raise RuntimeError(
                "Volume index axis {} maps to world axis {}, but GenZI export expects XYZ-aligned volumes.".format(
                    axis_names[expected_axis], axis_names[dominant_axis]
                )
            )

        off_axis = np.linalg.norm(np.delete(vec, expected_axis))
        main_axis = abs(vec[expected_axis])
        if main_axis > 0 and off_axis / main_axis > 1e-4:
            raise RuntimeError(
                "Volume axis {} is not axis-aligned enough for GenZI dense SDF export.".format(
                    axis_names[expected_axis]
                )
            )

        if vec[expected_axis] < 0:
            sdf = np.flip(sdf, axis=expected_axis)
            print("Flipped array axis {} to match world min-to-max order.".format(axis_names[expected_axis]))

    return np.ascontiguousarray(sdf.astype(np.float32))


def _validate_config(config: ExportConfig) -> tuple[str, str]:
    if config.unit_scale <= 0:
        raise RuntimeError("unit_scale must be positive.")

    layout = str(config.layout)
    if layout not in ("auto", "x_fastest", "raw_xyz"):
        raise RuntimeError("layout must be auto, x_fastest, or raw_xyz. Got {!r}.".format(layout))

    out_json = os.path.normpath(config.out_json)
    out_root, ext = os.path.splitext(out_json)
    if ext.lower() != ".json":
        raise RuntimeError("out_json must end with .json. Got: {}".format(out_json))

    out_npy = out_root + "_sdf.npy"

    if not config.overwrite:
        existing = [path for path in (out_json, out_npy) if os.path.exists(path)]
        if existing:
            raise RuntimeError("Output exists and overwrite is disabled: {}".format(existing))

    return out_json, out_npy


def export_from_geometry(geo: Any, config: ExportConfig) -> dict[str, Any]:
    out_json, out_npy = _validate_config(config)

    sdf_prim = _find_sdf_volume(geo, config.volume_name)
    nx, ny, nz = sdf_prim.resolution()
    nx, ny, nz = int(nx), int(ny), int(nz)

    print("SDF resolution: {} x {} x {}".format(nx, ny, nz))

    if not (nx == ny == nz):
        raise RuntimeError(
            "Current GenZI loader expects cubic dense SDF, but got {} x {} x {}. "
            "Resample to N x N x N before exporting.".format(nx, ny, nz)
        )

    expected_count = int(nx * ny * nz)
    flat = _read_flat_voxels(sdf_prim, expected_count)

    sdf = _choose_layout(sdf_prim, flat, nx, ny, nz, str(config.layout))
    sdf = _orient_array_to_world_xyz(sdf, sdf_prim, nx, ny, nz)

    if config.flip_sign:
        sdf = -sdf
        print("Flipped SDF sign.")

    sdf = np.ascontiguousarray(sdf.astype(np.float32) * float(config.unit_scale))

    p0 = np.asarray(sdf_prim.indexToPos((0, 0, 0)), dtype=np.float32)
    p1 = np.asarray(sdf_prim.indexToPos((nx - 1, ny - 1, nz - 1)), dtype=np.float32)
    sdf_min = np.minimum(p0, p1).astype(np.float32) * float(config.unit_scale)
    sdf_max = np.maximum(p0, p1).astype(np.float32) * float(config.unit_scale)

    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    meta = {
        "dim": int(nx),
        "min": sdf_min.tolist(),
        "max": sdf_max.tolist(),
    }

    np.save(out_npy, sdf)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    summary = {
        "json": out_json,
        "npy": out_npy,
        "dim": int(nx),
        "min": meta["min"],
        "max": meta["max"],
        "value_min": float(np.min(sdf)),
        "value_max": float(np.max(sdf)),
        "negative_voxels": int(np.sum(sdf < 0.0)),
        "positive_voxels": int(np.sum(sdf > 0.0)),
    }

    print_summary(summary)
    return summary


def export_from_node(node: Any, config: ExportConfig) -> dict[str, Any]:
    geo = _get_geometry(node, config.input_index)
    return export_from_geometry(geo, config)


def print_summary(summary: dict[str, Any]) -> None:
    print("Exported GenZI SDF")
    print("  json: {}".format(summary["json"]))
    print("  npy : {}".format(summary["npy"]))
    print("  dim : {}".format(summary["dim"]))
    print("  min : {}".format(summary["min"]))
    print("  max : {}".format(summary["max"]))
    print("  value range: [{:.6f}, {:.6f}]".format(summary["value_min"], summary["value_max"]))
    print("  negative voxels: {}".format(summary["negative_voxels"]))
    print("  positive voxels: {}".format(summary["positive_voxels"]))
