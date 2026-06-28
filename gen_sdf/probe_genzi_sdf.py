"""Probe GenZI dense SDF values at explicit world-space sample positions.

Example:

    python gen_sdf/probe_genzi_sdf.py --sdf gen_sdf/export/scene.json --point 0 0 0 --point 1 0 0

The input can be either:

    scene.json
    scene_sdf.npy

This script intentionally matches the current GenZI dense SDF query path:

    sdf.reshape(1, 1, dim, dim, dim)
    world_xyz -> normalized [-1, 1]
    grid_sample(..., padding_mode="border", align_corners=True)
    sample grid coordinates use xyz -> zyx, as in Scene.get_sdf()

Houdini VEX cross-check:

    // Detail wrangle or point wrangle.
    // Input 1 should be the same dense SDF volume used for export.
    vector sample_pos = chv("sample_pos");
    float sdf = volumesample(1, "sdf", sample_pos);
    printf("sample_pos=(%g, %g, %g), sdf=%g\\n", sample_pos.x, sample_pos.y, sample_pos.z, sdf);

Point wrangle version for checking the current point position:

    f@sdf = volumesample(1, "sdf", @P);

Make sure Houdini samples the same dense volume and the same coordinate space
used by the exported JSON min/max. If the Houdini volume node has transforms,
verify that the VEX sample position and exported GenZI world coordinates match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def resolve_sdf_paths(sdf_path: str) -> tuple[Path, Path]:
    path = Path(sdf_path)
    path_str = str(path)

    if path_str.endswith(".json"):
        return path, Path(path_str[:-5] + "_sdf.npy")

    if path_str.endswith("_sdf.npy"):
        return Path(path_str[:-8] + ".json"), path

    raise RuntimeError("SDF path must end with .json or _sdf.npy: {}".format(sdf_path))


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("This probe tool requires numpy in the active GenZI Python environment.") from exc
    return np


def require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("This probe tool requires torch in the active GenZI Python environment.") from exc
    return torch, F


def load_sdf(sdf_path: str) -> dict[str, Any]:
    np = require_numpy()
    json_path, npy_path = resolve_sdf_paths(sdf_path)

    if not json_path.exists():
        raise FileNotFoundError("Missing SDF metadata JSON: {}".format(json_path))
    if not npy_path.exists():
        raise FileNotFoundError("Missing SDF NPY array: {}".format(npy_path))

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dim = int(meta["dim"])
    sdf_min = np.asarray(meta["min"], dtype=np.float32)
    sdf_max = np.asarray(meta["max"], dtype=np.float32)
    sdf = np.load(npy_path).astype(np.float32)

    if sdf_min.shape != (3,) or sdf_max.shape != (3,):
        raise RuntimeError("SDF metadata min/max must each contain 3 values.")

    expected_count = dim * dim * dim
    if sdf.size != expected_count:
        raise RuntimeError(
            "SDF voxel count mismatch: got {}, expected {} for dim {}.".format(
                sdf.size, expected_count, dim
            )
        )

    return {
        "json_path": json_path,
        "npy_path": npy_path,
        "dim": dim,
        "min": sdf_min,
        "max": sdf_max,
        "sdf": sdf.reshape(1, 1, dim, dim, dim),
    }


def probe_sdf(sdf_data: dict[str, Any], points: list[list[float]], device: str) -> Any:
    torch, F = require_torch()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")

    torch_device = torch.device(device)
    point_tensor = torch.as_tensor(points, dtype=torch.float32, device=torch_device)
    sdf_min = torch.as_tensor(sdf_data["min"], dtype=torch.float32, device=torch_device).reshape(1, 1, 3)
    sdf_max = torch.as_tensor(sdf_data["max"], dtype=torch.float32, device=torch_device).reshape(1, 1, 3)
    sdf_vol = torch.as_tensor(sdf_data["sdf"], dtype=torch.float32, device=torch_device)

    vertices = point_tensor.reshape(1, -1, 3)
    vertices = (vertices - sdf_min) / (sdf_max - sdf_min) * 2 - 1
    grid = vertices[..., [2, 1, 0]].view(1, len(points), 1, 1, 3)

    with torch.no_grad():
        sdf_values = F.grid_sample(
            sdf_vol,
            grid,
            padding_mode="border",
            align_corners=True,
        )

    return sdf_values.reshape(len(points)).detach().cpu().numpy()


def point_in_bounds(point: list[float], sdf_min: Any, sdf_max: Any) -> bool:
    np = require_numpy()
    p = np.asarray(point, dtype=np.float32)
    return bool(np.all(p >= sdf_min) and np.all(p <= sdf_max))


def format_float(value: float) -> str:
    return "{:.9g}".format(float(value))


def print_table(sdf_data: dict[str, Any], points: list[list[float]], values: Any, device: str) -> None:
    print("sdf_json: {}".format(sdf_data["json_path"]))
    print("sdf_npy : {}".format(sdf_data["npy_path"]))
    print("dim     : {}".format(sdf_data["dim"]))
    print("min     : {}".format([format_float(v) for v in sdf_data["min"]]))
    print("max     : {}".format([format_float(v) for v in sdf_data["max"]]))
    print("device  : {}".format(device))
    print("")
    print("{:>5} {:>14} {:>14} {:>14} {:>14} {:>9}".format("index", "x", "y", "z", "sdf", "in_bbox"))
    print("-" * 79)

    for idx, (point, value) in enumerate(zip(points, values)):
        in_bbox = point_in_bounds(point, sdf_data["min"], sdf_data["max"])
        print(
            "{:>5} {:>14} {:>14} {:>14} {:>14} {:>9}".format(
                idx,
                format_float(point[0]),
                format_float(point[1]),
                format_float(point[2]),
                format_float(value),
                str(in_bbox).lower(),
            )
        )


def print_json(sdf_data: dict[str, Any], points: list[list[float]], values: Any, device: str) -> None:
    payload = {
        "sdf_json": str(sdf_data["json_path"]),
        "sdf_npy": str(sdf_data["npy_path"]),
        "dim": int(sdf_data["dim"]),
        "min": [float(v) for v in sdf_data["min"]],
        "max": [float(v) for v in sdf_data["max"]],
        "device": device,
        "samples": [
            {
                "index": idx,
                "point": [float(v) for v in point],
                "sdf": float(value),
                "in_bbox": point_in_bounds(point, sdf_data["min"], sdf_data["max"]),
            }
            for idx, (point, value) in enumerate(zip(points, values))
        ],
    }
    json.dump(payload, sys.stdout, indent=2)
    print("")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sdf", required=True, help="Path to GenZI SDF .json or _sdf.npy.")
    parser.add_argument(
        "--point",
        nargs=3,
        type=float,
        action="append",
        required=True,
        metavar=("X", "Y", "Z"),
        help="World-space sample point. Can be repeated.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Torch device for sampling.")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    points = [[float(v) for v in point] for point in args.point]
    sdf_data = load_sdf(args.sdf)
    values = probe_sdf(sdf_data, points, args.device)

    if args.json_output:
        print_json(sdf_data, points, values, args.device)
    else:
        print_table(sdf_data, points, values, args.device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
