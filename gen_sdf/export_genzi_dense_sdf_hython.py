"""Headless hython entrypoint for GenZI dense SDF export.

Example:

    set GENZI_ROOT=C:/path/to/GenZI
    hython %GENZI_ROOT%/gen_sdf/export_genzi_dense_sdf_hython.py ^
        --hip %GENZI_ROOT%/gen_sdf/vdb_sdf.hip ^
        --node /obj/geo1/OUT_SDF ^
        --out-json %GENZI_ROOT%/data/scenes/my_scene.json

This script has no UI dependency. It prints to stdout and raises exceptions
on failure.
"""

from __future__ import annotations

import argparse
import sys

from export_genzi_dense_sdf import ExportConfig, _require_hou, export_from_node


def _load_hip(path: str) -> None:
    hou = _require_hou()
    try:
        hou.hipFile.load(path, suppress_save_prompt=True, ignore_load_warnings=True)
    except TypeError:
        hou.hipFile.load(path, suppress_save_prompt=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hip", default=None, help="Optional .hip file to load before export.")
    parser.add_argument("--node", required=True, help="Node path to export from.")
    parser.add_argument("--out-json", required=True, help="Output GenZI .json path.")
    parser.add_argument("--volume-name", default="sdf", help="Dense Houdini Volume primitive name.")
    parser.add_argument(
        "--input-index",
        type=int,
        default=-1,
        help="Use this input geometry instead of the node's own cooked geometry. -1 means node geometry.",
    )
    parser.add_argument("--flip-sign", action="store_true", help="Multiply SDF values by -1 before export.")
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=1.0,
        help="Scale both SDF values and world min/max. Use 0.01 for cm-to-m.",
    )
    parser.add_argument(
        "--layout",
        default="auto",
        choices=("auto", "x_fastest", "raw_xyz"),
        help="Voxel buffer layout. Keep auto unless validation fails.",
    )
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if output files already exist.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    hou = _require_hou()
    args = _parse_args(argv)

    if args.hip:
        _load_hip(args.hip)

    node = hou.node(args.node)
    if node is None:
        raise RuntimeError("Could not find node: {}".format(args.node))

    config = ExportConfig(
        out_json=hou.expandString(args.out_json),
        volume_name=args.volume_name,
        input_index=args.input_index,
        flip_sign=args.flip_sign,
        unit_scale=args.unit_scale,
        layout=args.layout,
        overwrite=not args.no_overwrite,
    )

    export_from_node(node, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
