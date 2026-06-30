from __future__ import annotations

import argparse
import os.path as osp
import sys
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from genzi.usd_io import UsdMeshLoadOptions, load_usd_mesh_with_metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a flattened USD scene mesh to a GenZI-supported mesh file."
    )
    parser.add_argument("--usd", required=True, help="Input .usd, .usda, or .usdc file.")
    parser.add_argument("--out", required=True, help="Output mesh path, for example scene.obj.")
    parser.add_argument("--time-code", type=float, default=None, help="USD time code to sample.")
    parser.add_argument(
        "--include-invisible",
        action="store_true",
        help="Include invisible UsdGeom.Mesh prims.",
    )
    parser.add_argument(
        "--purpose",
        default="default",
        choices=("default", "render", "proxy", "all"),
        help="USD purpose filter. render/proxy include default meshes too.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print mesh count, geometry size, bbox, and stage unit metadata.",
    )
    return parser.parse_args(argv)


def format_float(value):
    return "{:.9g}".format(float(value))


def print_summary(result, out_path):
    mesh = result.mesh
    stats = result.stats
    bounds = mesh.bounds

    print("Converted USD mesh")
    print("  usd: {}".format(stats.source_path))
    print("  out: {}".format(out_path))
    print("  mesh prims: {}".format(stats.mesh_count))
    print("  vertices: {}".format(len(mesh.vertices)))
    print("  faces: {}".format(len(mesh.faces)))
    print("  skipped invisible: {}".format(stats.skipped_invisible_count))
    print("  skipped by purpose: {}".format(stats.skipped_purpose_count))
    print("  skipped empty: {}".format(stats.skipped_empty_count))
    if stats.stage_meters_per_unit is None:
        print("  meters per unit: unknown")
    else:
        print("  meters per unit: {}".format(format_float(stats.stage_meters_per_unit)))
    print("  bbox min: {}".format([format_float(value) for value in bounds[0]]))
    print("  bbox max: {}".format([format_float(value) for value in bounds[1]]))


def main(argv=None):
    args = parse_args(argv)
    options = UsdMeshLoadOptions(
        time_code=args.time_code,
        include_invisible=args.include_invisible,
        purpose=args.purpose,
    )
    result = load_usd_mesh_with_metadata(args.usd, options=options)

    out_path = Path(args.out)
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(str(out_path))

    if args.print_summary:
        print_summary(result, out_path)
    else:
        print("Converted USD mesh to {}".format(out_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
