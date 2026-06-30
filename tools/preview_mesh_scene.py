from __future__ import annotations

import argparse
import os.path as osp
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from genzi.usd_io import USD_EXTENSIONS, UsdMeshLoadOptions, load_usd_mesh


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Preview a GenZI scene mesh from USD or a trimesh-supported mesh file."
    )
    parser.add_argument("--mesh", required=True, help="Input mesh path.")
    parser.add_argument(
        "--out",
        default="",
        help="Output preview PNG. If omitted, only prints summary unless --interactive is set.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive trimesh viewer window.",
    )
    parser.add_argument("--time-code", type=float, default=None, help="USD time code to sample.")
    parser.add_argument(
        "--include-invisible",
        action="store_true",
        help="Include invisible USD mesh prims.",
    )
    parser.add_argument(
        "--purpose",
        default="default",
        choices=("default", "render", "proxy", "all"),
        help="USD purpose filter. render/proxy include default meshes too.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=50000,
        help="Maximum number of faces to draw in the PNG preview.",
    )
    parser.add_argument(
        "--sample-by",
        default="area",
        choices=("area", "random", "stride"),
        help="Face sampling strategy for the PNG preview.",
    )
    parser.add_argument(
        "--style",
        default="wireframe",
        choices=("wireframe", "solid", "solid-wire"),
        help="PNG preview shading style.",
    )
    parser.add_argument(
        "--up-axis",
        default="y",
        choices=("x", "y", "z"),
        help="Source mesh up axis to map onto the preview's vertical axis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for random face sampling in large meshes.",
    )
    return parser.parse_args(argv)


def load_mesh(path, args):
    mesh_path = Path(path)
    if mesh_path.suffix.lower() in USD_EXTENSIONS:
        options = UsdMeshLoadOptions(
            time_code=args.time_code,
            include_invisible=args.include_invisible,
            purpose=args.purpose,
        )
        return load_usd_mesh(str(mesh_path), options=options)

    import trimesh

    loaded = trimesh.load(str(mesh_path), process=False, validate=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    return loaded


def format_float(value):
    return "{:.9g}".format(float(value))


def print_summary(mesh, path):
    bounds = mesh.bounds
    print("Mesh preview source")
    print("  path: {}".format(path))
    print("  vertices: {}".format(len(mesh.vertices)))
    print("  faces: {}".format(len(mesh.faces)))
    print("  bbox min: {}".format([format_float(value) for value in bounds[0]]))
    print("  bbox max: {}".format([format_float(value) for value in bounds[1]]))
    print("  extents : {}".format([format_float(value) for value in mesh.extents]))


def _sample_faces(faces, areas, max_faces, sample_by, seed):
    if max_faces <= 0 or len(faces) <= max_faces:
        return faces

    if sample_by == "area":
        areas = np.asarray(areas)
        if len(areas) != len(faces):
            raise RuntimeError("Face area count does not match face count.")
        face_ids = np.argpartition(areas, -max_faces)[-max_faces:]
        return faces[np.sort(face_ids)]

    if sample_by == "stride":
        face_ids = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        return faces[face_ids]

    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(faces), size=max_faces, replace=False)
    return faces[np.sort(face_ids)]


def _set_equal_axes(ax, bounds):
    center = bounds.mean(axis=0)
    radius = float(np.max(bounds[1] - bounds[0]) * 0.55)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _to_display_coords(coords, up_axis):
    coords = np.asarray(coords)
    if up_axis == "z":
        return coords
    if up_axis == "y":
        return coords[..., [0, 2, 1]]
    return coords[..., [1, 2, 0]]


def _style_args(style):
    if style == "wireframe":
        return {
            "facecolors": (0.00, 0.00, 0.00, 0.00),
            "edgecolors": (0.42, 0.86, 1.00, 0.45),
            "linewidths": 0.10,
        }
    if style == "solid-wire":
        return {
            "facecolors": (0.18, 0.58, 0.90, 0.42),
            "edgecolors": (0.88, 0.96, 1.00, 0.24),
            "linewidths": 0.06,
        }
    return {
        "facecolors": (0.30, 0.68, 0.95, 0.96),
        "edgecolors": (0.86, 0.95, 1.00, 0.10),
        "linewidths": 0.05,
    }


def save_preview(mesh, out_path, max_faces, sample_by, style, up_axis, seed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_path = Path(out_path)
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = _sample_faces(
        np.asarray(mesh.faces, dtype=np.int64),
        mesh.area_faces,
        max_faces,
        sample_by,
        seed,
    )
    triangles = _to_display_coords(vertices[faces], up_axis)
    display_bounds = _to_display_coords(mesh.bounds, up_axis)

    views = [
        ("iso", 28, -45),
        ("front", 0, -90),
        ("side", 0, 0),
        ("top", 90, -90),
    ]
    fig = plt.figure(figsize=(12, 12), dpi=150)
    fig.patch.set_facecolor((0.06, 0.07, 0.08))
    style_kwargs = _style_args(style)
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        collection = Poly3DCollection(triangles, **style_kwargs)
        ax.add_collection3d(collection)
        ax.set_facecolor((0.06, 0.07, 0.08))
        _set_equal_axes(ax, display_bounds)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, color=(0.92, 0.95, 0.98))
        ax.set_axis_off()

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)
    print("Saved preview PNG to {}".format(out_path))


def show_interactive(mesh):
    import trimesh

    scene = trimesh.Scene(mesh)
    scene.show()


def main(argv=None):
    args = parse_args(argv)
    mesh = load_mesh(args.mesh, args)
    print_summary(mesh, args.mesh)

    if args.out:
        save_preview(
            mesh,
            args.out,
            args.max_faces,
            args.sample_by,
            args.style,
            args.up_axis,
            args.seed,
        )
    if args.interactive:
        show_interactive(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
