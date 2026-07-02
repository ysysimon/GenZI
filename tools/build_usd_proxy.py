from __future__ import annotations

import argparse
import os.path as osp
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from genzi.usd_io import UsdMeshLoadOptions, load_usd_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight USD proxy mesh.")
    parser.add_argument("input_usd", type=Path)
    parser.add_argument("output_usd", type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.06)
    parser.add_argument("--max-cells", type=int, default=250_000)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--purpose", default="default")
    parser.add_argument("--prim-path", default=None)
    parser.add_argument("--include-invisible", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def iter_surface_cells(mesh, voxel_size: float, chunk_size: int):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    bounds_min = vertices.min(axis=0)

    key_chunks = []
    center_chunks = []
    axis_chunks = []
    sign_chunks = []

    for start in range(0, len(faces), chunk_size):
        stop = min(start + chunk_size, len(faces))
        tri = vertices[faces[start:stop]]
        edge_a = tri[:, 1] - tri[:, 0]
        edge_b = tri[:, 2] - tri[:, 0]
        normals = np.cross(edge_a, edge_b)
        norm_len = np.linalg.norm(normals, axis=1)
        valid = norm_len > 1e-10
        if not np.any(valid):
            continue

        tri = tri[valid]
        normals = normals[valid] / norm_len[valid, None]
        centers = tri.mean(axis=1)

        q = np.floor((centers - bounds_min) / voxel_size).astype(np.int64)
        axes = np.argmax(np.abs(normals), axis=1).astype(np.int64)
        signs = (normals[np.arange(len(normals)), axes] >= 0).astype(np.int64)

        # Pack quantized xyz plus dominant normal direction into one stable key.
        keys = (((q[:, 0] * 1_000_003 + q[:, 1]) * 1_000_003 + q[:, 2]) * 6) + (
            axes * 2 + signs
        )
        _, unique_idx = np.unique(keys, return_index=True)

        key_chunks.append(keys[unique_idx])
        center_chunks.append(centers[unique_idx])
        axis_chunks.append(axes[unique_idx])
        sign_chunks.append(signs[unique_idx])
        print(f"processed faces {stop}/{len(faces)}; candidate cells {sum(len(k) for k in key_chunks)}")

    keys = np.concatenate(key_chunks, axis=0)
    centers = np.concatenate(center_chunks, axis=0)
    axes = np.concatenate(axis_chunks, axis=0)
    signs = np.concatenate(sign_chunks, axis=0)
    _, unique_idx = np.unique(keys, return_index=True)
    return centers[unique_idx], axes[unique_idx], signs[unique_idx]


def limit_cells(centers, axes, signs, max_cells: int, seed: int):
    if max_cells <= 0 or len(centers) <= max_cells:
        return centers, axes, signs
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(centers), size=max_cells, replace=False))
    return centers[selected], axes[selected], signs[selected]


def cells_to_mesh(centers, axes, signs, voxel_size: float):
    half = voxel_size * 0.65
    vertices = np.empty((len(centers) * 4, 3), dtype=np.float32)
    faces = np.empty((len(centers) * 2, 3), dtype=np.int32)

    for idx, (center, axis, sign) in enumerate(zip(centers, axes, signs)):
        base = idx * 4
        quad = np.repeat(center[None, :], 4, axis=0)
        if axis == 0:
            quad[:, 1:] += np.array(
                [[-half, -half], [half, -half], [half, half], [-half, half]],
                dtype=np.float32,
            )
        elif axis == 1:
            quad[:, [0, 2]] += np.array(
                [[-half, -half], [half, -half], [half, half], [-half, half]],
                dtype=np.float32,
            )
        else:
            quad[:, :2] += np.array(
                [[-half, -half], [half, -half], [half, half], [-half, half]],
                dtype=np.float32,
            )
        vertices[base : base + 4] = quad

        if sign:
            faces[idx * 2 : idx * 2 + 2] = [[base, base + 1, base + 2], [base, base + 2, base + 3]]
        else:
            faces[idx * 2 : idx * 2 + 2] = [[base, base + 2, base + 1], [base, base + 3, base + 2]]

    return vertices, faces


def write_usd_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    from pxr import Usd, UsdGeom

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    mesh = UsdGeom.Mesh.Define(stage, "/World/ProxyMesh")
    mesh.CreatePointsAttr(vertices.astype(np.float32).tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.astype(np.int64).reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()


def main() -> None:
    args = parse_args()
    mesh = load_usd_mesh(
        str(args.input_usd),
        options=UsdMeshLoadOptions(
            include_invisible=args.include_invisible,
            purpose=args.purpose,
            prim_path=args.prim_path,
        ),
    )
    print(f"loaded mesh vertices={len(mesh.vertices)} faces={len(mesh.faces)}")

    centers, axes, signs = iter_surface_cells(mesh, args.voxel_size, args.chunk_size)
    print(f"unique surface cells={len(centers)}")
    centers, axes, signs = limit_cells(centers, axes, signs, args.max_cells, args.seed)
    print(f"selected surface cells={len(centers)}")

    vertices, faces = cells_to_mesh(centers, axes, signs, args.voxel_size)
    print(f"proxy vertices={len(vertices)} faces={len(faces)}")
    write_usd_mesh(args.output_usd, vertices, faces)
    print(f"wrote {args.output_usd}")


if __name__ == "__main__":
    main()
