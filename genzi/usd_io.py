from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


USD_EXTENSIONS = {".usd", ".usda", ".usdc"}
PURPOSES = {"default", "render", "proxy", "all"}


@dataclass(frozen=True)
class UsdMeshLoadOptions:
    time_code: Optional[float] = None
    include_invisible: bool = False
    purpose: str = "default"
    prim_path: Optional[str] = None


@dataclass(frozen=True)
class UsdMeshLoadStats:
    source_path: str
    mesh_count: int
    skipped_invisible_count: int
    skipped_purpose_count: int
    skipped_empty_count: int
    stage_meters_per_unit: Optional[float]


@dataclass(frozen=True)
class UsdMeshLoadResult:
    mesh: Any
    stats: UsdMeshLoadStats


def load_usd_mesh(path: str, options: Optional[UsdMeshLoadOptions] = None) -> Any:
    return load_usd_mesh_with_metadata(path, options=options).mesh


def load_usd_mesh_with_metadata(
    path: str, options: Optional[UsdMeshLoadOptions] = None
) -> UsdMeshLoadResult:
    Usd, UsdGeom, Gf = _require_usd()
    trimesh = _require_trimesh()

    options = options or UsdMeshLoadOptions()
    _validate_options(options)

    usd_path = Path(path)
    if usd_path.suffix.lower() not in USD_EXTENSIONS:
        raise RuntimeError("USD mesh path must end with .usd, .usda, or .usdc: {}".format(path))

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError("Could not open USD stage: {}".format(path))

    time_code = (
        Usd.TimeCode.Default()
        if options.time_code is None
        else Usd.TimeCode(float(options.time_code))
    )
    xform_cache = UsdGeom.XformCache(time_code)

    all_vertices: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    vertex_offset = 0
    mesh_count = 0
    skipped_invisible_count = 0
    skipped_purpose_count = 0
    skipped_empty_count = 0

    for prim in _iter_prims(stage, options, Usd):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        imageable = UsdGeom.Imageable(prim)
        if not options.include_invisible and _is_invisible(imageable, time_code, UsdGeom):
            skipped_invisible_count += 1
            continue

        prim_purpose = _compute_purpose(imageable, time_code)
        if not _purpose_is_included(prim_purpose, options.purpose):
            skipped_purpose_count += 1
            continue

        usd_mesh = UsdGeom.Mesh(prim)
        vertices, faces = _read_mesh_prim(usd_mesh, xform_cache, time_code, Gf)
        if len(vertices) == 0 or len(faces) == 0:
            skipped_empty_count += 1
            continue

        all_vertices.append(vertices)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(vertices)
        mesh_count += 1

    if not all_vertices or not all_faces:
        scope = (
            " under prim path {}".format(options.prim_path)
            if _has_prim_path(options)
            else ""
        )
        raise RuntimeError(
            "No usable UsdGeom.Mesh prims found in USD stage{}: {}".format(
                scope, path
            )
        )

    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)

    stats = UsdMeshLoadStats(
        source_path=str(usd_path),
        mesh_count=mesh_count,
        skipped_invisible_count=skipped_invisible_count,
        skipped_purpose_count=skipped_purpose_count,
        skipped_empty_count=skipped_empty_count,
        stage_meters_per_unit=_get_stage_meters_per_unit(stage, UsdGeom),
    )
    return UsdMeshLoadResult(mesh=mesh, stats=stats)


def _require_usd() -> Tuple[Any, Any, Any]:
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError(
            "USD mesh conversion requires the pxr module. Install usd-core==26.3 "
            "in the active GenZI environment."
        ) from exc
    return Usd, UsdGeom, Gf


def _require_trimesh() -> Any:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("USD mesh conversion requires trimesh.") from exc
    return trimesh


def _validate_options(options: UsdMeshLoadOptions) -> None:
    if options.purpose not in PURPOSES:
        raise RuntimeError(
            "USD purpose must be one of {}. Got: {}".format(
                sorted(PURPOSES), options.purpose
            )
        )
    if _has_prim_path(options) and not str(options.prim_path).startswith("/"):
        raise RuntimeError(
            "USD prim path must be an absolute prim path starting with '/'. Got: {}".format(
                options.prim_path
            )
        )


def _has_prim_path(options: UsdMeshLoadOptions) -> bool:
    return options.prim_path is not None and str(options.prim_path).strip() != ""


def _iter_prims(stage: Any, options: UsdMeshLoadOptions, Usd: Any) -> Any:
    if not _has_prim_path(options):
        return stage.Traverse()

    prim_path = str(options.prim_path).strip()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise RuntimeError("USD prim path not found in stage: {}".format(prim_path))
    return Usd.PrimRange(root_prim)


def _is_invisible(imageable: Any, time_code: Any, UsdGeom: Any) -> bool:
    visibility = imageable.ComputeVisibility(time_code)
    return visibility == UsdGeom.Tokens.invisible or str(visibility) == "invisible"


def _compute_purpose(imageable: Any, time_code: Any) -> str:
    if hasattr(imageable, "ComputePurpose"):
        purpose = imageable.ComputePurpose()
        if purpose:
            return str(purpose)

    purpose_attr = imageable.GetPurposeAttr()
    if purpose_attr:
        purpose = purpose_attr.Get(time_code)
        if purpose:
            return str(purpose)

    return "default"


def _purpose_is_included(prim_purpose: str, requested_purpose: str) -> bool:
    if requested_purpose == "all":
        return True
    if requested_purpose == "default":
        return prim_purpose == "default"
    return prim_purpose in {"default", requested_purpose}


def _read_mesh_prim(
    usd_mesh: Any, xform_cache: Any, time_code: Any, Gf: Any
) -> Tuple[np.ndarray, np.ndarray]:
    prim = usd_mesh.GetPrim()
    points = usd_mesh.GetPointsAttr().Get(time_code)
    face_vertex_counts = usd_mesh.GetFaceVertexCountsAttr().Get(time_code)
    face_vertex_indices = usd_mesh.GetFaceVertexIndicesAttr().Get(time_code)

    if points is None or face_vertex_counts is None or face_vertex_indices is None:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    vertices = _transform_points(points, xform_cache.GetLocalToWorldTransform(prim), Gf)
    faces = _triangulate_faces(face_vertex_counts, face_vertex_indices, len(vertices))
    return vertices, faces


def _transform_points(points: Sequence[Any], local_to_world: Any, Gf: Any) -> np.ndarray:
    vertices = []
    for point in points:
        world_point = local_to_world.Transform(
            Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
        )
        vertices.append((float(world_point[0]), float(world_point[1]), float(world_point[2])))
    return np.asarray(vertices, dtype=np.float64).reshape((-1, 3))


def _triangulate_faces(
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    num_vertices: int,
) -> np.ndarray:
    counts = [int(count) for count in face_vertex_counts]
    indices = [int(index) for index in face_vertex_indices]

    if sum(counts) != len(indices):
        raise RuntimeError(
            "USD mesh faceVertexCounts/faceVertexIndices mismatch: {} vs {}".format(
                sum(counts), len(indices)
            )
        )

    faces = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count

        if count < 3:
            continue
        if min(face) < 0 or max(face) >= num_vertices:
            raise RuntimeError("USD mesh face index is outside points array bounds.")

        if count == 3:
            faces.append(face)
        elif count == 4:
            faces.append([face[0], face[1], face[2]])
            faces.append([face[0], face[2], face[3]])
        else:
            for idx in range(1, count - 1):
                faces.append([face[0], face[idx], face[idx + 1]])

    return np.asarray(faces, dtype=np.int64).reshape((-1, 3))


def _get_stage_meters_per_unit(stage: Any, UsdGeom: Any) -> Optional[float]:
    if not hasattr(UsdGeom, "GetStageMetersPerUnit"):
        return None
    try:
        return float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        return None
