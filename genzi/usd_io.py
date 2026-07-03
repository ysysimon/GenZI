from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


USD_EXTENSIONS = {".usd", ".usda", ".usdc"}
PURPOSES = {"default", "render", "proxy", "all"}
TEXTURE_MODES = {"none", "diffuse"}


@dataclass(frozen=True)
class UsdMeshLoadOptions:
    time_code: Optional[float] = None
    include_invisible: bool = False
    purpose: str = "default"
    prim_path: Optional[str] = None
    preserve_materials: bool = False
    texture_mode: str = "diffuse"
    rotation_degrees: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class UsdMeshLoadStats:
    source_path: str
    mesh_count: int
    skipped_invisible_count: int
    skipped_purpose_count: int
    skipped_empty_count: int
    stage_meters_per_unit: Optional[float]
    material_count: int = 0
    textured_material_count: int = 0


@dataclass(frozen=True)
class UsdMeshLoadResult:
    mesh: Any
    stats: UsdMeshLoadStats


@dataclass(frozen=True)
class _UsdMeshPrimData:
    vertices: np.ndarray
    faces: np.ndarray
    face_corner_indices: np.ndarray
    source_face_indices: np.ndarray


def load_usd_mesh(path: str, options: Optional[UsdMeshLoadOptions] = None) -> Any:
    return load_usd_mesh_with_metadata(path, options=options).mesh


def load_usd_mesh_with_metadata(
    path: str, options: Optional[UsdMeshLoadOptions] = None
) -> UsdMeshLoadResult:
    Usd, UsdGeom, Gf, UsdShade = _require_usd()
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
    extra_transform = _build_extra_transform(options.rotation_degrees)

    if options.preserve_materials:
        return _load_usd_scene_with_materials(
            usd_path=usd_path,
            stage=stage,
            options=options,
            time_code=time_code,
            xform_cache=xform_cache,
            extra_transform=extra_transform,
            Usd=Usd,
            UsdGeom=UsdGeom,
            Gf=Gf,
            UsdShade=UsdShade,
            trimesh=trimesh,
        )

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
        mesh_data = _read_mesh_prim_data(
            usd_mesh, xform_cache, time_code, Gf, extra_transform=extra_transform
        )
        vertices, faces = mesh_data.vertices, mesh_data.faces
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


def _load_usd_scene_with_materials(
    usd_path: Path,
    stage: Any,
    options: UsdMeshLoadOptions,
    time_code: Any,
    xform_cache: Any,
    extra_transform: Optional[np.ndarray],
    Usd: Any,
    UsdGeom: Any,
    Gf: Any,
    UsdShade: Any,
    trimesh: Any,
) -> UsdMeshLoadResult:
    scene = trimesh.Scene()
    material_cache: Dict[str, Any] = {}
    mesh_count = 0
    skipped_invisible_count = 0
    skipped_purpose_count = 0
    skipped_empty_count = 0
    material_count = 0
    textured_material_count = 0

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
        mesh_data = _read_mesh_prim_data(
            usd_mesh, xform_cache, time_code, Gf, extra_transform=extra_transform
        )
        if len(mesh_data.vertices) == 0 or len(mesh_data.faces) == 0:
            skipped_empty_count += 1
            continue

        uv_values, uv_interpolation = _read_uv_primvar(usd_mesh, time_code, UsdGeom)
        default_color = _read_display_color(usd_mesh, time_code)
        groups = _get_material_face_groups(
            prim=prim,
            num_source_faces=int(np.max(mesh_data.source_face_indices)) + 1,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
        )
        mesh_count += 1

        for group_idx, (source_face_ids, material) in enumerate(groups):
            tri_mask = np.isin(mesh_data.source_face_indices, source_face_ids)
            if not np.any(tri_mask):
                continue

            face_ids = np.nonzero(tri_mask)[0]
            submesh = _build_textured_submesh(
                vertices=mesh_data.vertices,
                faces=mesh_data.faces[face_ids],
                face_corner_indices=mesh_data.face_corner_indices[face_ids],
                uv_values=uv_values,
                uv_interpolation=uv_interpolation,
                material=material,
                default_color=default_color,
                usd_path=usd_path,
                options=options,
                time_code=time_code,
                UsdShade=UsdShade,
                trimesh=trimesh,
                material_cache=material_cache,
            )
            if submesh is None:
                continue

            if material:
                material_count += 1
                if getattr(submesh.visual, "kind", None) == "texture":
                    textured_material_count += 1
            geom_name = "{}_{}".format(_safe_name(str(prim.GetPath())), group_idx)
            scene.add_geometry(submesh, geom_name=geom_name)

    if not scene.geometry:
        scope = (
            " under prim path {}".format(options.prim_path)
            if _has_prim_path(options)
            else ""
        )
        raise RuntimeError(
            "No usable UsdGeom.Mesh prims found in USD stage{}: {}".format(
                scope, usd_path
            )
        )

    stats = UsdMeshLoadStats(
        source_path=str(usd_path),
        mesh_count=mesh_count,
        skipped_invisible_count=skipped_invisible_count,
        skipped_purpose_count=skipped_purpose_count,
        skipped_empty_count=skipped_empty_count,
        stage_meters_per_unit=_get_stage_meters_per_unit(stage, UsdGeom),
        material_count=material_count,
        textured_material_count=textured_material_count,
    )
    return UsdMeshLoadResult(mesh=scene, stats=stats)


def _require_usd() -> Tuple[Any, Any, Any, Any]:
    try:
        from pxr import Gf, Usd, UsdGeom, UsdShade
    except ImportError as exc:
        raise RuntimeError(
            "USD mesh conversion requires the pxr module. Install usd-core==26.3 "
            "in the active GenZI environment."
        ) from exc
    return Usd, UsdGeom, Gf, UsdShade


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
    if options.texture_mode not in TEXTURE_MODES:
        raise RuntimeError(
            "USD texture mode must be one of {}. Got: {}".format(
                sorted(TEXTURE_MODES), options.texture_mode
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
    mesh_data = _read_mesh_prim_data(usd_mesh, xform_cache, time_code, Gf)
    return mesh_data.vertices, mesh_data.faces


def _read_mesh_prim_data(
    usd_mesh: Any,
    xform_cache: Any,
    time_code: Any,
    Gf: Any,
    extra_transform: Optional[np.ndarray] = None,
) -> _UsdMeshPrimData:
    prim = usd_mesh.GetPrim()
    points = usd_mesh.GetPointsAttr().Get(time_code)
    face_vertex_counts = usd_mesh.GetFaceVertexCountsAttr().Get(time_code)
    face_vertex_indices = usd_mesh.GetFaceVertexIndicesAttr().Get(time_code)

    if points is None or face_vertex_counts is None or face_vertex_indices is None:
        empty_faces = np.empty((0, 3), dtype=np.int64)
        return _UsdMeshPrimData(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=empty_faces,
            face_corner_indices=empty_faces,
            source_face_indices=np.empty((0,), dtype=np.int64),
        )

    vertices = _transform_points(points, xform_cache.GetLocalToWorldTransform(prim), Gf)
    vertices = _apply_extra_transform(vertices, extra_transform)
    faces, corner_indices, source_face_indices = _triangulate_faces_with_corners(
        face_vertex_counts, face_vertex_indices, len(vertices)
    )
    return _UsdMeshPrimData(
        vertices=vertices,
        faces=faces,
        face_corner_indices=corner_indices,
        source_face_indices=source_face_indices,
    )


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
    faces, _, _ = _triangulate_faces_with_corners(
        face_vertex_counts, face_vertex_indices, num_vertices
    )
    return faces


def _triangulate_faces_with_corners(
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    num_vertices: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = [int(count) for count in face_vertex_counts]
    indices = [int(index) for index in face_vertex_indices]

    if sum(counts) != len(indices):
        raise RuntimeError(
            "USD mesh faceVertexCounts/faceVertexIndices mismatch: {} vs {}".format(
                sum(counts), len(indices)
            )
        )

    faces = []
    corner_indices = []
    source_face_indices = []
    offset = 0
    for source_face_id, count in enumerate(counts):
        face = indices[offset : offset + count]
        corners = list(range(offset, offset + count))
        offset += count

        if count < 3:
            continue
        if min(face) < 0 or max(face) >= num_vertices:
            raise RuntimeError("USD mesh face index is outside points array bounds.")

        if count == 3:
            faces.append(face)
            corner_indices.append(corners)
            source_face_indices.append(source_face_id)
        elif count == 4:
            faces.append([face[0], face[1], face[2]])
            corner_indices.append([corners[0], corners[1], corners[2]])
            source_face_indices.append(source_face_id)
            faces.append([face[0], face[2], face[3]])
            corner_indices.append([corners[0], corners[2], corners[3]])
            source_face_indices.append(source_face_id)
        else:
            for idx in range(1, count - 1):
                faces.append([face[0], face[idx], face[idx + 1]])
                corner_indices.append([corners[0], corners[idx], corners[idx + 1]])
                source_face_indices.append(source_face_id)

    return (
        np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
        np.asarray(corner_indices, dtype=np.int64).reshape((-1, 3)),
        np.asarray(source_face_indices, dtype=np.int64).reshape((-1,)),
    )


def _get_material_face_groups(
    prim: Any,
    num_source_faces: int,
    UsdGeom: Any,
    UsdShade: Any,
) -> List[Tuple[np.ndarray, Any]]:
    assigned = np.zeros((num_source_faces,), dtype=bool)
    groups: List[Tuple[np.ndarray, Any]] = []

    for child in prim.GetChildren():
        if not child.IsA(UsdGeom.Subset):
            continue
        subset = UsdGeom.Subset(child)
        indices = subset.GetIndicesAttr().Get()
        if indices is None:
            continue
        face_ids = np.asarray([int(index) for index in indices], dtype=np.int64)
        face_ids = face_ids[(face_ids >= 0) & (face_ids < num_source_faces)]
        if face_ids.size == 0:
            continue
        material = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial()[0]
        assigned[face_ids] = True
        groups.append((face_ids, material))

    remaining = np.nonzero(~assigned)[0].astype(np.int64)
    if remaining.size > 0:
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        groups.append((remaining, material))
    return groups


def _build_textured_submesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_corner_indices: np.ndarray,
    uv_values: Optional[np.ndarray],
    uv_interpolation: Optional[str],
    material: Any,
    default_color: np.ndarray,
    usd_path: Path,
    options: UsdMeshLoadOptions,
    time_code: Any,
    UsdShade: Any,
    trimesh: Any,
    material_cache: Dict[str, Any],
) -> Optional[Any]:
    if len(faces) == 0:
        return None

    unique_vertices, inverse = np.unique(faces.reshape((-1,)), return_inverse=True)
    compact_vertices = vertices[unique_vertices]
    compact_faces = inverse.reshape((-1, 3))

    texture_path = None
    diffuse_color = default_color
    if material:
        texture_path, material_color = _read_lambert_material_info(
            material=material,
            usd_path=usd_path,
            time_code=time_code,
            UsdShade=UsdShade,
        )
        if material_color is not None:
            diffuse_color = material_color

    if (
        options.texture_mode == "diffuse"
        and texture_path is not None
        and uv_values is not None
    ):
        uv = _compact_uvs(
            unique_vertices=unique_vertices,
            inverse=inverse,
            face_corner_indices=face_corner_indices,
            uv_values=uv_values,
            uv_interpolation=uv_interpolation,
        )
        if uv is not None:
            pbr_material = _get_trimesh_pbr_material(
                texture_path=texture_path,
                base_color=diffuse_color,
                trimesh=trimesh,
                material_cache=material_cache,
            )
            visual = trimesh.visual.texture.TextureVisuals(
                uv=uv,
                material=pbr_material,
            )
            return trimesh.Trimesh(
                vertices=compact_vertices,
                faces=compact_faces,
                visual=visual,
                process=False,
                validate=False,
            )

    color = np.clip(np.append(diffuse_color, 1.0) * 255.0, 0, 255).astype(np.uint8)
    vertex_colors = np.tile(color.reshape((1, 4)), (len(compact_vertices), 1))
    return trimesh.Trimesh(
        vertices=compact_vertices,
        faces=compact_faces,
        vertex_colors=vertex_colors,
        process=False,
        validate=False,
    )


def _read_uv_primvar(
    usd_mesh: Any, time_code: Any, UsdGeom: Any
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    primvar = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).GetPrimvar("st")
    if not primvar:
        return None, None
    values = primvar.ComputeFlattened(time_code)
    if values is None:
        return None, None
    uv = np.asarray(values, dtype=np.float32).reshape((-1, 2))
    return uv, str(primvar.GetInterpolation())


def _compact_uvs(
    unique_vertices: np.ndarray,
    inverse: np.ndarray,
    face_corner_indices: np.ndarray,
    uv_values: np.ndarray,
    uv_interpolation: Optional[str],
) -> Optional[np.ndarray]:
    if uv_values.size == 0:
        return None

    interpolation = str(uv_interpolation or "")
    if interpolation in {"vertex", "varying"} and len(uv_values) > int(unique_vertices.max()):
        return uv_values[unique_vertices].astype(np.float32, copy=True)
    if interpolation == "constant" and len(uv_values) >= 1:
        return np.tile(uv_values[0].reshape((1, 2)), (len(unique_vertices), 1))
    if interpolation in {"faceVarying", "uniform", ""}:
        flat_corners = face_corner_indices.reshape((-1,))
        if np.any(flat_corners >= len(uv_values)):
            return None
        flat_uv = uv_values[flat_corners]
        local_ids, first_indices = np.unique(inverse, return_index=True)
        compact_uv = np.zeros((len(unique_vertices), 2), dtype=np.float32)
        compact_uv[local_ids] = flat_uv[first_indices]
        return compact_uv
    return None


def _read_display_color(usd_mesh: Any, time_code: Any) -> np.ndarray:
    primvar = usd_mesh.GetDisplayColorPrimvar()
    if primvar:
        colors = primvar.ComputeFlattened(time_code)
        if colors is not None and len(colors) > 0:
            color = np.asarray(colors[0], dtype=np.float32).reshape((3,))
            return np.clip(color, 0.0, 1.0)
    return np.asarray([0.6, 0.6, 0.6], dtype=np.float32)


def _read_lambert_material_info(
    material: Any,
    usd_path: Path,
    time_code: Any,
    UsdShade: Any,
) -> Tuple[Optional[Path], Optional[np.ndarray]]:
    surface_shader = _find_usd_preview_surface(material, UsdShade)
    if surface_shader is None:
        return _find_named_diffuse_texture(material, usd_path, time_code, UsdShade), None

    diffuse_input = surface_shader.GetInput("diffuseColor")
    if not diffuse_input:
        return _find_named_diffuse_texture(material, usd_path, time_code, UsdShade), None

    texture_path = _find_connected_texture_path(diffuse_input, usd_path, time_code, UsdShade)
    diffuse_color = None
    value = diffuse_input.Get(time_code)
    if value is not None:
        diffuse_color = np.clip(np.asarray(value, dtype=np.float32).reshape((3,)), 0.0, 1.0)
    if texture_path is None:
        texture_path = _find_named_diffuse_texture(material, usd_path, time_code, UsdShade)
    return texture_path, diffuse_color


def _find_usd_preview_surface(material: Any, UsdShade: Any) -> Optional[Any]:
    for child in material.GetPrim().GetChildren():
        shader = UsdShade.Shader(child)
        if shader and str(shader.GetIdAttr().Get()) == "UsdPreviewSurface":
            return shader
    return None


def _find_connected_texture_path(
    input_attr: Any,
    usd_path: Path,
    time_code: Any,
    UsdShade: Any,
) -> Optional[Path]:
    sources, _ = input_attr.GetConnectedSources()
    for source in sources:
        shader = UsdShade.Shader(source.source)
        texture_path = _read_texture_shader_path(shader, usd_path, time_code)
        if texture_path is not None:
            return texture_path
    return None


def _find_named_diffuse_texture(
    material: Any,
    usd_path: Path,
    time_code: Any,
    UsdShade: Any,
) -> Optional[Path]:
    for child in material.GetPrim().GetChildren():
        name = str(child.GetName()).lower()
        if "diffuse" not in name and "basecolor" not in name and "albedo" not in name:
            continue
        shader = UsdShade.Shader(child)
        texture_path = _read_texture_shader_path(shader, usd_path, time_code)
        if texture_path is not None:
            return texture_path
    return None


def _read_texture_shader_path(shader: Any, usd_path: Path, time_code: Any) -> Optional[Path]:
    if not shader or str(shader.GetIdAttr().Get()) != "UsdUVTexture":
        return None
    file_input = shader.GetInput("file")
    if not file_input:
        return None
    asset_path = file_input.Get(time_code)
    if asset_path is None:
        return None

    resolved_path = getattr(asset_path, "resolvedPath", None)
    authored_path = getattr(asset_path, "path", None) or getattr(asset_path, "authoredPath", None)
    candidate = Path(str(resolved_path)) if resolved_path else Path(str(authored_path))
    if not candidate.is_absolute():
        candidate = usd_path.parent / candidate
    if candidate.is_file():
        return candidate
    return None


def _get_trimesh_pbr_material(
    texture_path: Path,
    base_color: np.ndarray,
    trimesh: Any,
    material_cache: Dict[str, Any],
) -> Any:
    cache_key = "{}|{}".format(texture_path, ",".join(map(str, base_color.tolist())))
    if cache_key in material_cache:
        return material_cache[cache_key]

    from PIL import Image

    image = Image.open(texture_path).convert("RGBA")
    base = np.clip(np.append(base_color, 1.0), 0.0, 1.0)
    material = trimesh.visual.texture.PBRMaterial(
        baseColorTexture=image,
        baseColorFactor=base,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
    )
    material_cache[cache_key] = material
    return material


def _build_extra_transform(
    rotation_degrees: Optional[Tuple[float, float, float]]
) -> Optional[np.ndarray]:
    if rotation_degrees is None:
        return None
    rx, ry, rz = [float(value) * np.pi / 180.0 for value in rotation_degrees]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64
    )
    my = np.asarray(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64
    )
    mz = np.asarray(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return mz @ my @ mx


def _apply_extra_transform(
    vertices: np.ndarray, extra_transform: Optional[np.ndarray]
) -> np.ndarray:
    if extra_transform is None:
        return vertices
    return vertices @ extra_transform.T


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _get_stage_meters_per_unit(stage: Any, UsdGeom: Any) -> Optional[float]:
    if not hasattr(UsdGeom, "GetStageMetersPerUnit"):
        return None
    try:
        return float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        return None
