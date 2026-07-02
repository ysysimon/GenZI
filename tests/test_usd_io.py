from __future__ import annotations

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from genzi.usd_io import UsdMeshLoadOptions, load_usd_mesh

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Gf = None
    Usd = None
    UsdGeom = None


def require_pxr(func):
    return unittest.skipUnless(Usd is not None, "usd-core/pxr is not installed")(func)


def save_stage(path, mesh_specs):
    stage = Usd.Stage.CreateNew(str(path))
    for spec in mesh_specs:
        prim_path = spec["path"]
        if spec.get("translate") is not None:
            parent_path = "/".join(prim_path.split("/")[:-1])
            xform = UsdGeom.Xform.Define(stage, parent_path)
            xform.AddTranslateOp().Set(Gf.Vec3d(*spec["translate"]))

        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in spec["points"]])
        mesh.CreateFaceVertexCountsAttr(spec["counts"])
        mesh.CreateFaceVertexIndicesAttr(spec["indices"])

        imageable = UsdGeom.Imageable(mesh.GetPrim())
        if spec.get("invisible"):
            imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        if spec.get("purpose") is not None:
            imageable.CreatePurposeAttr().Set(spec["purpose"])

    stage.GetRootLayer().Save()


class TestUsdIo(unittest.TestCase):
    @require_pxr
    def test_load_triangle_mesh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "triangle.usda"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/Triangle",
                        "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    }
                ],
            )

            mesh = load_usd_mesh(str(usd_path))

            self.assertEqual(len(mesh.vertices), 3)
            self.assertEqual(len(mesh.faces), 1)
            np.testing.assert_array_equal(mesh.faces, np.asarray([[0, 1, 2]]))

    @require_pxr
    def test_triangulates_quad_and_ngon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "polygons.usda"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/Polygons",
                        "points": [
                            (0, 0, 0),
                            (1, 0, 0),
                            (1, 1, 0),
                            (0, 1, 0),
                            (-1, 1, 0),
                        ],
                        "counts": [4, 5],
                        "indices": [0, 1, 2, 3, 0, 1, 2, 3, 4],
                    }
                ],
            )

            mesh = load_usd_mesh(str(usd_path))

            self.assertEqual(len(mesh.faces), 5)
            np.testing.assert_array_equal(
                mesh.faces,
                np.asarray(
                    [
                        [0, 1, 2],
                        [0, 2, 3],
                        [0, 1, 2],
                        [0, 2, 3],
                        [0, 3, 4],
                    ]
                ),
            )

    @require_pxr
    def test_bakes_parent_transform(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "translated.usda"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/Root/Triangle",
                        "translate": (1, 2, 3),
                        "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    }
                ],
            )

            mesh = load_usd_mesh(str(usd_path))

            np.testing.assert_allclose(mesh.bounds[0], [1, 2, 3])
            np.testing.assert_allclose(mesh.bounds[1], [2, 3, 3])

    @require_pxr
    def test_skips_invisible_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "visibility.usda"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/Visible",
                        "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                    {
                        "path": "/Hidden",
                        "points": [(10, 0, 0), (11, 0, 0), (10, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                        "invisible": True,
                    },
                ],
            )

            mesh = load_usd_mesh(str(usd_path))
            with_hidden = load_usd_mesh(
                str(usd_path), options=UsdMeshLoadOptions(include_invisible=True)
            )

            self.assertEqual(len(mesh.faces), 1)
            self.assertEqual(len(with_hidden.faces), 2)

    @require_pxr
    def test_filters_purpose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "purpose.usda"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/Default",
                        "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                    {
                        "path": "/Render",
                        "points": [(10, 0, 0), (11, 0, 0), (10, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                        "purpose": UsdGeom.Tokens.render,
                    },
                    {
                        "path": "/Proxy",
                        "points": [(20, 0, 0), (21, 0, 0), (20, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                        "purpose": UsdGeom.Tokens.proxy,
                    },
                ],
            )

            default_mesh = load_usd_mesh(str(usd_path))
            render_mesh = load_usd_mesh(
                str(usd_path), options=UsdMeshLoadOptions(purpose="render")
            )
            all_mesh = load_usd_mesh(
                str(usd_path), options=UsdMeshLoadOptions(purpose="all")
            )

            self.assertEqual(len(default_mesh.faces), 1)
            self.assertEqual(len(render_mesh.faces), 2)
            self.assertEqual(len(all_mesh.faces), 3)

    @require_pxr
    def test_prim_path_filters_parent_subtree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "subtree.usda"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/World/Keep/TriangleA",
                        "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                    {
                        "path": "/World/Keep/TriangleB",
                        "points": [(2, 0, 0), (3, 0, 0), (2, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                    {
                        "path": "/World/Skip/TriangleC",
                        "points": [(10, 0, 0), (11, 0, 0), (10, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                ],
            )

            all_mesh = load_usd_mesh(str(usd_path))
            keep_mesh = load_usd_mesh(
                str(usd_path), options=UsdMeshLoadOptions(prim_path="/World/Keep")
            )

            self.assertEqual(len(all_mesh.faces), 3)
            self.assertEqual(len(keep_mesh.faces), 2)
            np.testing.assert_allclose(keep_mesh.bounds[0], [0, 0, 0])
            np.testing.assert_allclose(keep_mesh.bounds[1], [3, 1, 0])

    @require_pxr
    def test_prim_path_can_point_to_mesh_prim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "mesh_root.usda"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/World/Keep/TriangleA",
                        "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                    {
                        "path": "/World/Keep/TriangleB",
                        "points": [(2, 0, 0), (3, 0, 0), (2, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                ],
            )

            mesh = load_usd_mesh(
                str(usd_path),
                options=UsdMeshLoadOptions(prim_path="/World/Keep/TriangleB"),
            )

            self.assertEqual(len(mesh.faces), 1)
            np.testing.assert_allclose(mesh.bounds[0], [2, 0, 0])
            np.testing.assert_allclose(mesh.bounds[1], [3, 1, 0])

    @require_pxr
    def test_missing_prim_path_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "missing_prim.usda"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/World/Triangle",
                        "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    }
                ],
            )

            with self.assertRaisesRegex(RuntimeError, "USD prim path not found"):
                load_usd_mesh(
                    str(usd_path),
                    options=UsdMeshLoadOptions(prim_path="/World/Missing"),
                )

    @require_pxr
    def test_prim_path_combines_with_visibility_and_purpose_filters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "subtree_filters.usda"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/World/Keep/Default",
                        "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                    {
                        "path": "/World/Keep/Render",
                        "points": [(2, 0, 0), (3, 0, 0), (2, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                        "purpose": UsdGeom.Tokens.render,
                    },
                    {
                        "path": "/World/Keep/Hidden",
                        "points": [(4, 0, 0), (5, 0, 0), (4, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                        "invisible": True,
                    },
                    {
                        "path": "/World/Skip/Default",
                        "points": [(10, 0, 0), (11, 0, 0), (10, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                ],
            )

            default_mesh = load_usd_mesh(
                str(usd_path), options=UsdMeshLoadOptions(prim_path="/World/Keep")
            )
            render_mesh = load_usd_mesh(
                str(usd_path),
                options=UsdMeshLoadOptions(
                    prim_path="/World/Keep",
                    purpose="render",
                    include_invisible=True,
                ),
            )

            self.assertEqual(len(default_mesh.faces), 1)
            self.assertEqual(len(render_mesh.faces), 3)

    @require_pxr
    def test_cli_exports_obj_readable_by_trimesh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            usd_path = tmp_path / "scene.usda"
            out_path = tmp_path / "scene.obj"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/World/Keep/Triangle",
                        "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                    {
                        "path": "/World/Skip/Triangle",
                        "points": [(10, 0, 0), (11, 0, 0), (10, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    }
                ],
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "tools/convert_usd_scene.py",
                    "--usd",
                    str(usd_path),
                    "--out",
                    str(out_path),
                    "--prim-path",
                    "/World/Keep",
                    "--print-summary",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("Converted USD mesh", proc.stdout)
            import trimesh

            converted = trimesh.load(str(out_path), process=False, validate=False)
            self.assertEqual(len(converted.faces), 1)
            np.testing.assert_allclose(converted.bounds[0], [0, 0, 0])
            np.testing.assert_allclose(converted.bounds[1], [1, 1, 0])

    @require_pxr
    def test_preview_cli_exports_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            usd_path = tmp_path / "scene.usda"
            out_path = tmp_path / "preview.png"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/World/Keep/Triangle",
                        "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                    {
                        "path": "/World/Skip/Triangle",
                        "points": [(10, 0, 0), (11, 0, 0), (10, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    }
                ],
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "tools/preview_mesh_scene.py",
                    "--mesh",
                    str(usd_path),
                    "--out",
                    str(out_path),
                    "--prim-path",
                    "/World/Keep",
                    "--max-faces",
                    "100",
                    "--sample-by",
                    "area",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("Saved preview PNG", proc.stdout)
            self.assertTrue(out_path.is_file())
            self.assertGreater(out_path.stat().st_size, 0)

    @require_pxr
    def test_misc_load_trimesh_loads_usd_in_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "scene.usda"
            save_stage(
                usd_path,
                [
                    {
                        "path": "/World/Keep/Triangle",
                        "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    },
                    {
                        "path": "/World/Skip/Triangle",
                        "points": [(10, 0, 0), (11, 0, 0), (10, 1, 0)],
                        "counts": [3],
                        "indices": [0, 1, 2],
                    }
                ],
            )

            sys.modules.setdefault("open3d", types.ModuleType("open3d"))
            from genzi.misc import load_trimesh

            mesh = load_trimesh(
                str(usd_path),
                usd_options=UsdMeshLoadOptions(prim_path="/World/Keep"),
            )

            self.assertEqual(len(mesh.vertices), 3)
            self.assertEqual(len(mesh.faces), 1)
            np.testing.assert_allclose(mesh.bounds[0], [0, 0, 0])
            np.testing.assert_allclose(mesh.bounds[1], [1, 1, 0])

    def test_scene_config_builds_usd_options(self):
        from genzi.misc import get_scene_usd_options

        options = get_scene_usd_options(
            {
                "scene.usd_prim_path": "/World/Room/ChairArea",
                "scene.usd_purpose": "all",
                "scene.usd_include_invisible": True,
                "scene.usd_time_code": 12.0,
            }
        )

        self.assertEqual(options.prim_path, "/World/Room/ChairArea")
        self.assertEqual(options.purpose, "all")
        self.assertTrue(options.include_invisible)
        self.assertEqual(options.time_code, 12.0)


if __name__ == "__main__":
    unittest.main()
