from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from genzi.misc import get_optim_human_mesh_path, save_smplx_mesh


class TestOptimHumanMeshPath(unittest.TestCase):
    def test_prefers_obj_when_obj_and_ply_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            obj_path = root / "optim_human.obj"
            ply_path = root / "optim_human.ply"
            obj_path.write_text("# obj\n", encoding="utf-8")
            ply_path.write_text("ply\n", encoding="utf-8")

            self.assertEqual(get_optim_human_mesh_path(root), obj_path)

    def test_falls_back_to_ply_when_obj_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ply_path = root / "optim_human.ply"
            ply_path.write_text("ply\n", encoding="utf-8")

            self.assertEqual(get_optim_human_mesh_path(root), ply_path)

    def test_raises_clear_error_when_no_mesh_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                FileNotFoundError, "optim_human.obj.*optim_human.ply"
            ):
                get_optim_human_mesh_path(tmpdir)


class TestSaveSmplxMesh(unittest.TestCase):
    def test_skips_obj_export_when_uv_template_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "optim_human.obj"
            vertices = np.zeros((3, 3), dtype=np.float32)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = save_smplx_mesh(
                    str(out_path),
                    str(Path(tmpdir) / "missing_uv_template.txt"),
                    "",
                    vertices,
                )

            self.assertFalse(result)
            self.assertFalse(out_path.exists())
            self.assertFalse((Path(tmpdir) / "optim_human.mtl").exists())
            self.assertTrue(
                any("UV template is missing" in str(item.message) for item in caught)
            )

    def test_missing_texture_does_not_block_untextured_obj_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_path = root / "optim_human.obj"
            uv_template = root / "smplx_uv_template.txt"
            missing_texture = root / "missing_texture.png"
            vertices = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            )
            uv_template.write_text(
                "\n".join(
                    [
                        "vt 0.000000 0.000000",
                        "vt 1.000000 0.000000",
                        "vt 0.000000 1.000000",
                        "usemtl material_0",
                        "f 1/1 2/2 3/3",
                    ]
                ),
                encoding="utf-8",
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = save_smplx_mesh(
                    str(out_path),
                    str(uv_template),
                    str(missing_texture),
                    vertices,
                )

            self.assertTrue(result)
            self.assertTrue(out_path.is_file())
            mtl_text = (root / "optim_human.mtl").read_text(encoding="utf-8")
            self.assertNotIn("map_Kd", mtl_text)
            self.assertTrue(
                any("texture file is missing" in str(item.message) for item in caught)
            )


if __name__ == "__main__":
    unittest.main()
