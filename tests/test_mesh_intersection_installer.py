from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import check_special_deps
from tools import install_mesh_intersection as installer


class TestMeshIntersectionInstaller(unittest.TestCase):
    def test_resolve_cuda_samples_inc_prefers_cli_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cli_dir = root / "cli"
            env_dir = root / "env"
            cli_dir.mkdir()
            env_dir.mkdir()
            (cli_dir / installer.HELPER_MATH_HEADER).write_text("", encoding="utf-8")

            resolved = installer.resolve_cuda_samples_inc(
                str(cli_dir),
                {"CUDA_SAMPLES_INC": str(env_dir)},
            )

            self.assertEqual(resolved, cli_dir.resolve())

    def test_resolve_cuda_samples_inc_uses_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            include_dir = Path(tmpdir) / "include"
            include_dir.mkdir()
            (include_dir / installer.HELPER_MATH_HEADER).write_text(
                "",
                encoding="utf-8",
            )

            resolved = installer.resolve_cuda_samples_inc(
                None,
                {"CUDA_SAMPLES_INC": str(include_dir)},
            )

            self.assertEqual(resolved, include_dir.resolve())

    def test_resolve_cuda_samples_inc_requires_helper_math(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            include_dir = Path(tmpdir) / "include"
            include_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, installer.HELPER_MATH_HEADER):
                installer.resolve_cuda_samples_inc(str(include_dir), {})

    def test_ensure_checkout_rejects_existing_non_git_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "torch-mesh-isect"
            target.mkdir()

            with self.assertRaisesRegex(RuntimeError, "git checkout"):
                installer.ensure_checkout(target, installer.DEFAULT_REV)

    def test_install_command_uses_uv_current_python_and_no_build_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "torch-mesh-isect"
            include_dir = Path(tmpdir) / "cuda-samples"
            target.mkdir()
            include_dir.mkdir()
            (include_dir / installer.HELPER_MATH_HEADER).write_text(
                "",
                encoding="utf-8",
            )

            with mock.patch.object(installer, "uv_command", return_value="uv"):
                with mock.patch.object(installer, "run") as run_mock:
                    installer.install_mesh_intersection(target, include_dir)

            cmd = run_mock.call_args.args[0]
            env = run_mock.call_args.kwargs["env"]
            self.assertEqual(
                cmd,
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "--no-build-isolation",
                    str(target.resolve()),
                ],
            )
            self.assertEqual(env["CUDA_SAMPLES_INC"], str(include_dir.resolve()))

    def test_doctor_mesh_intersection_hint_names_installer_and_cuda_header(self):
        with mock.patch.object(
            check_special_deps.importlib,
            "import_module",
            side_effect=ImportError("boom"),
        ):
            result = check_special_deps.check_import(
                "mesh_intersection",
                "mesh_intersection.bvh_search_tree",
                check_special_deps.MESH_INTERSECTION_INSTALL_HINT,
            )

        self.assertIn("tools/install_mesh_intersection.py", result.detail)
        self.assertIn("CUDA_SAMPLES_INC", result.detail)
        self.assertIn("helper_math.h", result.detail)


if __name__ == "__main__":
    unittest.main()
