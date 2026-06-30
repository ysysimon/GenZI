from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import check_special_deps
from tools import install_nvdiffrast as installer


def make_cuda_home(root: Path) -> Path:
    cuda_home = root / "cuda"
    bin_dir = cuda_home / "bin"
    bin_dir.mkdir(parents=True)
    nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
    (bin_dir / nvcc_name).write_text("", encoding="utf-8")
    return cuda_home


class TestNVDiffRastInstaller(unittest.TestCase):
    def test_resolve_cuda_home_prefers_cli_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cli_cuda = make_cuda_home(root / "cli")
            env_cuda = make_cuda_home(root / "env")

            resolved = installer.resolve_cuda_home(
                str(cli_cuda),
                {"CUDA_HOME": str(env_cuda)},
            )

            self.assertEqual(resolved, cli_cuda.resolve())

    def test_resolve_cuda_home_uses_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cuda_home = make_cuda_home(Path(tmpdir))

            resolved = installer.resolve_cuda_home(
                None,
                {"CUDA_HOME": str(cuda_home)},
            )

            self.assertEqual(resolved, cuda_home.resolve())

    def test_resolve_cuda_home_requires_nvcc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cuda_home = Path(tmpdir) / "cuda"
            cuda_home.mkdir()

            with self.assertRaisesRegex(RuntimeError, "nvcc"):
                installer.resolve_cuda_home(str(cuda_home), {})

    def test_build_install_env_sets_cuda_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cuda_home = make_cuda_home(Path(tmpdir))

            env = installer.build_install_env(
                cuda_home,
                base_env={"PATH": "old-path"},
            )

            self.assertEqual(env["CUDA_HOME"], str(cuda_home.resolve()))
            self.assertEqual(env["CUDA_PATH"], str(cuda_home.resolve()))
            self.assertTrue(env["PATH"].startswith(str(cuda_home.resolve() / "bin")))

    def test_build_install_env_can_add_windows_compat_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cuda_home = make_cuda_home(Path(tmpdir))

            with mock.patch.object(installer.platform, "system", return_value="Windows"):
                env = installer.build_install_env(
                    cuda_home,
                    allow_unsupported_msvc=True,
                    base_env={"PATH": "old-path"},
                )

            self.assertEqual(env["DISTUTILS_USE_SDK"], "1")
            self.assertIn(
                "-allow-unsupported-compiler",
                env["NVCC_PREPEND_FLAGS"],
            )

    def test_install_build_helpers_uses_current_python(self):
        with mock.patch.object(installer, "uv_command", return_value="uv"):
            with mock.patch.object(installer, "run") as run_mock:
                installer.install_build_helpers({"PATH": "test"})

        self.assertEqual(
            run_mock.call_args.args[0],
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "setuptools",
                "wheel",
                "ninja",
            ],
        )

    def test_install_nvdiffrast_uses_pinned_git_spec_and_no_build_isolation(self):
        with mock.patch.object(installer, "uv_command", return_value="uv"):
            with mock.patch.object(installer, "run") as run_mock:
                installer.install_nvdiffrast(
                    installer.DEFAULT_REV,
                    {"PATH": "test"},
                    force_reinstall=True,
                )

        cmd = run_mock.call_args.args[0]
        self.assertEqual(
            cmd[:6],
            ["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation"],
        )
        self.assertIn("--force-reinstall", cmd)
        self.assertEqual(
            cmd[-1],
            f"git+{installer.NVDIFFRAST_URL}@{installer.DEFAULT_REV}",
        )

    def test_doctor_nvdiffrast_hint_names_installer(self):
        with mock.patch.object(
            check_special_deps.importlib,
            "import_module",
            side_effect=ImportError("boom"),
        ):
            result = check_special_deps.check_import(
                "nvdiffrast",
                "nvdiffrast.torch",
                check_special_deps.NVDIFFRAST_INSTALL_HINT,
            )

        self.assertIn("tools/install_nvdiffrast.py", result.detail)
        self.assertIn("setuptools", result.detail)
        self.assertIn("CUDA_HOME", result.detail)


if __name__ == "__main__":
    unittest.main()
