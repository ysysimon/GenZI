"""Build AlphaPose DCN CUDA extensions in place."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHAPOSE_ROOT = REPO_ROOT / "external" / "alphapose"


def make_cuda_ext(name: str, sources: list[str]) -> CUDAExtension:
    module = "alphapose.models.layers.dcn"
    return CUDAExtension(
        name=f"{module}.{name}",
        sources=[str(ALPHAPOSE_ROOT / module.replace(".", os.sep) / src) for src in sources],
        extra_compile_args={
            "cxx": ["/Zc:preprocessor"],
            "nvcc": [
                "-allow-unsupported-compiler",
                "-std=c++17",
                "-Xcompiler=/Zc:preprocessor",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )


def main() -> None:
    if os.environ.get("GENZI_ALLOW_CUDA_MISMATCH") == "1":
        cpp_extension._check_cuda_version = lambda compiler_name, compiler_version: None

    os.chdir(ALPHAPOSE_ROOT)
    setup(
        name="alphapose_dcn_extensions",
        ext_modules=[
            make_cuda_ext(
                "deform_conv_cuda",
                [
                    "src/deform_conv_cuda.cpp",
                    "src/deform_conv_cuda_kernel.cu",
                ],
            ),
            make_cuda_ext(
                "deform_pool_cuda",
                [
                    "src/deform_pool_cuda.cpp",
                    "src/deform_pool_cuda_kernel.cu",
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        script_args=["build_ext", "--inplace"],
    )


if __name__ == "__main__":
    main()
