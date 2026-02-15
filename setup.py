from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="TorchFusedOps",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # ← THIS IS THE FIX
    ext_modules=[
        CUDAExtension(
            name="torch_fused_ops._vm_backend",
            sources=[
                "src/torch_fused_ops/cpp/vm_kernel.cu",
                "src/torch_fused_ops/cpp/binding.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
