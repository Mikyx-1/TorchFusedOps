import torch
from torch.utils.cpp_extension import load_inline
import textwrap
import hashlib

# in-memory cache (avoid recompiling in same run)
_KERNEL_CACHE = {}

def fuse_codegen(fn):

    # ----------------------------
    # Extract args + expression
    # ----------------------------
    arg_names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    expr = fn(*arg_names)   # pass strings


    expr_cuda = expr
    for name in arg_names:
        expr_cuda = expr_cuda.replace(name, f"{name}[i]")

    # ----------------------------
    # Temporary kernel name (will hash later)
    # ----------------------------
    tmp_name = "fused_tmp"

    cuda_src = f"""
    #include <torch/extension.h>

    __global__ void {tmp_name}_kernel(
        float* out,
        {", ".join([f"const float* {n}" for n in arg_names])},
        int64_t N
    )
    {{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {{
            out[i] = {expr_cuda};
        }}
    }}

    torch::Tensor {tmp_name}(
        {", ".join([f"torch::Tensor {n}" for n in arg_names])}
    )
    {{
        auto out = torch::empty_like({arg_names[0]});
        int64_t N = out.numel();

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        {tmp_name}_kernel<<<blocks, threads>>>(
            out.data_ptr<float>(),
            {", ".join([f"{n}.data_ptr<float>()" for n in arg_names])},
            N
        );

        return out;
    }}
    """

    # ----------------------------
    # Hash FULL SOURCE
    # ----------------------------
    kernel_hash = hashlib.md5(cuda_src.encode()).hexdigest()[:8]
    kernel_name = f"fused_{kernel_hash}"

    # Replace temp name with hashed name
    cuda_src = cuda_src.replace(tmp_name, kernel_name)

    # Check in-memory cache
    if kernel_name in _KERNEL_CACHE:
        return _KERNEL_CACHE[kernel_name]

    cpp_src = f"""
    #include <torch/extension.h>

    torch::Tensor {kernel_name}(
        {", ".join([f"torch::Tensor {n}" for n in arg_names])}
    );

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
        m.def("{kernel_name}", &{kernel_name});
    }}
    """

    # Compile (will hit disk cache if same hash)
    module = load_inline(
        name=kernel_name,
        cpp_sources=cpp_src,
        cuda_sources=textwrap.dedent(cuda_src),
        verbose=False,
    )

    def runner(*inputs):
        return getattr(module, kernel_name)(*inputs)

    _KERNEL_CACHE[kernel_name] = runner
    return runner

