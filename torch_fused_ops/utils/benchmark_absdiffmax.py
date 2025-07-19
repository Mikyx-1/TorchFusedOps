import torch
from torch.utils.cpp_extension import load
from benchmark_base import warmup, benchmark_fn, register_op, OP_REGISTRY

# Load custom CUDA extension
ext = load(
    name="add",
    sources=["./torch_fused_ops/fused_kernels/absdiffmax.cu"],
    extra_cuda_cflags=["-O2"],
    extra_cflags=["-O2"],
    verbose=True,
)

# Input setup
device = "cuda:0"
dtype = torch.float32
a = torch.randn(100_000_000, device=device, dtype=dtype)
b = torch.randn(100_000_000, device=device, dtype=dtype)

# Register implementations
@register_op("pytorch")
def pytorch_impl():
    return (a - b).abs().max()

@register_op("fused")
def fused_impl():
    return ext.absdiffmax(a, b)

# Run
for name, fn in OP_REGISTRY.items():
    print(f"== Running: {name} ==")
    warmup(fn)
    result, mean, std = benchmark_fn(fn)
    print(f"{name:10}: {mean*1e3:.3f} ms ± {std*1e3:.3f} ms | Value: {result.item()}")
