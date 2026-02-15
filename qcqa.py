import torch
import torch_fused_ops as tfo
import time

shape = (10_000_000,)
a = torch.randn(shape, device="cuda")
b = torch.randn(shape, device="cuda")
c = torch.randn(shape, device="cuda")
d = torch.randn(shape, device="cuda")

# fused_fn = tfo.fuse(
#     lambda a, b, c, d: a + b + c - d,
#     backend="vm"
# )

# result = fused_fn(a, b, c, d)
# gt =  a + b + c - d

# print(f"all_match: {torch.allclose(result, gt)}")

def benchmark(fn, iters=100):
    # Warmup
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iters):
        fn()

    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000 / iters  # ms


# ----------------------------
# Prepare fused and eager
# ----------------------------

fused_fn = tfo.fuse(
    # lambda a, b, c, d: a + b + c + d,
    lambda a, b, c, d: f"{a} + {b} + {c} + {d}",
    backend="codegen"
)

def fused_run():
    return fused_fn(a, b, c, d)

def eager_run():
    return a + b + c + d


# ----------------------------
# Benchmark
# ----------------------------

fused_time = benchmark(fused_run)
eager_time = benchmark(eager_run)

print(f"Correct: {torch.allclose(fused_run(), eager_run())}")
print(f"Eager:  {eager_time:.4f} ms")
print(f"Fused:  {fused_time:.4f} ms")
print(f"Speedup: {eager_time / fused_time:.2f}x")