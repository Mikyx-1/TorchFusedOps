I want you to reformat so this package is ready for production


Users only need to do

`pip install TorchFusedOps`


And then usage looks like this


```python
import torch
import torch_fused_ops as tfo

# Create some tensors
a = torch.randn(1000000, device="cuda")
b = torch.randn(1000000, device="cuda")
c = torch.randn(1000000, device="cuda")
d = torch.randn(1000000, device="cuda")

# Option 1: Codegen Backend (default)
fused_fn = tfo.fuse(
    lambda a, b, c, d: a + b + c - d,
    backend="codegen"
)

# Option 2: VM Backend
fused_fn = tfo.fuse(
    lambda a, b, c, d: a + b * c - d,
    backend="vm"
)

# Use the fused function
result = fused_fn(a, b, c, d)

# Compare with PyTorch eager execution
eager_result = a + b + c - d
print(f"Results match: {torch.allclose(result, eager_result)}")
```


Please help me.