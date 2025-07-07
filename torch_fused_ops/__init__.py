from .python_api.abs_sub_max import abs_sub_max
from .python_api.squared_diff_sum import squared_diff_sum
from .registry import FUSED_OPS

__all__ = ["abs_sub_max", "squared_diff_sum", "list_fused_ops"]

def list_fused_ops():
    """List all available fused ops with descriptions."""
    for name, info in FUSED_OPS.items():
        devices = ", ".join(info["devices"])
        dtypes = ", ".join(info["dtypes"])
        print(f"- {name}: {info['description']}  (devices: {devices}, dtypes: {dtypes})")