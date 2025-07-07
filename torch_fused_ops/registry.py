FUSED_OPS = {
    "abs_sub_max": {
        "description": "Compute max(abs(a - b)) in a fused pass.",
        "devices": ["CUDA", "CPU"],
        "dtypes": ["float32"],
        "added_in": "0.1.0"
    },
    "squared_diff_sum": {
        "description": "Compute sum((a - b)**2) in a fused pass.",
        "devices": ["CUDA", "CPU"],
        "dtypes": ["float32"],
        "added_in": "0.1.0"
    }
    # ... add more as you create them
}
