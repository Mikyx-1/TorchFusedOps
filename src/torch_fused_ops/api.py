from .backends.vm_backend import fuse_vm
from .backends.codegen_backend import fuse_codegen


def fuse(fn, backend="codegen"):
    if backend == "codegen":
        return fuse_codegen(fn)
    elif backend == "vm":
        return fuse_vm(fn)
    else:
        raise ValueError(f"Unknown backend: {backend}")
