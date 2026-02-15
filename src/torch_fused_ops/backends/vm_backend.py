from .. import _vm_backend
from ..tracer import Tracer, build_program
import torch



def fuse_vm(fn):
    arg_names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    tracers = [Tracer(name) for name in arg_names]
    expr = fn(*tracers)

    program = build_program(expr, list(arg_names))
    program_tensor = torch.tensor(program, dtype=torch.int32, device="cuda")

    def runner(*real_inputs):
        return _vm_backend.run_fused(list(real_inputs), program_tensor)

    return runner


