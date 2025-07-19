import time
import torch



def warmup(fn, iters=10):
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()


def benchmark_fn(fn, iters=50):
    times = []
    result = None
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.time()
        result = fn()
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

    return result, torch.tensor(times).mean().item(), torch.tensor(times).std().item()


OP_REGISTRY = {}

def register_op(name):
    def wrapper(fn):
        OP_REGISTRY[name] = fn
        return fn
    return wrapper


