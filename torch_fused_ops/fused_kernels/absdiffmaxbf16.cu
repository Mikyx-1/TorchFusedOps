#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>  // For __nv_bfloat16 and math ops
#include <torch/types.h>
#include <torch/extension.h>

__device__ __nv_bfloat16 bf16_absdiff(const __nv_bfloat16 a, const __nv_bfloat16 b) {
    return __habs(__hsub(a, b));  // abs(a - b)
}

__device__ __nv_bfloat16 bf16_max(const __nv_bfloat16 a, const __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hmax(a, b);  // Ampere+ supports this
#else
    return (__hlt(a, b) ? b : a);  // Manual fallback for older GPUs
#endif
}

__global__
void absdiffmax_blocks_kernel_bf16(const __nv_bfloat16* __restrict__ a,
                                   const __nv_bfloat16* __restrict__ b,
                                   __nv_bfloat16* __restrict__ block_max,
                                   const int N)
{
    extern __shared__ __nv_bfloat16 smem[];

    int tid = threadIdx.x;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    __nv_bfloat16 local_max = __float2bfloat16(0.0f);

    for (int i = thread_idx; i < N; i += blockDim.x * gridDim.x)
    {
        __nv_bfloat16 val = bf16_absdiff(a[i], b[i]);
        local_max = bf16_max(local_max, val);
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            smem[tid] = bf16_max(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        block_max[blockIdx.x] = smem[0];
    }
}

torch::Tensor absdiffmax_bf16(torch::Tensor a, torch::Tensor b)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors!");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "Only bfloat16 tensors are supported!");
    TORCH_CHECK(a.numel() == b.numel(), "Inputs must have the same shape!");

    const int N = a.numel();
    const int threads = 256;
    const int blocks = 128;
    const int shared_mem_size = threads * sizeof(__nv_bfloat16);

    torch::Tensor block_max = torch::empty({blocks}, torch::TensorOptions().dtype(torch::kBFloat16).device(a.device()));

    absdiffmax_blocks_kernel_bf16<<<blocks, threads, shared_mem_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(block_max.data_ptr<at::BFloat16>()),
        N
    );

    cudaDeviceSynchronize();

    return block_max.max();  // result is bfloat16
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("absdiffmax", torch::wrap_pybind_function(absdiffmax_bf16), "AbsDiffMax for bfloat16");
}

