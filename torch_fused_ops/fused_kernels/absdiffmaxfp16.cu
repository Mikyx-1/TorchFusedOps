#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/types.h>
#include <torch/extension.h>

/**
 * @brief AbsDiffMax kernel for float16 (~ torch.abs(a - b).max())
 *
 *
 *
 *
 * */
__global__
void absdiffmax_blocks_kernel(const __half* __restrict__ a,
                              const __half* __restrict__ b,
                              __half* __restrict__ block_max,
                              const int N)
{


  extern __shared__ __half smem[];

  int tid = threadIdx.x;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  __half local_max = __float2half(0.0f);

  for (int i = thread_idx; i < N; i += blockDim.x * gridDim.x)
  {

    __half val = __habs(__hsub(a[i], b[i]));
    local_max = __hmax(local_max, val);
  }
  smem[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      smem[tid] = __hmax(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0){
    block_max[blockIdx.x] = smem[0];
  }
}



torch::Tensor absdiffmax(torch::Tensor a, torch::Tensor b)
{
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors!");
  TORCH_CHECK(a.scalar_type() == torch::kFloat16, "Only float16 tensors are supported!");
  TORCH_CHECK(a.numel() == b.numel(), "Inputs must have the same shape!");

  const int N = a.numel();
  const int threads = 256;
  const int blocks = 128;
  const int shared_mem_size = threads * sizeof(__half);

  torch::Tensor block_max = torch::empty({blocks}, torch::TensorOptions().dtype(torch::kFloat16).device(a.device()));

  absdiffmax_blocks_kernel<<<blocks, threads, shared_mem_size>>>(
      reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(block_max.data_ptr<at::Half>()),
      N
      );

  cudaDeviceSynchronize();

  return block_max.max();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("absdiffmax", torch::wrap_pybind_function(absdiffmax), "...");
}
