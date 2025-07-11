#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/extension.h>

__global__
void absdiffmax_blocks_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ block_max,
                               const int N)
{
  __shared__ float smem[256];

  int tid = threadIdx.x;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  float local_max = 0.0f;

  for (int i = thread_idx; i < N; i += gridDim.x * blockDim.x) {
    float val = fabsf(a[i] - b[i]);
    local_max = fmaxf(local_max, val);
  }

  smem[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    //__syncthreads();
  }
  __syncthreads();
  if (tid == 0) {
    block_max[blockIdx.x] = smem[0];
  }
}


torch::Tensor absdiffmax(torch::Tensor a, torch::Tensor b)
{
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors!");
  TORCH_CHECK(a.numel() == b.numel(), "Inputs must have same shape!");

  const int N = a.numel();
  const int threads = 256;
  const int blocks = 128;  // Increased to 128

  auto block_max = torch::empty({blocks}, torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));

  absdiffmax_blocks_kernel<<<blocks, threads>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      block_max.data_ptr<float>(),
      N
  );

  // Final reduction on GPU
  auto final_max = block_max.max();

  return final_max;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("absdiffmax", torch::wrap_pybind_function(absdiffmax), "Element-wise abs(a - b) (float16, CUDA)");
}
