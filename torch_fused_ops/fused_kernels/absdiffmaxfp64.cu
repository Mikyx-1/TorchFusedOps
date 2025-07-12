#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/extension.h>


__global__
void absdiffmax_blocks_kernel(const double* __restrict__ a,
                              const double* __restrict__ b,
                              double* __restrict__ block_max,
                              const int N)
{

  extern __shared__ double smem[];

  int tid = threadIdx.x;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  double local_max = 0.0;

  for (unsigned int i = thread_idx; i < N; i += gridDim.x * blockDim.x){
    double val = fabs(a[i] - b[i]);
    local_max = fmax(val, local_max);
  }
  smem[tid] = local_max;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride){
      smem[tid] = fmax(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    block_max[blockIdx.x] = smem[0];
  }
}



torch::Tensor absdiffmax(torch::Tensor a, torch::Tensor b)
{
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors!");
  TORCH_CHECK(a.numel() == b.numel(), "Inputs must have same shape!");

  const int N = a.numel();
  const int threads = 256;
  const int blocks = 128;
  const int shared_mem_size = threads * sizeof(double);
    
  torch::Tensor block_max = torch::empty({blocks}, torch::TensorOptions().dtype(torch::kDouble).device(a.device()));

  absdiffmax_blocks_kernel<<<blocks, threads, shared_mem_size>>>(
      a.data_ptr<double>(),
      b.data_ptr<double>(),
      block_max.data_ptr<double>(),
      N
      );
  return block_max.max();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("absdiffmax", torch::wrap_pybind_function(absdiffmax), "...");
}
