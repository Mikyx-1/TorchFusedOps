#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


template <typename scalar_t>
__device__ __forceinline__ scalar_t scalar_abs_diff(scalar_t a, scalar_t b)
{
  scalar_t diff = a - b;
  return (diff >= 0) ? diff : -diff;
}

template <typename scalar_t>
__global__
void absdiffmax_blocks_kernel(const scalar_t* __restrict__ a,
                              const scalar_t* __restrict__ b,
                              scalar_t* __restrict__ block_max,
                              const uint32_t N)
{
  
  extern __shared__ char smem_raw[];
  scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);

  uint32_t tid = threadIdx.x;
  uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  scalar_t local_max = 0;
  
  for (uint32_t i = thread_idx; i < N; i += gridDim.x * blockDim.x)
  {
    scalar_t val = scalar_abs_diff<scalar_t>(a[i], b[i]);
    local_max = max(val, local_max);
  }

  smem[tid] = local_max;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1){
    if (tid < stride){
      smem[tid] = max(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  
  if (tid == 0)
  {
    block_max[blockIdx.x] = smem[tid];
  }
}


torch::Tensor absdiffmax(torch::Tensor a, torch::Tensor b)
{
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors!");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "Inputs mst have same dtype!");
  TORCH_CHECK(a.numel() == b.numel(), "Inputs must have same shape!");

  const int N = a.numel();
  const int threads = 256;
  const int blocks = 128;
  const int shared_mem_size = threads * a.element_size(); // handles int64 too

  torch::Tensor block_max = torch::empty({blocks}, a.options());

  AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "absdiffmax_blocks_kernel", [&] {
      absdiffmax_blocks_kernel<<<blocks, threads, shared_mem_size>>>(
          a.data_ptr<scalar_t>(),
          b.data_ptr<scalar_t>(),
          block_max.data_ptr<scalar_t>(),
          N
          );
      });

  return block_max.max();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("absdiffmax", torch::wrap_pybind_function(absdiffmax), "...");
}
