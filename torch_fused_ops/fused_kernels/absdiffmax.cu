#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/extension.h>


__global__
void absdiffmax_kernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ max_out,
                              const int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N)
  {
    float val = fabsf(a[idx] - b[idx]);
    atomicMax((int*)max_out, __float_as_int(val));   // Still needs atomicMax
  }
}



torch::Tensor absdiffmax(torch::Tensor a, torch::Tensor b)
{

  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors!");
  TORCH_CHECK(a.numel() == b.numel(), "Inputs must have same shape!");

  const int N = a.numel();
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  
  torch::Tensor max_out = torch::zeros({}, torch::TensorOptions().device(a.device()).dtype(torch::kFloat32));

  absdiffmax_kernel<<<blocks, threads>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      max_out.data_ptr<float>(),
      N
      );

  return max_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("absdiffmax", torch::wrap_pybind_function(absdiffmax), "Compute max(abs(a-b)) (CUDA)");
}
