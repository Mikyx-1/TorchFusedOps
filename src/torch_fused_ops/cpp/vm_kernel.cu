#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

enum OpCode { LOAD = 0, ADD = 1, SUB = 2, MUL = 3 };

__global__ void fused_kernel(const float **__restrict__ inputs,
                             float *__restrict__ out,
                             const int *__restrict__ program, int program_size,
                             int64_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= numel)
    return;

  float stack[32];
  int sp = 0;

  for (int pc = 0; pc < program_size; ++pc) {

    int op = program[pc];

    if (op == LOAD) {
      int tensor_id = program[++pc];
      stack[sp++] = inputs[tensor_id][idx];
    } else if (op == ADD) {
      float b = stack[--sp];
      float a = stack[--sp];
      stack[sp++] = a + b;
    } else if (op == SUB) {
      float b = stack[--sp];
      float a = stack[--sp];
      stack[sp++] = a - b;
    } else if (op == MUL) {
      float b = stack[--sp];
      float a = stack[--sp];
      stack[sp++] = a * b;
    }
  }

  out[idx] = stack[0];
}

torch::Tensor run_fused(std::vector<torch::Tensor> tensors,
                        torch::Tensor program_tensor) {

  TORCH_CHECK(!tensors.empty(), "Need at least 1 tensor");

  int n_inputs = tensors.size();

  for (int i = 0; i < n_inputs; ++i) {
    TORCH_CHECK(tensors[i].is_cuda(), "Must be CUDA");
    TORCH_CHECK(tensors[i].scalar_type() == torch::kFloat32, "float32 only");
    tensors[i] = tensors[i].contiguous();
  }

  int64_t numel = tensors[0].numel();
  auto out = torch::empty_like(tensors[0]);

  // Create host pointer array
  std::vector<const float *> h_ptrs(n_inputs);
  for (int i = 0; i < n_inputs; ++i) {
    h_ptrs[i] = tensors[i].data_ptr<float>();
  }

  // Copy pointer array to device
  const float **d_ptrs;
  cudaMalloc(&d_ptrs, n_inputs * sizeof(float *));
  cudaMemcpy(d_ptrs, h_ptrs.data(), n_inputs * sizeof(float *),
             cudaMemcpyHostToDevice);

  const int threads = 256;
  const int blocks = (numel + threads - 1) / threads;

  fused_kernel<<<blocks, threads>>>(d_ptrs, out.data_ptr<float>(),
                                    program_tensor.data_ptr<int>(),
                                    program_tensor.numel(), numel);

  cudaFree(d_ptrs);

  return out;
}
