#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/*
 * @brief AbsDiffMax kernel for float16 (~torch.abs(a-b).max())
 *
 * */
__global__
void absdiffmax_blocks_kernel_fp16(const __half* __restrict__ a,
                              const __half* __restrict__ b,
                              __half* __restrict__ block_max,
                              const int N)
{

  extern __shared__ __half smemfp16[];

  int tid = threadIdx.x;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  __half local_max = __float2half(0.0f);

  for (int i = thread_idx; i < N; i += blockDim.x * gridDim.x)
  {
    __half val = __habs(__hsub(a[i], b[i]));
    local_max = __hmax(local_max, val);
  }
  smemfp16[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      smemfp16[tid] = __hmax(smemfp16[tid], smemfp16[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    block_max[blockIdx.x] = smemfp16[0];
  }
}


__device__ __forceinline__ __nv_bfloat16 bf16_absdiff(const __nv_bfloat16 a, const __nv_bfloat16 b){
  return __habs(__hsub(a, b));  // abs(a - b)
}

__device__ __forceinline__ nv_bfloat16 bf16_max(const __nv_bfloat16 a, const __nv_bfloat16 b){
#if __CUDA_ARCH__ >= 800
  return __hmax(a, b);
#else
  return (__hlt(a, b) ? b : a); // Manual fallback for older GPUs
#endif
}

/*
 * @brief AbsDiffMax kernel for Bfloat16
 * */
__global__
void absdiffmax_blocks_kernel_bf16(const __nv_bfloat16* __restrict__ a,
                                   const __nv_bfloat16* __restrict__ b,
                                   __nv_bfloat16* __restrict__ block_max,
                                   const int N)
{
  extern __shared__ __nv_bfloat16 smem_bf16[];

  int tid = threadIdx.x;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  __nv_bfloat16 local_max = __float2bfloat16(0.0f);

  for (int i = thread_idx; i < N; i += blockDim.x * gridDim.x)
  {
    __nv_bfloat16 val = bf16_absdiff(a[i], b[i]);
    local_max = bf16_max(val, local_max);
  }
  smem_bf16[tid] = local_max;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1){
    smem_bf16[tid] = bf16_max(smem_bf16[tid], smem_bf16[tid + stride]);
    __syncthreads();
  }

  if (tid == 0)
  {
    block_max[blockIdx.x] = smem_bf16[0];
  }
}

/*
 * @brief AbsDiffMax kernel for float32
 * */
__global__
void absdiffmax_blocks_kernel_fp32(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ block_max,
                                   const int N)
{
  extern __shared__ float smem_fp32[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float local_max = 0.0f;

  for (uint32_t i = idx; i < N; i += blockDim.x * gridDim.x)
  {
    float val = fabsf(a[i] - b[i]);
    local_max = fmaxf(val, local_max);
  }
  smem_fp32[tid] = local_max;
  __syncthreads();

  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      smem_fp32[tid] = fmaxf(smem_fp32[tid], smem_fp32[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    block_max[blockIdx.x] = smem_fp32[0];
  }
}

/*
 * @brief AbsDiffMax kernel for float64
 * */
__global__
void absdiffmax_blocks_kernel_fp64(const double* __restrict__ a,
                              const double* __restrict__ b,
                              double* __restrict__ block_max,
                              const int N)
{

  extern __shared__ double smem_fp64[];

  int tid = threadIdx.x;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  double local_max = 0.0;

  for (unsigned int i = thread_idx; i < N; i += gridDim.x * blockDim.x){
    double val = fabs(a[i] - b[i]);
    local_max = fmax(val, local_max);
  }
  smem_fp64[tid] = local_max;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride){
      smem_fp64[tid] = fmax(smem_fp64[tid], smem_fp64[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    block_max[blockIdx.x] = smem_fp64[0];
  }
}

template <typename int_type>
__device__ __forceinline__ int_type int_abs_diff(int_type a, int_type b)
{
  int_type diff = a - b;
  return (diff > 0) ? diff : -diff;
}

template <typename int_type>
__global__
void absdiffmax_blocks_kernel_int(const int_type* __restrict__ a,
                                  const int_type* __restrict__ b,
                                  int_type* __restrict__ block_max,
                                  const int N)
{
  extern __shared__ char smem_raw[];
  int_type* smem_int = reinterpret_cast<int_type*>(smem_raw);


  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int_type local_max = 0;

  for (uint32_t i = idx; i < N; i += blockDim.x * gridDim.x)
  {
    int_type val = int_abs_diff<int_type>(a[i], b[i]);
    local_max = max(val, local_max);
  }
  smem_int[tid] = local_max;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      smem_int[tid] = max(smem_int[tid], smem_int[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    block_max[blockIdx.x] = smem_int[0];
  }
}

torch::Tensor absdiffmax(torch::Tensor a, torch::Tensor b)
{
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(a.numel() == b.numel(), "Inputs must have same shape!");

  int device_id = a.device().index();
  int current_device;
  cudaGetDevice(&current_device);

  if (current_device != device_id){
    cudaSetDevice(device_id);
  }

  auto dtype = a.scalar_type();  // Use scalar_type() instead of dtype()
  
  const int N = a.numel();
  const int threads = 256;
  const int blocks = 128;
  const int shared_mem_size = threads * a.element_size();
  torch::Tensor block_max = torch::empty({blocks}, a.options());

  if (dtype == torch::kFloat16){
    absdiffmax_blocks_kernel_fp16<<<blocks, threads, shared_mem_size>>>(
        reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(block_max.data_ptr<at::Half>()),
        N
        );
  }
  else if (dtype == torch::kFloat32){
    absdiffmax_blocks_kernel_fp32<<<blocks, threads, shared_mem_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        block_max.data_ptr<float>(),
        N
        );
  }
  else if (dtype == torch::kBFloat16){
    absdiffmax_blocks_kernel_bf16<<<blocks, threads, shared_mem_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(block_max.data_ptr<at::BFloat16>()),
        N
        );
  }
  else if (dtype == torch::kFloat64){
    absdiffmax_blocks_kernel_fp64<<<blocks, threads, shared_mem_size>>>(
        a.data_ptr<double>(),
        b.data_ptr<double>(),
        block_max.data_ptr<double>(),
        N
        );
  }
  else if (torch::isIntegralType(dtype, /*includeBool=*/false)){
    AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), "absdiffmax_blocks_kernel_int", [&] {
        absdiffmax_blocks_kernel_int<<<blocks, threads, shared_mem_size>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            block_max.data_ptr<scalar_t>(),
            N
            );
        });
  }
  else{
    TORCH_CHECK(false, "Unsupported tensor dtype: ", dtype);
  }

  // Restore the original device if we changed it
  if (current_device != device_id){
    cudaSetDevice(current_device);
  }
  return block_max.max();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("absdiffmax", torch::wrap_pybind_function(absdiffmax), "...");
}
