
#include <torch/types.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>  // for FLT_MAX

/*
 * @brief Warp-level max reduction
 */
__device__ __forceinline__ float warp_reduction_max(float val)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

template <size_t NUM_THREADS>
__device__ float block_reduce_max(const float* __restrict__ batch_ptr,
                                  float* shared_data,
                                  const size_t num_elements_per_batch)
{
    constexpr size_t NUM_WARPS = NUM_THREADS / 32;
    size_t tid = threadIdx.x;

    // Initialize with the lowest possible float value
    float max_val = -FLT_MAX;

    // Each thread processes a chunk of elements
    for (size_t idx = tid; idx < num_elements_per_batch; idx += blockDim.x)
    {
        max_val = fmaxf(max_val, batch_ptr[idx]);
    }

    // Intra-warp max reduction
    max_val = warp_reduction_max(max_val);

    // First thread in each warp writes its result to shared memory
    if (tid % 32 == 0)
        shared_data[tid / 32] = max_val;

    __syncthreads();

    // Let warp 0 reduce all warp results
    float warp_max = -FLT_MAX;
    if (tid < NUM_WARPS)
        warp_max = shared_data[tid];

    return warp_reduction_max(warp_max);
}

template <size_t NUM_THREADS>
__global__ void batched_reduce_max_kernel(const float* input,
                                          float* output,
                                          size_t num_elements_per_batch)
{
    extern __shared__ float shared_data[];
    const float* batch_ptr = input + blockIdx.x * num_elements_per_batch;

    float result = block_reduce_max<NUM_THREADS>(batch_ptr,
                                                 shared_data,
                                                 num_elements_per_batch);

    if (threadIdx.x == 0)
        output[blockIdx.x] = result;
}

torch::Tensor batched_reduce_max(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device!!");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32!!");

    const uint32_t NUM_THREADS = 256;
    const uint32_t NUM_WARPS = NUM_THREADS / 32;
    const size_t batch_size = input.size(0);
    const size_t num_elements_per_batch = input.size(1);
    const dim3 blocks(batch_size);
    const dim3 threads(NUM_THREADS);
    const size_t shared_mem = NUM_WARPS * sizeof(float);

    torch::Tensor output = torch::empty({(long)batch_size, 1}, input.options());

    batched_reduce_max_kernel<NUM_THREADS><<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements_per_batch
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("batched_reduce_max", &batched_reduce_max, "Batched reduce max");
}
