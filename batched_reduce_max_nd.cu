
/*

result: tensor([5.6025], device='cuda:0'), pytorch   : 0.001 ± 0.000 ms
result: tensor([5.6025], device='cuda:0'), custom    : 0.011 ± 0.000 ms

 * */
#include <torch/types.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>  // for FLT_MAX

/*
 * @brief Warp-level max reduction
 */
__device__ __forceinline__ float warp_reduction_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

/*
 * @brief Block-level 2D batched max reduction kernel
 * Input: (batch, reduce_size)
 * Output: (batch)
 */
template <int THREADS>
__global__ void batched_reduce_max_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int batch,
                                          int reduce_size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    const float* batch_ptr = input + bid * reduce_size;

    // Parallel reduction in each thread
    float val = -FLT_MAX;
    for (int idx = tid; idx < reduce_size; idx += THREADS) {
        val = fmaxf(val, batch_ptr[idx]);
    }
    // Warp reduction
    val = warp_reduction_max(val);
    if ((tid & 31) == 0) sdata[tid >> 5] = val;
    __syncthreads();
    // Final reduce by first warp
    val = (tid < (THREADS/32)) ? sdata[tid] : -FLT_MAX;
    val = warp_reduction_max(val);

    if (tid == 0) output[bid] = val;
}

/*
 * @brief N-D reduce-max wrapper that flattens to 2D
 */
torch::Tensor reduce_max_nd(torch::Tensor input,
                             std::vector<int64_t> dims = {},
                             bool keepdim = false) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device!");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32!");
    
    int ndim = input.dim();
    if (dims.empty()) {
        // reduce all dims => treat as 1D
        dims.resize(ndim);
        std::iota(dims.begin(), dims.end(), 0);
    }
    // normalize dims
    for (auto &d : dims) if (d < 0) d += ndim;
    std::sort(dims.begin(), dims.end());
    dims.erase(std::unique(dims.begin(), dims.end()), dims.end());

    // Build permute order: non-reduce dims first, then reduce dims
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int i = 0; i < ndim; ++i)
        if (std::find(dims.begin(), dims.end(), i) == dims.end()) perm.push_back(i);
    for (auto d : dims) perm.push_back(d);

    auto permuted = input.permute(perm).contiguous();
    // compute sizes
    int batch = 1;
    for (int i = 0; i < (ndim - (int)dims.size()); ++i) batch *= permuted.size(i);
    int reduce_size = 1;
    for (int i = ndim - dims.size(); i < ndim; ++i) reduce_size *= permuted.size(i);

    auto out = torch::empty({batch}, input.options());
    const int THREADS = 256;
    const int WARPS = THREADS / 32;
    const size_t shared_mem = WARPS * sizeof(float);
    
    dim3 grid(batch);
    dim3 block(THREADS);
    batched_reduce_max_kernel<THREADS><<<grid, block, shared_mem>>>(
        permuted.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        reduce_size);
    cudaDeviceSynchronize();

    // reshape back
    std::vector<int64_t> out_shape;
    if (keepdim) {
        for (int i = 0, ni=0; i < ndim; ++i) {
            if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
                out_shape.push_back(input.size(i));
                ++ni;
            } else {
                out_shape.push_back(1);
            }
        }
    } else {
        for (int i = 0; i < ndim; ++i)
            if (std::find(dims.begin(), dims.end(), i) == dims.end())
                out_shape.push_back(input.size(i));
    }
    return out.reshape(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_max_nd", &reduce_max_nd,
          "N-dimensional reduce max (optimized flatten+batched kernel)",
          py::arg("input"),
          py::arg("dims") = std::vector<int64_t>(),
          py::arg("keepdim") = false);
}
