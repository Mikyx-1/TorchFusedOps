/*
 * Batched 2D max reduction (B, N) in float16 with rows-per-thread dispatch.
 *
 * - Fast path for N == 1: returns input.select(1,0).clone()
 * - If N <= ROWS_PER_THREAD_THRESHOLD (default 16): use rows-per-thread kernel
 *   where each CUDA thread processes many rows serially (good for tiny N).
 * - Else: use warp-per-row kernel (each warp reduces one row).
 *
 * Author: adapted for user
 * The code matches Pytorch's performance in most cases.
 * It still lags behind in cases where batch sizes are small and num_elements are large (e.g. shape = (4, 65536*18))
 */

#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <ATen/ATen.h> // for select/clone on tensors
#include <cstdint>
#include <limits>
#include <cuda_fp16.h>

#define MAX_BLOCKS 65535u
// threshold for switching to rows-per-thread kernel (tuneable)
#define ROWS_PER_THREAD_THRESHOLD 16u


// -----------------------------------------------------------------------------
// Rows-per-thread kernel (best when N is small).
// Each thread handles many rows in a strided loop: row = gid, row += stride.
// Accumulates values in float for speed and converts back to __half when writing.
template <unsigned int THREADS>
__global__ void batched_rows_per_thread_kernel( 
    const __half* __restrict__ g_idata,   // (B, N) contiguous row-major
    __half* __restrict__ g_result,        // (B,)
    unsigned long long B, 
    unsigned int N 
) { 
    const unsigned int tid_in_block = threadIdx.x;
    const unsigned int block_threads = blockDim.x;
    const unsigned int gid = blockIdx.x * block_threads + tid_in_block;
    const unsigned int stride = gridDim.x * block_threads;
 
    // Each thread processes rows: row = gid, gid + stride, gid + 2*stride, ...
    for (unsigned long long row = (unsigned long long)gid; row < B; row += stride) {
        const __half* row_ptr = g_idata + row * (unsigned long long)N;
        __half acc = __float2half(-INFINITY); // accumulate in float
 
        // N is small here (<= ROWS_PER_THREAD_THRESHOLD)
        for (unsigned int j = 0; j < N; ++j) 
        { 
          acc = __hmax(acc, row_ptr[j]); 
        } 
 
        g_result[row] = acc; 
    } 
} 
 
// -----------------------------------------------------------------------------
// Warp-per-row kernel (your original approach).
// Each warp reduces one row; blocks contain warps_per_block warps; kernel loops by row_stride.
__device__ __forceinline__ __half warpReduceMaxHalf_shfl(__half val) {
    // reduce __half by pairwise comparing using shuffle (keeps everything in __half)
    #pragma unroll 
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = __hmax(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
    } 
    return val; 
} 


template <unsigned int THREADS> 
__global__ void batched_warp_per_row_kernel( 
    const __half* __restrict__ g_idata,  // (B, N)
    __half* __restrict__ g_result,       // (B,)
    unsigned int B, 
    unsigned int N 
) { 
    const unsigned int warp_id_in_block = (threadIdx.x >> 5);   // warp index
    const unsigned int lane = threadIdx.x & 31;                 // lane within warp
    const unsigned int warps_per_block = (THREADS >> 5);
 
    unsigned int base_row = blockIdx.x * warps_per_block;
    unsigned int row_stride = gridDim.x * warps_per_block; 
 
    for (unsigned int row = base_row + warp_id_in_block; row < B; row += row_stride) {
        const __half* row_ptr = g_idata + static_cast<size_t>(row) * N;
 
        __half myMax = __float2half(-INFINITY);

        // process 2 elements per iteration
        for (unsigned int i = lane; i + 32 < N; i += 64) {
            __half v0 = row_ptr[i];
            __half v1 = row_ptr[i + 32];
            myMax = __hmax(myMax, __hmax(v0, v1));
        }

        // handle leftover elements (if N not multiple of 64)
        for (unsigned int i = (N & ~31u) + lane; i < N; i += 32) {
            myMax = __hmax(myMax, row_ptr[i]);
        }

        // warp reduce
        __half warpMax = warpReduceMaxHalf_shfl(myMax);
        if (lane == 0) g_result[row] = warpMax;
        __syncwarp();
    }
} 

// -----------------------------------------------------------------------------
// Host dispatch wrapper: chooses fast-path / rows-per-thread / warp-per-row
torch::Tensor k6_batched_max_reduction(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (B, N)");
    TORCH_CHECK(input.dtype() == torch::kHalf, "Only float16 tensors are supported");
 
    const int64_t B = input.size(0); 
    const int64_t N = input.size(1); 
 
    if (N == 0) { 
        return torch::full({B}, -std::numeric_limits<float>::infinity(), input.options());
    } 
 
    // FAST PATH for N == 1: no kernel launch, return column clone (very cheap)
    if (N == 1) { 
        // input.select(1,0) returns a view shape (B,), clone to get contiguous memory and correct semantics
        return input.select(1, 0).clone();  
    }
  
    // common launch params 
    const int threads = 512; // must be multiple of 32
    const unsigned int max_blocks = MAX_BLOCKS;
    auto result = torch::empty({B}, input.options());

    if (static_cast<unsigned int>(N) <= ROWS_PER_THREAD_THRESHOLD) {
        // Use rows-per-thread kernel
        // choose a blocks count: at least enough to have one thread per warp across grid,
        // but cap by MAX_BLOCKS. We'll compute blocks so that blocks*threads >= min(B, some_target)
        // Simpler: choose blocks = min(MAX_BLOCKS, ceil(B / threads))
        uint64_t needed_blocks = ( (uint64_t)B + (uint64_t)threads - 1 ) / (uint64_t)threads;
        unsigned int blocks = static_cast<unsigned int>(needed_blocks > max_blocks ? max_blocks : needed_blocks);

        // If B is tiny and needed_blocks becomes 0, ensure at least 1 block
        if (blocks == 0) blocks = 1;

        batched_rows_per_thread_kernel<threads><<<blocks, threads>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(result.data_ptr<at::Half>()),
            static_cast<unsigned long long>(B),
            static_cast<unsigned int>(N)
        ); 
    } else { 
        // Use warp-per-row kernel 
        const unsigned int warps_per_block = threads / 32;
        uint64_t needed_blocks = ( (uint64_t)B + (uint64_t)warps_per_block - 1 ) / (uint64_t)warps_per_block;
        unsigned int blocks = static_cast<unsigned int>(needed_blocks > max_blocks ? max_blocks : needed_blocks);
        if (blocks == 0) blocks = 1; 
 
        batched_warp_per_row_kernel<threads><<<blocks, threads>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(result.data_ptr<at::Half>()),
            static_cast<unsigned int>(B),
            static_cast<unsigned int>(N)
        ); 
    } 
 
    // Optionally check for kernel errors (uncomment to help debugging)
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) { 
    //     TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    // } 
 
    return result; 
} 
 
// -----------------------------------------------------------------------------
// Pybind 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("k6_batched_max_reduction", &k6_batched_max_reduction,
          "Batched reduction with rows-per-thread dispatch (float16 input -> float16 per batch)");
}  
                                        
