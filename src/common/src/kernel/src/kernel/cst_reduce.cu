#include "fixed_math.cuh"
#include <cuda_runtime.h>

namespace tl {
namespace kernel {

    // THE DETERMINISTIC REDUCTION KERNEL
    // Input: Array of fixed_t forces/energies
    // Output: Single fixed_t sum
    // Constraint: Must return exact same bits on A100, H100, RTX 4090.
    
    __global__ void cst_reduce(const fixed_t* __restrict__ input, 
                               fixed_t* __restrict__ output, 
                               int n) {
        
        extern __shared__ fixed_t sdata[]; // Size determined at launch

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        // 1. DETERMINISTIC LOAD
        // No race conditions here, standard parallel load.
        sdata[tid] = (i < n) ? input[i] : 0;
        __syncthreads();

        // 2. CANONICAL SUMMATION TREE
        // Iterate through strides: 128 -> 64 -> 32 -> ...
        // This imposes a rigid addition topology.
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                // The order (a + b) is strictly enforced by thread ID
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads(); // HARD BARRIER
        }

        // 3. WRITE RESULT
        if (tid == 0) {
            output[blockIdx.x] = sdata[0];
        }
    }

    // Host wrapper to launch kernel
    void launch_cst_reduce(const fixed_t* d_in, fixed_t* d_out, int n, int threads, int blocks) {
        size_t smem_size = threads * sizeof(fixed_t);
        cst_reduce<<<blocks, threads, smem_size>>>(d_in, d_out, n);
        cudaDeviceSynchronize();
    }

} // namespace kernel
} // namespace tl
