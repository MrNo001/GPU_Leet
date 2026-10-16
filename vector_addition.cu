// LeetGPU — Vector Addition
// https://leetgpu.com/challenges/vector-addition
//
// Problem: given two float32 vectors A and B of length N, compute C[i] = A[i] + B[i].
// Constraints: 1 <= N <= 100,000,000 (performance measured at N = 25,000,000).
//
// Core idea: every element of C is independent of every other element, so we
// can give each GPU thread exactly one element to compute. With millions of
// threads running in parallel, the whole vector gets added at once instead of
// in a sequential loop like on the CPU.

#include <cuda_runtime.h>

// __global__ marks this as a "kernel": a function that runs on the GPU and is
// launched from the CPU. Every launched thread runs this same code; the only
// thing that differs between threads is their ID (blockIdx / threadIdx), which
// is how each thread figures out which element it is responsible for.
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // Threads are organised into blocks, and blocks into a grid:
    //   blockIdx.x  = which block this thread is in   (0 .. gridDim.x - 1)
    //   blockDim.x  = how many threads are in a block (256, set in solve())
    //   threadIdx.x = this thread's position inside its block (0 .. 255)
    //
    // Multiplying the block index by the block size skips over all the
    // threads in earlier blocks; adding threadIdx.x gives a unique global
    // index for this thread. Example with 256 threads/block:
    //   block 0, thread 5  -> i = 0*256 + 5 = 5
    //   block 2, thread 10 -> i = 2*256 + 10 = 522
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check. We launch threads in whole blocks of 256, so the total
    // number of threads is rounded UP to a multiple of 256. If N = 1000, we
    // get 4 blocks = 1024 threads, and threads 1000..1023 have no element to
    // work on. Without this check they would read/write past the end of the
    // arrays (out-of-bounds memory access -> wrong results or a crash).
    if (i < N) {
        // The actual work: one load from A, one load from B, one add, one
        // store to C. Neighbouring threads (i, i+1, i+2, ...) access
        // neighbouring addresses, so the hardware can combine a warp's 32
        // loads into a few wide memory transactions ("coalesced" access).
        // This matters because the kernel is memory-bound: the add itself is
        // trivial, and the runtime is dominated by moving data to and from
        // GPU memory.
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU).
// The memory is already allocated and filled by the judge, so there is no
// cudaMalloc/cudaMemcpy here: we only launch the kernel.
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    // 256 threads per block is a common, safe default: it's a multiple of the
    // warp size (32), so no warp is left partly empty, and it's small enough
    // that many blocks can be resident on each streaming multiprocessor at
    // once, which helps hide memory latency.
    int threadsPerBlock = 256;

    // We need at least N threads in total. (N + threadsPerBlock - 1) /
    // threadsPerBlock is integer "ceiling division": it rounds up, so a
    // partial final block is still launched. E.g. N = 1000:
    //   (1000 + 255) / 256 = 1255 / 256 = 4 blocks  (plain 1000/256 = 3 would
    //   cover only 768 elements and miss the last 232).
    // The surplus threads in the last block are handled by the bounds check.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel. The <<<grid, block>>> syntax tells CUDA how many
    // blocks to create and how many threads per block.
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    // Kernel launches are asynchronous: the CPU queues the work and moves on
    // immediately. cudaDeviceSynchronize() blocks until the GPU has finished,
    // so C is fully written before solve() returns and the judge checks it.
    cudaDeviceSynchronize();
}
