#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "symbolic.h"
#include <cmath>

using namespace std;


__global__ void RL(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const int* __restrict__ level_idx_dev,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = threadIdx.x / 32;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    extern __shared__ REAL s[];

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
    __syncthreads();

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
    const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
    unsigned subCol;
    const int tidInWarp = threadIdx.x % 32;
    unsigned subColElem = 0;

    int woffset = 0;
    while (subMatSize > woffset)
    {
        if (wid + woffset < subMatSize)
        {
            offset = 0;
            subCol = csr_c_idx_dev[subColPos + woffset];
            while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
            {
                if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
                {

                    subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
                    unsigned ridx = sym_r_idx_dev[subColElem];

                    if (ridx == currentCol)
                    {
                        s[wid] = val_dev[subColElem];
                    }
                    //Threads in a warp are always synchronized
                    //__syncthreads();
                    if (ridx > currentCol)
                    {
                        //elem in currentCol same row with subColElem might be 0, so
                        //clearing tmpMem is necessary
                        atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
                    }
                }
                offset += 32;
            }
        }
        woffset += blockDim.x/32;
    }

    __syncthreads();
    //Clear tmpMem
    offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[bid*n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}

__global__ void RL_perturb(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const int* __restrict__ level_idx_dev,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos,
        const float pert)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = threadIdx.x / 32;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    extern __shared__ REAL s[];

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            if (fabs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
                val_dev[l_col_ptr_dev[currentCol]] = pert;

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
    __syncthreads();

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
    const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
    unsigned subCol;
    const int tidInWarp = threadIdx.x % 32;
    unsigned subColElem = 0;

    int woffset = 0;
    while (subMatSize > woffset)
    {
        if (wid + woffset < subMatSize)
        {
            offset = 0;
            subCol = csr_c_idx_dev[subColPos + woffset];
            while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
            {
                if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
                {

                    subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
                    unsigned ridx = sym_r_idx_dev[subColElem];

                    if (ridx == currentCol)
                    {
                        s[wid] = val_dev[subColElem];
                    }
                    //Threads in a warp are always synchronized
                    //__syncthreads();
                    if (ridx > currentCol)
                    {
                        //elem in currentCol same row with subColElem might be 0, so
                        //clearing tmpMem is necessary
                        atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
                    }
                }
                offset += 32;
            }
        }
        woffset += blockDim.x/32;
    }

    __syncthreads();
    //Clear tmpMem
    offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[bid*n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_factorizeCurrentCol(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_factorizeCurrentCol_perturb(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n,
        const float pert)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            if (fabs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
                val_dev[l_col_ptr_dev[currentCol]] = pert;

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_updateSubmat(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ REAL s;

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + bid + 1;
    unsigned subCol;
    unsigned subColElem = 0;

    int offset = 0;
    subCol = csr_c_idx_dev[subColPos];
    if (tid == 0)
        s = 0;
    __syncthreads();
    while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
    {
        bool active = (tid + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol]);
        unsigned ridx = 0;
        if (active)
        {
            subColElem = sym_c_ptr_dev[subCol] + tid + offset;
            ridx = sym_r_idx_dev[subColElem];

            if (ridx == currentCol)
            {
                s = val_dev[subColElem];
            }
        }
        // Every thread must reach both barriers; gating __syncthreads by "active"
        // can deadlock when sub-column nnz is smaller than blockDim.x.
        __syncthreads();
        if (active && ridx > currentCol)
        {
            atomicAdd(&val_dev[subColElem], -tmpMem[stream * n + ridx] * s);
        }
        __syncthreads();
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_cleartmpMem(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    unsigned offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[stream * n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}

void LUonDevice(Symbolic_Matrix &A_sym, ostream &out, ostream &err, bool PERTURB)
{
    if (A_sym.n == 0 || A_sym.nnz == 0) {
        err << "Matrix is empty; skipping GPU factorization." << endl;
        return;
    }

    auto cudaCheck = [&](cudaError_t status, const char *op) -> bool {
        if (status != cudaSuccess) {
            err << op << " failed: " << cudaGetErrorString(status) << endl;
            return false;
        }
        return true;
    };

    unsigned n = A_sym.n;
    unsigned nnz = A_sym.nnz;
    unsigned num_lev = A_sym.num_lev;

    int deviceCount = 0;
    if (!cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount"))
        return;
    if (deviceCount <= 0) {
        err << "No CUDA-capable GPU detected." << endl;
        return;
    }

    int dev = 0;
    cudaDeviceProp deviceProp;
    if (!cudaCheck(cudaGetDeviceProperties(&deviceProp, dev), "cudaGetDeviceProperties"))
        return;
    if (!cudaCheck(cudaSetDevice(dev), "cudaSetDevice"))
        return;
    out << "Device " << dev << ": " << deviceProp.name << " has been selected." << endl;

    cudaEvent_t start = nullptr, stop = nullptr;
    unsigned *sym_c_ptr_dev = nullptr, *sym_r_idx_dev = nullptr, *l_col_ptr_dev = nullptr;
    REAL *val_dev = nullptr;
    unsigned *csr_r_ptr_dev = nullptr, *csr_c_idx_dev = nullptr, *csr_diag_ptr_dev = nullptr;
    int *level_idx_dev = nullptr;
    REAL *tmpMem = nullptr;
    float time = 0.0f;

    constexpr int Nstreams = 16;
    cudaStream_t streams[Nstreams];
    bool stream_created[Nstreams] = {false};

    auto cleanup = [&]() {
        for (int j = 0; j < Nstreams; ++j) {
            if (stream_created[j])
                cudaStreamDestroy(streams[j]);
        }
        if (tmpMem != nullptr)
            cudaFree(tmpMem);
        if (sym_c_ptr_dev != nullptr)
            cudaFree(sym_c_ptr_dev);
        if (sym_r_idx_dev != nullptr)
            cudaFree(sym_r_idx_dev);
        if (val_dev != nullptr)
            cudaFree(val_dev);
        if (l_col_ptr_dev != nullptr)
            cudaFree(l_col_ptr_dev);
        if (csr_c_idx_dev != nullptr)
            cudaFree(csr_c_idx_dev);
        if (csr_r_ptr_dev != nullptr)
            cudaFree(csr_r_ptr_dev);
        if (csr_diag_ptr_dev != nullptr)
            cudaFree(csr_diag_ptr_dev);
        if (level_idx_dev != nullptr)
            cudaFree(level_idx_dev);
        if (start != nullptr)
            cudaEventDestroy(start);
        if (stop != nullptr)
            cudaEventDestroy(stop);
    };

#define CUDA_RETURN_ON_ERR(call, op_name) \
    do { \
        if (!cudaCheck((call), (op_name))) { \
            cleanup(); \
            return; \
        } \
    } while (0)

    CUDA_RETURN_ON_ERR(cudaEventCreate(&start), "cudaEventCreate(start)");
    CUDA_RETURN_ON_ERR(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    CUDA_RETURN_ON_ERR(cudaEventRecord(start, 0), "cudaEventRecord(start)");

    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&sym_c_ptr_dev, (n + 1) * sizeof(unsigned)), "cudaMalloc(sym_c_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&sym_r_idx_dev, nnz * sizeof(unsigned)), "cudaMalloc(sym_r_idx_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&val_dev, nnz * sizeof(REAL)), "cudaMalloc(val_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&l_col_ptr_dev, n * sizeof(unsigned)), "cudaMalloc(l_col_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&csr_r_ptr_dev, (n + 1) * sizeof(unsigned)), "cudaMalloc(csr_r_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&csr_c_idx_dev, nnz * sizeof(unsigned)), "cudaMalloc(csr_c_idx_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&csr_diag_ptr_dev, n * sizeof(unsigned)), "cudaMalloc(csr_diag_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&level_idx_dev, n * sizeof(int)), "cudaMalloc(level_idx_dev)");

    CUDA_RETURN_ON_ERR(cudaMemcpy(sym_c_ptr_dev, &(A_sym.sym_c_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(sym_c_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(sym_r_idx_dev, &(A_sym.sym_r_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(sym_r_idx_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(val_dev, &(A_sym.val[0]), nnz * sizeof(REAL), cudaMemcpyHostToDevice), "cudaMemcpy(val_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(l_col_ptr_dev, &(A_sym.l_col_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(l_col_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(csr_r_ptr_dev, &(A_sym.csr_r_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(csr_r_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(csr_c_idx_dev, &(A_sym.csr_c_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(csr_c_idx_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(csr_diag_ptr_dev, &(A_sym.csr_diag_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice),
        "cudaMemcpy(csr_diag_ptr_dev)");
    CUDA_RETURN_ON_ERR(cudaMemcpy(level_idx_dev, &(A_sym.level_idx[0]), n * sizeof(int), cudaMemcpyHostToDevice),
        "cudaMemcpy(level_idx_dev)");

    for (int j = 0; j < Nstreams; ++j) {
        CUDA_RETURN_ON_ERR(cudaStreamCreate(&streams[j]), "cudaStreamCreate");
        stream_created[j] = true;
    }

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_RETURN_ON_ERR(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    (void)total_bytes;

    const size_t reserve_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    size_t max_tmp_mem_size = (free_bytes > reserve_bytes) ? (free_bytes - reserve_bytes) : (free_bytes / 2);
    const unsigned first_level_cols = (A_sym.level_ptr.size() > 1) ?
        static_cast<unsigned>(A_sym.level_ptr[1] - A_sym.level_ptr[0]) : 1u;
    const size_t good_tmp_mem_choice = sizeof(REAL) * size_t(n) * size_t(first_level_cols);
    const size_t bytes_per_column = sizeof(REAL) * size_t(n);
    if (max_tmp_mem_size < bytes_per_column)
        max_tmp_mem_size = bytes_per_column;

    unsigned TMPMEMNUM = 0;
    if (good_tmp_mem_choice <= max_tmp_mem_size)
        TMPMEMNUM = first_level_cols;
    else
        TMPMEMNUM = static_cast<unsigned>(max_tmp_mem_size / bytes_per_column);
    if (TMPMEMNUM == 0)
        TMPMEMNUM = 1;

    const size_t tmp_bytes = size_t(TMPMEMNUM) * size_t(n) * sizeof(REAL);
    CUDA_RETURN_ON_ERR(cudaMalloc((void**)&tmpMem, tmp_bytes), "cudaMalloc(tmpMem)");
    CUDA_RETURN_ON_ERR(cudaMemset(tmpMem, 0, tmp_bytes), "cudaMemset(tmpMem)");

    // calculate 1-norm of A and perturbation value for perturbation
    float pert = 0;
    if (PERTURB)
    {
        float norm_A = 0;
        for (unsigned i = 0; i < n; ++i)
        {
            float tmp = 0;
            for (unsigned j = A_sym.sym_c_ptr[i]; j < A_sym.sym_c_ptr[i+1]; ++j)
                tmp += fabs(A_sym.val[j]);
            if (norm_A < tmp)
                norm_A = tmp;
        }
        pert = 3.45e-4f * norm_A;
        out << "Gaussian elimination with static pivoting (GESP)..." << endl;
        out << "1-Norm of A matrix is " << norm_A << ", Perturbation value is " << pert << endl;
    }

    auto launch_batched_level = [&](unsigned level_head, int level_size, unsigned warps_per_block) {
        dim3 dimBlock(warps_per_block * 32, 1);
        size_t mem_size = warps_per_block * sizeof(REAL);

        int remaining = level_size;
        unsigned chunk_idx = 0;
        while (remaining > 0) {
            unsigned rest_col = static_cast<unsigned>(remaining) > TMPMEMNUM ?
                TMPMEMNUM : static_cast<unsigned>(remaining);
            dim3 dimGrid(rest_col, 1);
            int in_level_pos = static_cast<int>(chunk_idx * TMPMEMNUM);
            if (!PERTURB)
                RL<<<dimGrid, dimBlock, mem_size>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    val_dev,
                                                    l_col_ptr_dev,
                                                    csr_r_ptr_dev,
                                                    csr_c_idx_dev,
                                                    csr_diag_ptr_dev,
                                                    level_idx_dev,
                                                    tmpMem,
                                                    n,
                                                    level_head,
                                                    in_level_pos);
            else
                RL_perturb<<<dimGrid, dimBlock, mem_size>>>(sym_c_ptr_dev,
                                                            sym_r_idx_dev,
                                                            val_dev,
                                                            l_col_ptr_dev,
                                                            csr_r_ptr_dev,
                                                            csr_c_idx_dev,
                                                            csr_diag_ptr_dev,
                                                            level_idx_dev,
                                                            tmpMem,
                                                            n,
                                                            level_head,
                                                            in_level_pos,
                                                            pert);
            remaining -= static_cast<int>(rest_col);
            ++chunk_idx;
        }
    };

    for (unsigned i = 0; i < num_lev; ++i)
    {
        int lev_size = A_sym.level_ptr[i + 1] - A_sym.level_ptr[i];
        if (lev_size <= 0)
            continue;

        if (lev_size > 896) {
            launch_batched_level(A_sym.level_ptr[i], lev_size, 2);
        }
        else if (lev_size > 448) {
            launch_batched_level(A_sym.level_ptr[i], lev_size, 4);
        }
        else if (lev_size > Nstreams) {
            launch_batched_level(A_sym.level_ptr[i], lev_size, 32);
        }
        else {
            // Small levels are mapped to one stream per column to reduce launch overhead.
            for (int offset = 0; offset < lev_size; offset += Nstreams) {
                for (int j = 0; j < Nstreams; j++) {
                    if (j + offset < lev_size) {
                        const unsigned currentCol = A_sym.level_idx[A_sym.level_ptr[i] + j + offset];
                        const unsigned subMatSize = A_sym.csr_r_ptr[currentCol + 1]
                            - A_sym.csr_diag_ptr[currentCol] - 1;

                        if (!PERTURB)
                            RL_onecol_factorizeCurrentCol<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                                                        sym_r_idx_dev,
                                                                                        val_dev,
                                                                                        l_col_ptr_dev,
                                                                                        currentCol,
                                                                                        tmpMem,
                                                                                        j,
                                                                                        n);
                        else
                            RL_onecol_factorizeCurrentCol_perturb<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                                                                sym_r_idx_dev,
                                                                                                val_dev,
                                                                                                l_col_ptr_dev,
                                                                                                currentCol,
                                                                                                tmpMem,
                                                                                                j,
                                                                                                n,
                                                                                                pert);
                        if (subMatSize > 0)
                            RL_onecol_updateSubmat<<<subMatSize, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                                                          sym_r_idx_dev,
                                                                                          val_dev,
                                                                                          csr_c_idx_dev,
                                                                                          csr_diag_ptr_dev,
                                                                                          currentCol,
                                                                                          tmpMem,
                                                                                          j,
                                                                                          n);
                        RL_onecol_cleartmpMem<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                                           sym_r_idx_dev,
                                                                           l_col_ptr_dev,
                                                                           currentCol,
                                                                           tmpMem,
                                                                           j,
                                                                           n);
                    }
                }
            }
        }
        CUDA_RETURN_ON_ERR(cudaGetLastError(), "kernel launch");
        CUDA_RETURN_ON_ERR(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    CUDA_RETURN_ON_ERR(cudaMemcpy(&(A_sym.val[0]), val_dev, nnz * sizeof(REAL), cudaMemcpyDeviceToHost),
        "cudaMemcpy(A_sym.val)");
    CUDA_RETURN_ON_ERR(cudaEventRecord(stop, 0), "cudaEventRecord(stop)");
    CUDA_RETURN_ON_ERR(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    CUDA_RETURN_ON_ERR(cudaEventElapsedTime(&time, start, stop), "cudaEventElapsedTime");

    out << "Total GPU time: " << time << " ms" << endl;

#ifdef GLU_DEBUG
    // check NaN elements
    unsigned err_find = 0;
    for(unsigned i = 0; i < nnz; i++)
        if(isnan(A_sym.val[i]) || isinf(A_sym.val[i]))
            err_find++;

    if (err_find != 0)
        err << "LU data check: NaN/Inf found." << endl;
#endif
    cleanup();
#undef CUDA_RETURN_ON_ERR
}
