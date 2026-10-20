#ifndef HPRQP_PACKED_SPMV_CUH
#define HPRQP_PACKED_SPMV_CUH
#include "gpu/types.h"
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <stdexcept>

// Column-major ELL storage preserves each CSR row's entry order. Only the
// ordinary interval projection update uses this optional access path.
// Refresh from GPU CSR after scaling; CPU input values are no longer valid.

static void packed_spmv_free(HPRQPGPUELL *e) {
    cudaFree(e->columns); cudaFree(e->values); *e = HPRQPGPUELL{};
}
static unsigned packed_spmv_width(const HPRQPMatrixCSR *a) {
    if (!a->nrow || !a->nnz) return 0;
    size_t k=0;
    for(size_t i=0;i<a->nrow;++i) k=std::max(k,a->row_ptr[i+1]-a->row_ptr[i]);
    if(k>32 || (long double)k*a->nrow > 1.5L*a->nnz) return 0;
    return (unsigned)k;
}
static bool packed_spmv_pack(HPRQPGPUELL *e,const HPRQPMatrixCSR *a,unsigned width) {
    if (!width || a->nrow > SIZE_MAX / width / sizeof(double)) return false;
    size_t count=a->nrow*width;
    try {
        std::vector<int32_t> cols(count,-1); std::vector<double> vals(count,0.0);
        for(size_t i=0;i<a->nrow;++i) for(size_t j=a->row_ptr[i];j<a->row_ptr[i+1];++j) {
            size_t p=(j-a->row_ptr[i])*a->nrow+i;
            cols[p]=(int32_t)a->col_ind[j]; vals[p]=a->values[j];
        }
        if(cudaMalloc((void**)&e->columns,count*sizeof(int32_t))!=cudaSuccess || cudaMalloc((void**)&e->values,count*sizeof(double))!=cudaSuccess ||
           cudaMemcpy(e->columns,cols.data(),count*sizeof(int32_t),cudaMemcpyHostToDevice)!=cudaSuccess ||
           cudaMemcpy(e->values,vals.data(),count*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess) {
            packed_spmv_free(e); cudaGetLastError(); return false;
        }
        e->stride=a->nrow;e->width=width;return true;
    } catch(const std::bad_alloc&) { packed_spmv_free(e); return false; }
      catch(const std::length_error&) { packed_spmv_free(e); return false; }
}
static void packed_spmv_prepare_pair(HPRQPGPUProblem *g,const HPRQPModel *m) {
    auto start=std::chrono::steady_clock::now();unsigned ka=packed_spmv_width(&m->A),kt=packed_spmv_width(&m->AT);
    if(!ka||!kt) {std::fprintf(stderr,"PACKED_SPMV_SKIP row_shape widthA=%u widthAT=%u\n",ka,kt);return;}
    size_t free_b=0,total_b=0;
    if(cudaMemGetInfo(&free_b,&total_b)!=cudaSuccess) {
        cudaGetLastError(); std::fprintf(stderr,"PACKED_SPMV_SKIP memory_query\n"); return;
    }
    long double need=12.L*((long double)ka*m->A.nrow+(long double)kt*m->AT.nrow);
    if(need>free_b/2.L) {std::fprintf(stderr,"PACKED_SPMV_SKIP memory\n");return;}
    if(!packed_spmv_pack(&g->ell_A,&m->A,ka)||!packed_spmv_pack(&g->ell_AT,&m->AT,kt)) {
        packed_spmv_free(&g->ell_A);packed_spmv_free(&g->ell_AT);std::fprintf(stderr,"PACKED_SPMV_SKIP allocation\n");return;
    }
    double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::fprintf(stderr,"PACKED_SPMV_READY widthA=%u widthAT=%u bytes=%.0Lf setup_s=%.9f\n",ka,kt,need,sec);
}
static __device__ __forceinline__ double packed_spmv_dot(HPRQPGPUELL e,const double *x,size_t row) {
    double sum=0.0;
    for(unsigned k=0;k<e.width;++k) {
        size_t p=(size_t)k*e.stride+row;int j=e.columns[p];
        if(j>=0) sum=__fma_rn(e.values[p],x[j],sum);
    }
    return sum;
}
__global__ static void packed_spmv_refresh_kernel(HPRQPGPUELL e,HPRQPGPUMatrixCSR a) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=e.stride)return;
    int begin=a.row_ptr[i],end=a.row_ptr[i+1];
    for(unsigned k=0;k<e.width;k++) {
        size_t p=(size_t)k*e.stride+i;
        e.values[p]=(begin+(int)k<end)?a.values[begin+k]:0.0;
    }
}
static HPRQPStatus packed_spmv_refresh(HPRQPGPUProblem *g) {
    if(g->ell_A.width)packed_spmv_refresh_kernel<<<(g->m+127)/128,128>>>(g->ell_A,g->A);
    if(g->ell_AT.width)packed_spmv_refresh_kernel<<<(g->n+127)/128,128>>>(g->ell_AT,g->AT);
    return cudaGetLastError()==cudaSuccess?HPRQP_STATUS_OK:HPRQP_STATUS_NOT_IMPLEMENTED;
}
#endif
