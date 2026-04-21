#ifndef HPRQP_GPU_TYPES_H
#define HPRQP_GPU_TYPES_H

#include <stddef.h>
#include <stdint.h>

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HPRQPGPUContext {
    int requested_device;
    int initialized;
} HPRQPGPUContext;

typedef struct HPRQPGPUVector {
    size_t length;
    double *values;
    int allocated;
} HPRQPGPUVector;

typedef int32_t HPRQPGPUCsrIndex;

typedef struct HPRQPGPUSpMVA {
    void *handle;
    void *desc_A;
    void *desc_x_bar;
    void *desc_x_hat;
    void *desc_dx;
    void *desc_tempv;
    void *desc_Ax;
    void *buffer;
    size_t buffer_size;
    int prepared;
} HPRQPGPUSpMVA;

typedef struct HPRQPGPUSpMVAT {
    void *handle;
    void *desc_AT;
    void *desc_y_bar;
    void *desc_y;
    void *desc_ATy_bar;
    void *desc_ATy;
    void *desc_ATdy;
    void *buffer;
    size_t buffer_size;
    int prepared;
} HPRQPGPUSpMVAT;

typedef struct HPRQPGPUSpMVQ {
    void *handle;
    void *desc_Q;
    void *desc_w;
    void *desc_w_bar;
    void *desc_x_bar;
    void *desc_Qw;
    void *desc_Qw_bar;
    void *desc_Qx;
    void *buffer;
    size_t buffer_size;
    int prepared;
} HPRQPGPUSpMVQ;

typedef struct HPRQPGPUMatrixCSR {
    size_t nrow;
    size_t ncol;
    size_t nnz;
    HPRQPGPUCsrIndex *row_ptr;
    HPRQPGPUCsrIndex *col_ind;
    double *values;
    int uploaded;
} HPRQPGPUMatrixCSR;

typedef struct HPRQPGPUProblem {
    size_t m;
    size_t n;
    size_t nnz_q;
    size_t nnz_a;
    int empty_A;
    double obj_constant;
    HPRQPGPUMatrixCSR Q;
    HPRQPGPUMatrixCSR A;
    HPRQPGPUMatrixCSR AT;
    HPRQPGPUVector c;
    HPRQPGPUVector AL;
    HPRQPGPUVector AU;
    HPRQPGPUVector l;
    HPRQPGPUVector u;
    int uploaded;
} HPRQPGPUProblem;

typedef struct HPRQPGPUWorkspace {
    size_t m;
    size_t n;
    int empty_A;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;
    HPRQPSpmvMode spmv_mode_q;
    HPRQPSpmvMode spmv_mode_a;
    double sigma;
    double lambda_max_A;
    double lambda_max_Q;
    int Q_is_diag;
    int noC;
    int to_check;
    double scale_b_scale;
    double scale_c_scale;
    double scale_norm_b_org;
    double scale_norm_c_org;
    HPRQPGPUVector x;
    HPRQPGPUVector x_bar;
    HPRQPGPUVector x_hat;
    HPRQPGPUVector z_bar;
    HPRQPGPUVector dx;
    HPRQPGPUVector w;
    HPRQPGPUVector w_bar;
    HPRQPGPUVector dw;
    HPRQPGPUVector Qw;
    HPRQPGPUVector Qw_bar;
    HPRQPGPUVector Qx;
    HPRQPGPUVector y;
    HPRQPGPUVector y_bar;
    HPRQPGPUVector dy;
    HPRQPGPUVector s;
    HPRQPGPUVector Rp;
    HPRQPGPUVector Rd;
    HPRQPGPUVector Ax;
    HPRQPGPUVector ATy;
    HPRQPGPUVector ATy_bar;
    HPRQPGPUVector ATdy;
    HPRQPGPUVector dQw;
    HPRQPGPUVector QATdy;
    HPRQPGPUVector tempv;
    HPRQPGPUVector last_x;
    HPRQPGPUVector last_y;
    HPRQPGPUVector last_w;
    HPRQPGPUVector last_Qw;
    HPRQPGPUVector last_ATy;
    HPRQPGPUVector fact1;
    HPRQPGPUVector fact2;
    HPRQPGPUVector fact;
    HPRQPGPUVector fact_M;
    HPRQPGPUVector diag_Q;
    HPRQPGPUVector row_norm;
    HPRQPGPUVector col_norm;
    HPRQPGPUVector diag_tmp;
    HPRQPGPUVector reduce_tmp1;
    HPRQPGPUVector reduce_tmp2;
    void *blas_handle;
    void *reduce_buffer;
    size_t reduce_buffer_size;
    HPRQPGPUSpMVA spmv_A;
    HPRQPGPUSpMVAT spmv_AT;
    HPRQPGPUSpMVQ spmv_Q;
    int spmv_prepared;
    int allocated;
} HPRQPGPUWorkspace;

#ifdef __cplusplus
}
#endif

#endif