#include "spmv_operation.h"
#include "cuda_common.h"
#include <new>
#include "spmv_internal.h"

static __device__ __forceinline__ double csr_row_dot_device(const HPRQPGPUCsrIndex *row_ptr,
                                                            const HPRQPGPUCsrIndex *col_ind,
                                                            const double *values,
                                                            const double *vector,
                                                            size_t row)
{
    size_t idx;
    size_t row_start = (size_t)row_ptr[row];
    size_t row_end = (size_t)row_ptr[row + 1];
    double sum = 0.0;

    for (idx = row_start; idx < row_end; ++idx) {
        sum += values[idx] * vector[(size_t)col_ind[idx]];
    }

    return sum;
}

__global__ static void gpu_csr_spmv_kernel(double *output,
                                           const HPRQPGPUCsrIndex *row_ptr,
                                           const HPRQPGPUCsrIndex *col_ind,
                                           const double *values,
                                           const double *input,
                                           size_t nrow)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= nrow) {
        return;
    }

    output[i] = csr_row_dot_device(row_ptr, col_ind, values, input, i);
}

void gpu_spmv_a_init(HPRQPGPUSpMVA *spmv)
{
    if (spmv == NULL) {
        return;
    }
    std::memset(spmv, 0, sizeof(*spmv));
}

void gpu_spmv_at_init(HPRQPGPUSpMVAT *spmv)
{
    if (spmv == NULL) {
        return;
    }
    std::memset(spmv, 0, sizeof(*spmv));
}

void gpu_spmv_q_init(HPRQPGPUSpMVQ *spmv)
{
    if (spmv == NULL) {
        return;
    }
    std::memset(spmv, 0, sizeof(*spmv));
}

void gpu_spmv_a_free(HPRQPGPUSpMVA *spmv)
{
    if (spmv == NULL) {
        return;
    }
    delete static_cast<HPRQPSpMVOperation *>(spmv->operation);
    spmv->operation = NULL;
    spmv->buffer = NULL;
    spmv->buffer_size = 0;
    if (spmv->desc_x_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_x_bar);
    }
    if (spmv->desc_x_hat != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_x_hat);
    }
    if (spmv->desc_dx != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_dx);
    }
    if (spmv->desc_tempv != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_tempv);
    }
    if (spmv->desc_Ax != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_Ax);
    }
    if (spmv->desc_A != NULL) {
        cusparseDestroySpMat((cusparseSpMatDescr_t)spmv->desc_A);
    }
    if (spmv->handle != NULL) {
        cusparseDestroy((cusparseHandle_t)spmv->handle);
    }
    gpu_spmv_a_init(spmv);
}

void gpu_spmv_at_free(HPRQPGPUSpMVAT *spmv)
{
    if (spmv == NULL) {
        return;
    }
    delete static_cast<HPRQPSpMVOperation *>(spmv->operation);
    spmv->operation = NULL;
    spmv->buffer = NULL;
    spmv->buffer_size = 0;
    if (spmv->desc_y_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_y_bar);
    }
    if (spmv->desc_y != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_y);
    }
    if (spmv->desc_ATy_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_ATy_bar);
    }
    if (spmv->desc_ATy != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_ATy);
    }
    if (spmv->desc_ATdy != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_ATdy);
    }
    if (spmv->desc_AT != NULL) {
        cusparseDestroySpMat((cusparseSpMatDescr_t)spmv->desc_AT);
    }
    if (spmv->handle != NULL) {
        cusparseDestroy((cusparseHandle_t)spmv->handle);
    }
    gpu_spmv_at_init(spmv);
}

void gpu_spmv_q_free(HPRQPGPUSpMVQ *spmv)
{
    if (spmv == NULL) {
        return;
    }
    delete static_cast<HPRQPSpMVOperation *>(spmv->operation);
    spmv->operation = NULL;
    spmv->buffer = NULL;
    spmv->buffer_size = 0;
    if (spmv->desc_w != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_w);
    }
    if (spmv->desc_w_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_w_bar);
    }
    if (spmv->desc_x_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_x_bar);
    }
    if (spmv->desc_Qw != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_Qw);
    }
    if (spmv->desc_Qw_bar != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_Qw_bar);
    }
    if (spmv->desc_Qx != NULL) {
        cusparseDestroyDnVec((cusparseDnVecDescr_t)spmv->desc_Qx);
    }
    if (spmv->desc_Q != NULL) {
        cusparseDestroySpMat((cusparseSpMatDescr_t)spmv->desc_Q);
    }
    if (spmv->handle != NULL) {
        cusparseDestroy((cusparseHandle_t)spmv->handle);
    }
    gpu_spmv_q_init(spmv);
}

static HPRQPStatus create_csr_descriptor(cusparseSpMatDescr_t *descriptor, const HPRQPGPUMatrixCSR *matrix)
{
    return map_cusparse_status(cusparseCreateCsr(descriptor,
                                                 (int64_t)matrix->nrow,
                                                 (int64_t)matrix->ncol,
                                                 (int64_t)matrix->nnz,
                                                 matrix->row_ptr,
                                                 matrix->col_ind,
                                                 matrix->values,
                                                 CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_32I,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUDA_R_64F));
}

static HPRQPStatus create_vector_descriptor(cusparseDnVecDescr_t *descriptor, const HPRQPGPUVector *vector)
{
    return map_cusparse_status(cusparseCreateDnVec(descriptor,
                                                   (int64_t)vector->length,
                                                   vector->values,
                                                   CUDA_R_64F));
}

static HPRQPStatus prepare_spmv_common(cusparseHandle_t handle,
                                       cusparseSpMatDescr_t matrix_desc,
                                       cusparseDnVecDescr_t input_desc,
                                       cusparseDnVecDescr_t output_desc,
                                       void **operation,
                                       void **buffer,
                                       size_t *buffer_size)
{
    HPRQPSpMVOperation *op = new (std::nothrow) HPRQPSpMVOperation;
    if (!op) return map_cusparse_status(CUSPARSE_STATUS_ALLOC_FAILED);
    HPRQPStatus status = map_cusparse_status(op->prepare(handle, matrix_desc, input_desc, output_desc));
    if (status != HPRQP_STATUS_OK) {
        delete op;
        return status;
    }
    *operation = op;
    *buffer = op->buffer();
    *buffer_size = op->buffer_size();
    return HPRQP_STATUS_OK;
}

static HPRQPStatus prepare_spmv_A(HPRQPGPUSpMVA *spmv,
                                  const HPRQPGPUMatrixCSR *matrix,
                                  const HPRQPGPUWorkspace *workspace)
{
    HPRQPStatus status;
    cusparseHandle_t handle = NULL;

    status = map_cusparse_status(cusparseCreate(&handle));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    spmv->handle = handle;

    status = create_csr_descriptor((cusparseSpMatDescr_t *)&spmv->desc_A, matrix);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_x_bar, &workspace->x_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_x_hat, &workspace->x_hat);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_dx, &workspace->dx);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_tempv, &workspace->tempv);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_Ax, &workspace->Ax);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }

    status = prepare_spmv_common(handle,
                                 (cusparseSpMatDescr_t)spmv->desc_A,
                                 (cusparseDnVecDescr_t)spmv->desc_x_bar,
                                 (cusparseDnVecDescr_t)spmv->desc_Ax,
                                 &spmv->operation,
                                 &spmv->buffer,
                                 &spmv->buffer_size);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_a_free(spmv);
        return status;
    }

    spmv->prepared = 1;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus prepare_spmv_AT(HPRQPGPUSpMVAT *spmv,
                                   const HPRQPGPUMatrixCSR *matrix,
                                   const HPRQPGPUWorkspace *workspace)
{
    HPRQPStatus status;
    cusparseHandle_t handle = NULL;

    status = map_cusparse_status(cusparseCreate(&handle));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    spmv->handle = handle;

    status = create_csr_descriptor((cusparseSpMatDescr_t *)&spmv->desc_AT, matrix);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_y_bar, &workspace->y_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_y, &workspace->y);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_ATy_bar, &workspace->ATy_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_ATy, &workspace->ATy);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_ATdy, &workspace->ATdy);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }

    status = prepare_spmv_common(handle,
                                 (cusparseSpMatDescr_t)spmv->desc_AT,
                                 (cusparseDnVecDescr_t)spmv->desc_y_bar,
                                 (cusparseDnVecDescr_t)spmv->desc_ATy_bar,
                                 &spmv->operation,
                                 &spmv->buffer,
                                 &spmv->buffer_size);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_at_free(spmv);
        return status;
    }

    spmv->prepared = 1;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus prepare_spmv_Q(HPRQPGPUSpMVQ *spmv,
                                  const HPRQPGPUMatrixCSR *matrix,
                                  const HPRQPGPUWorkspace *workspace)
{
    HPRQPStatus status;
    cusparseHandle_t handle = NULL;

    status = map_cusparse_status(cusparseCreate(&handle));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    spmv->handle = handle;

    status = create_csr_descriptor((cusparseSpMatDescr_t *)&spmv->desc_Q, matrix);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_w, &workspace->w);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_w_bar, &workspace->w_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_x_bar, &workspace->x_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_Qw, &workspace->Qw);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_Qw_bar, &workspace->Qw_bar);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }
    status = create_vector_descriptor((cusparseDnVecDescr_t *)&spmv->desc_Qx, &workspace->Qx);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }

    status = prepare_spmv_common(handle,
                                 (cusparseSpMatDescr_t)spmv->desc_Q,
                                 (cusparseDnVecDescr_t)spmv->desc_w,
                                 (cusparseDnVecDescr_t)spmv->desc_Qw,
                                 &spmv->operation,
                                 &spmv->buffer,
                                 &spmv->buffer_size);
    if (status != HPRQP_STATUS_OK) {
        gpu_spmv_q_free(spmv);
        return status;
    }

    spmv->prepared = 1;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus apply_spmv(cusparseHandle_t handle,
                              cusparseSpMatDescr_t matrix_desc,
                              cusparseDnVecDescr_t input_desc,
                              cusparseDnVecDescr_t output_desc,
                              void *operation)
{
    if (!operation) return map_cusparse_status(CUSPARSE_STATUS_NOT_INITIALIZED);
    return map_cusparse_status(static_cast<HPRQPSpMVOperation *>(operation)->apply(
        handle, matrix_desc, input_desc, output_desc));
}

static HPRQPStatus apply_spmv_dynamic(cusparseHandle_t handle,
                                      cusparseSpMatDescr_t matrix_desc,
                                      double *input_values,
                                      size_t input_length,
                                      double *output_values,
                                      size_t output_length,
                                      void *operation)
{
    cusparseDnVecDescr_t input_desc = NULL;
    cusparseDnVecDescr_t output_desc = NULL;
    HPRQPStatus status;

    status = map_cusparse_status(cusparseCreateDnVec(&input_desc, (int64_t)input_length, input_values, CUDA_R_64F));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = map_cusparse_status(cusparseCreateDnVec(&output_desc, (int64_t)output_length, output_values, CUDA_R_64F));
    if (status != HPRQP_STATUS_OK) {
        cusparseDestroyDnVec(input_desc);
        return status;
    }

    status = apply_spmv(handle, matrix_desc, input_desc, output_desc, operation);
    cusparseDestroyDnVec(output_desc);
    cusparseDestroyDnVec(input_desc);
    return status;
}

static HPRQPStatus apply_spmv_custom(double *output,
                                     const HPRQPGPUMatrixCSR *matrix,
                                     const double *input)
{
    unsigned int threads;
    unsigned int blocks;

    if (output == NULL || matrix == NULL || input == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->nrow == 0) {
        return HPRQP_STATUS_OK;
    }

    threads = 256U;
    blocks = (unsigned int)((matrix->nrow + threads - 1) / threads);
    gpu_csr_spmv_kernel<<<blocks, threads>>>(output,
                                             matrix->row_ptr,
                                             matrix->col_ind,
                                             matrix->values,
                                             input,
                                             matrix->nrow);
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_prepare_spmv(HPRQPGPUWorkspace *workspace,
                                       const HPRQPGPUProblem *model,
                                       int verbose)
{
    HPRQPStatus status;

    if (workspace == NULL || model == NULL || !workspace->allocated || !model->uploaded) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    gpu_spmv_a_free(&workspace->spmv_A);
    gpu_spmv_at_free(&workspace->spmv_AT);
    gpu_spmv_q_free(&workspace->spmv_Q);
    workspace->spmv_prepared = 0;

    if (!workspace->empty_A) {
        status = prepare_spmv_A(&workspace->spmv_A, &model->A, workspace);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = prepare_spmv_AT(&workspace->spmv_AT, &model->AT, workspace);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    if (model->Q.nnz > 0) {
        status = prepare_spmv_Q(&workspace->spmv_Q, &model->Q, workspace);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    if (verbose) {
        const char *names[] = {"A", "AT", "Q"};
        void *operations[] = {workspace->spmv_A.operation, workspace->spmv_AT.operation,
                              workspace->spmv_Q.operation};
        for (int i = 0; i < 3; ++i) {
            const HPRQPSpMVOperation *op = static_cast<HPRQPSpMVOperation *>(operations[i]);
            if (!op) continue;
            std::printf("cuSPARSE %s backend: %s (scratch %zu bytes)\n", names[i],
                        op->uses_spmvop() ? "SpMVOp ALG1" : "SpMV CSR ALG2", op->buffer_size());
            if (op->fallback_status() != CUSPARSE_STATUS_SUCCESS)
                std::printf("cuSPARSE %s SpMVOp setup declined: %d; using legacy SpMV\n",
                            names[i], (int)op->fallback_status());
        }
    }
    workspace->spmv_prepared = 1;
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_spmv_apply_q_w(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    /* Zero-Q models have no descriptor; explicitly overwrite the product. */
    if (model->Q.nnz == 0)
        return map_cuda_status(cudaMemsetAsync(workspace->Qw.values, 0, workspace->n * sizeof(double)));
    (void)model;
    return apply_spmv((cusparseHandle_t)workspace->spmv_Q.handle,
                      (cusparseSpMatDescr_t)workspace->spmv_Q.desc_Q,
                      (cusparseDnVecDescr_t)workspace->spmv_Q.desc_w,
                      (cusparseDnVecDescr_t)workspace->spmv_Q.desc_Qw,
                      workspace->spmv_Q.operation);
}

HPRQPStatus gpu_spmv_apply_q_w_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    /* Zero-Q models have no descriptor; explicitly overwrite the product. */
    if (model->Q.nnz == 0)
        return map_cuda_status(cudaMemsetAsync(workspace->Qw_bar.values, 0, workspace->n * sizeof(double)));
    (void)model;
    return apply_spmv((cusparseHandle_t)workspace->spmv_Q.handle,
                      (cusparseSpMatDescr_t)workspace->spmv_Q.desc_Q,
                      (cusparseDnVecDescr_t)workspace->spmv_Q.desc_w_bar,
                      (cusparseDnVecDescr_t)workspace->spmv_Q.desc_Qw_bar,
                      workspace->spmv_Q.operation);
}

HPRQPStatus gpu_spmv_apply_q_x_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    /* Zero-Q models have no descriptor; explicitly overwrite the product. */
    if (model->Q.nnz == 0)
        return map_cuda_status(cudaMemsetAsync(workspace->Qx.values, 0, workspace->n * sizeof(double)));
    (void)model;
    return apply_spmv((cusparseHandle_t)workspace->spmv_Q.handle,
                      (cusparseSpMatDescr_t)workspace->spmv_Q.desc_Q,
                      (cusparseDnVecDescr_t)workspace->spmv_Q.desc_x_bar,
                      (cusparseDnVecDescr_t)workspace->spmv_Q.desc_Qx,
                      workspace->spmv_Q.operation);
}

HPRQPStatus gpu_spmv_apply_q_dw(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    /* Zero-Q models have no descriptor; explicitly overwrite the product. */
    if (model->Q.nnz == 0)
        return map_cuda_status(cudaMemsetAsync(workspace->dQw.values, 0, workspace->n * sizeof(double)));
    if (workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->dQw.values, &model->Q, workspace->dw.values);
    }
    return apply_spmv_dynamic((cusparseHandle_t)workspace->spmv_Q.handle,
                              (cusparseSpMatDescr_t)workspace->spmv_Q.desc_Q,
                              workspace->dw.values,
                              workspace->n,
                              workspace->dQw.values,
                              workspace->n,
                              workspace->spmv_Q.operation);
}

HPRQPStatus gpu_spmv_apply_q_atdy(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    /* Zero-Q models have no descriptor; explicitly overwrite the product. */
    if (model->Q.nnz == 0)
        return map_cuda_status(cudaMemsetAsync(workspace->QATdy.values, 0, workspace->n * sizeof(double)));
    if (workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->QATdy.values, &model->Q, workspace->ATdy.values);
    }
    return apply_spmv_dynamic((cusparseHandle_t)workspace->spmv_Q.handle,
                              (cusparseSpMatDescr_t)workspace->spmv_Q.desc_Q,
                              workspace->ATdy.values,
                              workspace->n,
                              workspace->QATdy.values,
                              workspace->n,
                              workspace->spmv_Q.operation);
}

HPRQPStatus gpu_spmv_apply_a_tempv(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    (void)model;
    return apply_spmv((cusparseHandle_t)workspace->spmv_A.handle,
                      (cusparseSpMatDescr_t)workspace->spmv_A.desc_A,
                      (cusparseDnVecDescr_t)workspace->spmv_A.desc_tempv,
                      (cusparseDnVecDescr_t)workspace->spmv_A.desc_Ax,
                      workspace->spmv_A.operation);
}

HPRQPStatus gpu_spmv_apply_a_x_hat(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    (void)model;
    return apply_spmv((cusparseHandle_t)workspace->spmv_A.handle,
                      (cusparseSpMatDescr_t)workspace->spmv_A.desc_A,
                      (cusparseDnVecDescr_t)workspace->spmv_A.desc_x_hat,
                      (cusparseDnVecDescr_t)workspace->spmv_A.desc_Ax,
                      workspace->spmv_A.operation);
}

HPRQPStatus gpu_spmv_apply_a_x_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    (void)model;
    return apply_spmv((cusparseHandle_t)workspace->spmv_A.handle,
                      (cusparseSpMatDescr_t)workspace->spmv_A.desc_A,
                      (cusparseDnVecDescr_t)workspace->spmv_A.desc_x_bar,
                      (cusparseDnVecDescr_t)workspace->spmv_A.desc_Ax,
                      workspace->spmv_A.operation);
}

HPRQPStatus gpu_spmv_apply_at_y(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->ATy.values, &model->AT, workspace->y.values);
    }
    return apply_spmv((cusparseHandle_t)workspace->spmv_AT.handle,
                      (cusparseSpMatDescr_t)workspace->spmv_AT.desc_AT,
                      (cusparseDnVecDescr_t)workspace->spmv_AT.desc_y,
                      (cusparseDnVecDescr_t)workspace->spmv_AT.desc_ATy,
                      workspace->spmv_AT.operation);
}

HPRQPStatus gpu_spmv_apply_at_y_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->ATy_bar.values, &model->AT, workspace->y_bar.values);
    }
    return apply_spmv((cusparseHandle_t)workspace->spmv_AT.handle,
                      (cusparseSpMatDescr_t)workspace->spmv_AT.desc_AT,
                      (cusparseDnVecDescr_t)workspace->spmv_AT.desc_y_bar,
                      (cusparseDnVecDescr_t)workspace->spmv_AT.desc_ATy_bar,
                      workspace->spmv_AT.operation);
}

HPRQPStatus gpu_spmv_apply_at_dy(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        return apply_spmv_custom(workspace->ATdy.values, &model->AT, workspace->dy.values);
    }
    return apply_spmv_dynamic((cusparseHandle_t)workspace->spmv_AT.handle,
                              (cusparseSpMatDescr_t)workspace->spmv_AT.desc_AT,
                              workspace->dy.values,
                              workspace->m,
                              workspace->ATdy.values,
                              workspace->n,
                              workspace->spmv_AT.operation);
}

HPRQPStatus gpu_spmv_apply_at_y_bar_residual(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model)
{
    (void)model;
    return apply_spmv((cusparseHandle_t)workspace->spmv_AT.handle,
                      (cusparseSpMatDescr_t)workspace->spmv_AT.desc_AT,
                      (cusparseDnVecDescr_t)workspace->spmv_AT.desc_y_bar,
                      (cusparseDnVecDescr_t)workspace->spmv_AT.desc_ATdy,
                      workspace->spmv_AT.operation);
}