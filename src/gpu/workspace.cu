#include "cuda_common.h"
#include "spmv_internal.h"

__global__ static void gpu_unscale_div_output_kernel(double *output,
                                                     const double *input,
                                                     const double *scale,
                                                     double scalar,
                                                     size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    output[i] = scalar * (input[i] / scale[i]);
}

__global__ static void gpu_unscale_mul_output_kernel(double *output,
                                                     const double *input,
                                                     const double *scale,
                                                     double scalar,
                                                     size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    output[i] = scalar * (input[i] * scale[i]);
}

void gpu_workspace_init(HPRQPGPUWorkspace *workspace)
{
    if (workspace == NULL) {
        return;
    }
    std::memset(workspace, 0, sizeof(*workspace));
    workspace->spmv_mode_q = HPRQP_SPMV_MODE_CUSPARSE;
    workspace->spmv_mode_a = HPRQP_SPMV_MODE_CUSPARSE;
    gpu_vector_init(&workspace->x);
    gpu_vector_init(&workspace->x_bar);
    gpu_vector_init(&workspace->x_hat);
    gpu_vector_init(&workspace->z_bar);
    gpu_vector_init(&workspace->dx);
    gpu_vector_init(&workspace->w);
    gpu_vector_init(&workspace->w_bar);
    gpu_vector_init(&workspace->dw);
    gpu_vector_init(&workspace->Qw);
    gpu_vector_init(&workspace->Qw_bar);
    gpu_vector_init(&workspace->Qx);
    gpu_vector_init(&workspace->y);
    gpu_vector_init(&workspace->y_bar);
    gpu_vector_init(&workspace->dy);
    gpu_vector_init(&workspace->s);
    gpu_vector_init(&workspace->Rp);
    gpu_vector_init(&workspace->Rd);
    gpu_vector_init(&workspace->Ax);
    gpu_vector_init(&workspace->ATy);
    gpu_vector_init(&workspace->ATy_bar);
    gpu_vector_init(&workspace->ATdy);
    gpu_vector_init(&workspace->dQw);
    gpu_vector_init(&workspace->QATdy);
    gpu_vector_init(&workspace->tempv);
    gpu_vector_init(&workspace->last_x);
    gpu_vector_init(&workspace->last_y);
    gpu_vector_init(&workspace->last_w);
    gpu_vector_init(&workspace->last_Qw);
    gpu_vector_init(&workspace->last_ATy);
    gpu_vector_init(&workspace->fact1);
    gpu_vector_init(&workspace->fact2);
    gpu_vector_init(&workspace->fact);
    gpu_vector_init(&workspace->fact_M);
    gpu_vector_init(&workspace->diag_Q);
    gpu_vector_init(&workspace->row_norm);
    gpu_vector_init(&workspace->col_norm);
    gpu_vector_init(&workspace->diag_tmp);
    gpu_vector_init(&workspace->reduce_tmp1);
    gpu_vector_init(&workspace->reduce_tmp2);
    gpu_spmv_a_init(&workspace->spmv_A);
    gpu_spmv_at_init(&workspace->spmv_AT);
    gpu_spmv_q_init(&workspace->spmv_Q);
}

void gpu_workspace_free(HPRQPGPUWorkspace *workspace)
{
    if (workspace == NULL) {
        return;
    }
    gpu_vector_free(&workspace->x);
    gpu_vector_free(&workspace->x_bar);
    gpu_vector_free(&workspace->x_hat);
    gpu_vector_free(&workspace->z_bar);
    gpu_vector_free(&workspace->dx);
    gpu_vector_free(&workspace->w);
    gpu_vector_free(&workspace->w_bar);
    gpu_vector_free(&workspace->dw);
    gpu_vector_free(&workspace->Qw);
    gpu_vector_free(&workspace->Qw_bar);
    gpu_vector_free(&workspace->Qx);
    gpu_vector_free(&workspace->y);
    gpu_vector_free(&workspace->y_bar);
    gpu_vector_free(&workspace->dy);
    gpu_vector_free(&workspace->s);
    gpu_vector_free(&workspace->Rp);
    gpu_vector_free(&workspace->Rd);
    gpu_vector_free(&workspace->Ax);
    gpu_vector_free(&workspace->ATy);
    gpu_vector_free(&workspace->ATy_bar);
    gpu_vector_free(&workspace->ATdy);
    gpu_vector_free(&workspace->dQw);
    gpu_vector_free(&workspace->QATdy);
    gpu_vector_free(&workspace->tempv);
    gpu_vector_free(&workspace->last_x);
    gpu_vector_free(&workspace->last_y);
    gpu_vector_free(&workspace->last_w);
    gpu_vector_free(&workspace->last_Qw);
    gpu_vector_free(&workspace->last_ATy);
    gpu_vector_free(&workspace->fact1);
    gpu_vector_free(&workspace->fact2);
    gpu_vector_free(&workspace->fact);
    gpu_vector_free(&workspace->fact_M);
    gpu_vector_free(&workspace->diag_Q);
    gpu_vector_free(&workspace->row_norm);
    gpu_vector_free(&workspace->col_norm);
    gpu_vector_free(&workspace->diag_tmp);
    gpu_vector_free(&workspace->reduce_tmp1);
    gpu_vector_free(&workspace->reduce_tmp2);
    if (workspace->blas_handle != NULL) {
        cublasDestroy((cublasHandle_t)workspace->blas_handle);
        workspace->blas_handle = NULL;
    }
    gpu_spmv_a_free(&workspace->spmv_A);
    gpu_spmv_at_free(&workspace->spmv_AT);
    gpu_spmv_q_free(&workspace->spmv_Q);
    gpu_workspace_init(workspace);
}

HPRQPStatus gpu_workspace_allocate(HPRQPGPUWorkspace *workspace,
                                   const HPRQPGPUProblem *model,
                                   const HPRQPParams *params)
{
    HPRQPStatus status;
    size_t reduce_length;
    cublasHandle_t blas_handle = NULL;
    (void)params;

    if (workspace == NULL || model == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (model->n == 0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    workspace->m = model->m;
    workspace->n = model->n;
    workspace->empty_A = model->empty_A;
    workspace->threads_n = 256U;
    workspace->blocks_n = (unsigned int)((model->n + workspace->threads_n - 1) / workspace->threads_n);
    workspace->threads_m = workspace->empty_A ? 0U : 256U;
    workspace->blocks_m = workspace->empty_A ? 0U : (unsigned int)((model->m + workspace->threads_m - 1) / workspace->threads_m);

    status = gpu_vector_alloc(&workspace->x, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->x_bar, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->x_hat, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->z_bar, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->dx, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->w, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->w_bar, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->dw, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->Qw, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->Qw_bar, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->Qx, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->y, model->m);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->y_bar, model->m);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->dy, model->m);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->s, model->m);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->Rp, model->m);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->Rd, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->Ax, model->m);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->ATy, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->ATy_bar, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->ATdy, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->dQw, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->QATdy, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->tempv, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->last_x, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->last_y, model->m);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->last_w, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->last_Qw, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->last_ATy, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->fact1, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->fact2, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->fact, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->fact_M, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->diag_Q, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->row_norm, model->m);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->col_norm, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->diag_tmp, model->n);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }

    reduce_length = model->n > model->m ? model->n : model->m;
    status = gpu_vector_alloc(&workspace->reduce_tmp1, reduce_length);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    status = gpu_vector_alloc(&workspace->reduce_tmp2, reduce_length);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }

    status = map_cublas_status(cublasCreate(&blas_handle));
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(workspace);
        return status;
    }
    workspace->blas_handle = blas_handle;

    workspace->allocated = 1;
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_upload_scaling(HPRQPGPUWorkspace *workspace,
                                         const HPRQPScaling *scaling)
{
    size_t i;
    HPRQPStatus status;
    double *host_row_norm;
    double *host_col_norm;

    if (workspace == NULL || scaling == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    workspace->scale_b_scale = scaling->b_scale;
    workspace->scale_c_scale = scaling->c_scale;
    workspace->scale_norm_b_org = scaling->norm_b_org;
    workspace->scale_norm_c_org = scaling->norm_c_org;

    host_row_norm = NULL;
    host_col_norm = NULL;

    if (!workspace->empty_A) {
        host_row_norm = (double *)malloc(workspace->m * sizeof(double));
        if (host_row_norm == NULL) {
            return HPRQP_STATUS_ALLOCATION_FAILED;
        }
        for (i = 0; i < workspace->m; ++i) {
            host_row_norm[i] = scaling->row_norm != NULL ? scaling->row_norm[i] : 1.0;
        }
        status = gpu_vector_copy_from_host(&workspace->row_norm, host_row_norm, workspace->m);
        free(host_row_norm);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    if (workspace->n > 0) {
        host_col_norm = (double *)malloc(workspace->n * sizeof(double));
        if (host_col_norm == NULL) {
            return HPRQP_STATUS_ALLOCATION_FAILED;
        }
        for (i = 0; i < workspace->n; ++i) {
            host_col_norm[i] = scaling->col_norm != NULL ? scaling->col_norm[i] : 1.0;
        }
        status = gpu_vector_copy_from_host(&workspace->col_norm, host_col_norm, workspace->n);
        free(host_col_norm);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_copy_output_unscaled_to_host(HPRQPGPUWorkspace *gpu_workspace,
                                                       double *x,
                                                       double *w,
                                                       double *z,
                                                       double *y)
{
    HPRQPStatus status;

    if (gpu_workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (gpu_workspace->n > 0) {
        if (x == NULL || w == NULL || z == NULL) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }

        gpu_unscale_div_output_kernel<<<gpu_workspace->blocks_n, gpu_workspace->threads_n>>>(gpu_workspace->diag_tmp.values,
                                                                                              gpu_workspace->x_bar.values,
                                                                                              gpu_workspace->col_norm.values,
                                                                                              gpu_workspace->scale_b_scale,
                                                                                              gpu_workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_vector_copy_to_host(&gpu_workspace->diag_tmp, x, gpu_workspace->n);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }

        gpu_unscale_div_output_kernel<<<gpu_workspace->blocks_n, gpu_workspace->threads_n>>>(gpu_workspace->diag_tmp.values,
                                                                                              gpu_workspace->w_bar.values,
                                                                                              gpu_workspace->col_norm.values,
                                                                                              gpu_workspace->scale_b_scale,
                                                                                              gpu_workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_vector_copy_to_host(&gpu_workspace->diag_tmp, w, gpu_workspace->n);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }

        gpu_unscale_mul_output_kernel<<<gpu_workspace->blocks_n, gpu_workspace->threads_n>>>(gpu_workspace->diag_tmp.values,
                                                                                              gpu_workspace->z_bar.values,
                                                                                              gpu_workspace->col_norm.values,
                                                                                              gpu_workspace->scale_c_scale,
                                                                                              gpu_workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_vector_copy_to_host(&gpu_workspace->diag_tmp, z, gpu_workspace->n);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    if (gpu_workspace->m > 0) {
        if (y == NULL) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }

        gpu_unscale_div_output_kernel<<<gpu_workspace->blocks_m, gpu_workspace->threads_m>>>(gpu_workspace->reduce_tmp1.values,
                                                                                              gpu_workspace->y_bar.values,
                                                                                              gpu_workspace->row_norm.values,
                                                                                              gpu_workspace->scale_c_scale,
                                                                                              gpu_workspace->m);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = map_cuda_status(cudaMemcpy(y,
                                            gpu_workspace->reduce_tmp1.values,
                                            gpu_workspace->m * sizeof(double),
                                            cudaMemcpyDeviceToHost));
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_sync_from_cpu(HPRQPGPUWorkspace *gpu_workspace,
                                        const HPRQPWorkspace *cpu_workspace)
{
    HPRQPStatus status;

    if (gpu_workspace == NULL || cpu_workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    gpu_workspace->sigma = cpu_workspace->sigma;
    gpu_workspace->lambda_max_A = cpu_workspace->lambda_max_A;
    gpu_workspace->lambda_max_Q = cpu_workspace->lambda_max_Q;
    gpu_workspace->Q_is_diag = cpu_workspace->Q_is_diag;
    gpu_workspace->noC = cpu_workspace->noC;
    gpu_workspace->to_check = cpu_workspace->to_check;

    status = gpu_vector_copy_from_host(&gpu_workspace->x, cpu_workspace->x, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->x_bar, cpu_workspace->x_bar, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->x_hat, cpu_workspace->x_hat, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->z_bar, cpu_workspace->z_bar, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->w, cpu_workspace->w, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->w_bar, cpu_workspace->w_bar, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->Qw, cpu_workspace->Qw, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->Qw_bar, cpu_workspace->Qw_bar, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->Qx, cpu_workspace->Qx, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->tempv, cpu_workspace->tempv, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->ATy, cpu_workspace->ATy, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->ATy_bar, cpu_workspace->ATy_bar, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->ATdy, cpu_workspace->ATdy, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->dQw, cpu_workspace->dQw, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->QATdy, cpu_workspace->QATdy, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->last_x, cpu_workspace->last_x, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->last_w, cpu_workspace->last_w, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->last_Qw, cpu_workspace->last_Qw, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->last_ATy, cpu_workspace->last_ATy, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->fact1, cpu_workspace->fact1, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->fact2, cpu_workspace->fact2, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->fact, cpu_workspace->fact, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->fact_M, cpu_workspace->fact_M, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_from_host(&gpu_workspace->diag_Q, cpu_workspace->diag_Q, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;

    if (!gpu_workspace->empty_A) {
        status = gpu_vector_copy_from_host(&gpu_workspace->y, cpu_workspace->y, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_vector_copy_from_host(&gpu_workspace->y_bar, cpu_workspace->y_bar, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_vector_copy_from_host(&gpu_workspace->Rp, cpu_workspace->Rp, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_vector_copy_from_host(&gpu_workspace->Ax, cpu_workspace->Ax, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_vector_copy_from_host(&gpu_workspace->last_y, cpu_workspace->last_y, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_vector_copy_from_host(&gpu_workspace->s, cpu_workspace->s, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_workspace_copy_scalars_to_cpu(HPRQPGPUWorkspace *gpu_workspace,
                                                     HPRQPWorkspace *cpu_workspace)
{
    if (gpu_workspace == NULL || cpu_workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    cpu_workspace->sigma = gpu_workspace->sigma;
    cpu_workspace->lambda_max_A = gpu_workspace->lambda_max_A;
    cpu_workspace->lambda_max_Q = gpu_workspace->lambda_max_Q;
    cpu_workspace->Q_is_diag = gpu_workspace->Q_is_diag;
    cpu_workspace->noC = gpu_workspace->noC;

    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_copy_main_state_to_cpu(HPRQPGPUWorkspace *gpu_workspace,
                                                 HPRQPWorkspace *cpu_workspace)
{
    HPRQPStatus status;

    status = gpu_workspace_copy_scalars_to_cpu(gpu_workspace, cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    status = gpu_vector_copy_to_host(&gpu_workspace->x, cpu_workspace->x, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->x_bar, cpu_workspace->x_bar, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->x_hat, cpu_workspace->x_hat, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->z_bar, cpu_workspace->z_bar, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->w, cpu_workspace->w, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->w_bar, cpu_workspace->w_bar, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->Qw, cpu_workspace->Qw, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->Qw_bar, cpu_workspace->Qw_bar, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->Qx, cpu_workspace->Qx, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->ATy, cpu_workspace->ATy, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->ATy_bar, cpu_workspace->ATy_bar, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->ATdy, cpu_workspace->ATdy, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->dQw, cpu_workspace->dQw, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->QATdy, cpu_workspace->QATdy, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->Rd, cpu_workspace->Rd, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->tempv, cpu_workspace->tempv, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;

    if (!gpu_workspace->empty_A) {
        status = gpu_vector_copy_to_host(&gpu_workspace->y, cpu_workspace->y, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_vector_copy_to_host(&gpu_workspace->y_bar, cpu_workspace->y_bar, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_vector_copy_to_host(&gpu_workspace->Rp, cpu_workspace->Rp, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_vector_copy_to_host(&gpu_workspace->Ax, cpu_workspace->Ax, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_vector_copy_to_host(&gpu_workspace->s, cpu_workspace->s, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_copy_history_state_to_cpu(HPRQPGPUWorkspace *gpu_workspace,
                                                    HPRQPWorkspace *cpu_workspace)
{
    HPRQPStatus status;

    status = gpu_workspace_copy_scalars_to_cpu(gpu_workspace, cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    status = gpu_vector_copy_to_host(&gpu_workspace->last_x, cpu_workspace->last_x, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->last_w, cpu_workspace->last_w, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->last_Qw, cpu_workspace->last_Qw, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->last_ATy, cpu_workspace->last_ATy, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    if (!gpu_workspace->empty_A) {
        status = gpu_vector_copy_to_host(&gpu_workspace->last_y, cpu_workspace->last_y, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_copy_check_state_to_cpu(HPRQPGPUWorkspace *gpu_workspace,
                                                  HPRQPWorkspace *cpu_workspace)
{
    HPRQPStatus status;

    status = gpu_workspace_copy_scalars_to_cpu(gpu_workspace, cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    status = gpu_vector_copy_to_host(&gpu_workspace->dx, cpu_workspace->dx, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_to_host(&gpu_workspace->dw, cpu_workspace->dw, cpu_workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    if (!gpu_workspace->empty_A) {
        status = gpu_vector_copy_to_host(&gpu_workspace->dy, cpu_workspace->dy, cpu_workspace->m);
        if (status != HPRQP_STATUS_OK) return status;
    }

    return HPRQP_STATUS_OK;
}