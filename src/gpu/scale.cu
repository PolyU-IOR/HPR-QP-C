#include "packed_spmv.cuh"
#include "cuda_common.h"
#include "soc_common.cuh"

__global__ static void gpu_fill_vector_kernel(double *values, double scalar, size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    values[i] = scalar;
}

__global__ static void gpu_compute_row_max_abs_with_q_kernel(double *output,
                                                             const HPRQPGPUCsrIndex *at_row_ptr,
                                                             const double *at_values,
                                                             const HPRQPGPUCsrIndex *q_row_ptr,
                                                             const double *q_values,
                                                             size_t n)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double max_abs = 0.0;
    size_t idx;

    if (row >= n) {
        return;
    }

    for (idx = at_row_ptr[row]; idx < at_row_ptr[row + 1]; ++idx) {
        double value = fabs(at_values[idx]);
        if (value > max_abs) {
            max_abs = value;
        }
    }
    for (idx = q_row_ptr[row]; idx < q_row_ptr[row + 1]; ++idx) {
        double value = fabs(q_values[idx]);
        if (value > max_abs) {
            max_abs = value;
        }
    }

    output[row] = max_abs > 0.0 ? sqrt(max_abs) : 1.0;
}

__global__ static void gpu_compute_row_max_abs_kernel(double *output,
                                                      const HPRQPGPUCsrIndex *row_ptr,
                                                      const double *values,
                                                      size_t nrow)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double max_abs = 0.0;
    size_t idx;

    if (row >= nrow) {
        return;
    }

    for (idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        double value = fabs(values[idx]);
        if (value > max_abs) {
            max_abs = value;
        }
    }

    output[row] = max_abs > 0.0 ? sqrt(max_abs) : 1.0;
}

__global__ static void gpu_compute_row_sum_abs_with_q_kernel(double *output,
                                                             const HPRQPGPUCsrIndex *at_row_ptr,
                                                             const double *at_values,
                                                             const HPRQPGPUCsrIndex *q_row_ptr,
                                                             const double *q_values,
                                                             size_t n)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double sum_abs = 0.0;
    size_t idx;

    if (row >= n) {
        return;
    }

    for (idx = at_row_ptr[row]; idx < at_row_ptr[row + 1]; ++idx) {
        sum_abs += fabs(at_values[idx]);
    }
    for (idx = q_row_ptr[row]; idx < q_row_ptr[row + 1]; ++idx) {
        sum_abs += fabs(q_values[idx]);
    }

    output[row] = sum_abs > 0.0 ? sqrt(sum_abs) : 1.0;
}

__global__ static void gpu_compute_row_sum_abs_kernel(double *output,
                                                      const HPRQPGPUCsrIndex *row_ptr,
                                                      const double *values,
                                                      size_t nrow)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double sum_abs = 0.0;
    size_t idx;

    if (row >= nrow) {
        return;
    }

    for (idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        sum_abs += fabs(values[idx]);
    }

    output[row] = sum_abs > 0.0 ? sqrt(sum_abs) : 1.0;
}

__global__ static void gpu_scale_rows_csr_kernel(const HPRQPGPUCsrIndex *row_ptr,
                                                 double *values,
                                                 const double *row_scale,
                                                 size_t nrow)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t idx;
    double scale;

    if (row >= nrow) {
        return;
    }

    scale = 1.0 / row_scale[row];
    for (idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        values[idx] *= scale;
    }
}

__global__ static void gpu_scale_cols_csr_kernel(const HPRQPGPUCsrIndex *row_ptr,
                                                 const HPRQPGPUCsrIndex *col_ind,
                                                 double *values,
                                                 const double *col_scale,
                                                 size_t nrow)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t idx;

    if (row >= nrow) {
        return;
    }

    for (idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        values[idx] /= col_scale[col_ind[idx]];
    }
}

__global__ static void gpu_scale_vector_div_kernel(double *values,
                                                   const double *scale,
                                                   size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    values[i] /= scale[i];
}

__global__ static void gpu_scale_vector_mul_kernel(double *values,
                                                   const double *scale,
                                                   size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    values[i] *= scale[i];
}

__global__ static void gpu_bound_measure_kernel(double *output,
                                                const double *al,
                                                const double *au,
                                                size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double lower;
    double upper;

    if (i >= length) {
        return;
    }

    lower = isinf(al[i]) ? 0.0 : fabs(al[i]);
    upper = isinf(au[i]) ? 0.0 : fabs(au[i]);
    output[i] = lower > upper ? lower : upper;
}

__global__ static void gpu_pointwise_multiply_kernel(double *output,
                                                     const double *lhs,
                                                     const double *rhs,
                                                     size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }
    output[i] = lhs[i] * rhs[i];
}

static HPRQPStatus gpu_launch_fill_vector(HPRQPGPUVector *vector, double scalar, unsigned int threads)
{
    unsigned int blocks;

    if (vector == NULL || (vector->length > 0 && vector->values == NULL)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (vector->length == 0) {
        return HPRQP_STATUS_OK;
    }

    blocks = (unsigned int)((vector->length + threads - 1) / threads);
    gpu_fill_vector_kernel<<<blocks, threads>>>(vector->values, scalar, vector->length);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_compute_bound_norm2_local(cublasHandle_t handle,
                                                 HPRQPGPUWorkspace *workspace,
                                                 const HPRQPGPUVector *al,
                                                 const HPRQPGPUVector *au,
                                                 double *result)
{
    HPRQPStatus status;

    if (result == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->m == 0) {
        *result = 0.0;
        return HPRQP_STATUS_OK;
    }

    gpu_bound_measure_kernel<<<workspace->blocks_m, workspace->threads_m>>>(workspace->reduce_tmp1.values,
                                                                             al->values,
                                                                             au->values,
                                                                             workspace->m);
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    return gpu_raw_norm2(handle, workspace->m, workspace->reduce_tmp1.values, result);
}

static HPRQPStatus gpu_compute_bound_max_local(cublasHandle_t handle,
                                               HPRQPGPUWorkspace *workspace,
                                               const HPRQPGPUVector *al,
                                               const HPRQPGPUVector *au,
                                               double *result)
{
    HPRQPStatus status;

    if (result == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->m == 0) {
        *result = 0.0;
        return HPRQP_STATUS_OK;
    }

    gpu_bound_measure_kernel<<<workspace->blocks_m, workspace->threads_m>>>(workspace->reduce_tmp1.values,
                                                                             al->values,
                                                                             au->values,
                                                                             workspace->m);
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    return gpu_raw_max_abs(handle, workspace->m, workspace->reduce_tmp1.values, result);
}

static HPRQPStatus gpu_download_scaling_to_host(HPRQPGPUWorkspace *workspace,
                                                HPRQPGPUProblem *gpu_problem,
                                                HPRQPScaling *host_scaling)
{
    cublasHandle_t handle;
    HPRQPStatus status;

    if (workspace == NULL || gpu_problem == NULL || host_scaling == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    handle = gpu_blas_handle(workspace);
    if (handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    scaling_free(host_scaling);
    host_scaling->b_scale = workspace->scale_b_scale;
    host_scaling->c_scale = workspace->scale_c_scale;
    host_scaling->norm_b_org = workspace->scale_norm_b_org;
    host_scaling->norm_c_org = workspace->scale_norm_c_org;

    if (workspace->n > 0) {
        status = gpu_raw_norm2(handle, workspace->n, gpu_problem->c.values, &host_scaling->norm_c);
        if (status != HPRQP_STATUS_OK) {
            scaling_free(host_scaling);
            return status;
        }
    }

    if (workspace->m > 0) {
        status = gpu_compute_bound_norm2_local(handle,
                                               workspace,
                                               &gpu_problem->AL,
                                               &gpu_problem->AU,
                                               &host_scaling->norm_b);
        if (status != HPRQP_STATUS_OK) {
            scaling_free(host_scaling);
            return status;
        }
    }

    if (gpu_problem->num_row_cones) {
        double rhs_norm;
        status = gpu_raw_norm2(handle, workspace->m, gpu_problem->soc_rhs.values, &rhs_norm);
        if (status != HPRQP_STATUS_OK) return status;
        host_scaling->norm_b = fmax(host_scaling->norm_b, rhs_norm);
    }
    return packed_spmv_refresh(gpu_problem);
}

HPRQPStatus gpu_problem_scale(HPRQPGPUProblem *gpu_problem,
                              HPRQPGPUWorkspace *workspace,
                              const HPRQPParams *params,
                              HPRQPScaling *host_scaling)
{
    cublasHandle_t handle;
    HPRQPStatus status;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;
    size_t iter;

    if (gpu_problem == NULL || workspace == NULL || params == NULL || host_scaling == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    handle = gpu_blas_handle(workspace);
    if (handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    threads_n = workspace->threads_n;
    blocks_n = workspace->blocks_n;
    threads_m = workspace->threads_m;
    blocks_m = workspace->blocks_m;

    workspace->scale_b_scale = 1.0;
    workspace->scale_c_scale = 1.0;
    status = gpu_compute_bound_max_local(handle,
                                         workspace,
                                         &gpu_problem->AL,
                                         &gpu_problem->AU,
                                         &workspace->scale_norm_b_org);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    if (gpu_problem->num_row_cones) {
        double rhs_max;
        status = gpu_raw_max_abs(handle, workspace->m, gpu_problem->soc_rhs.values, &rhs_max);
        if (status != HPRQP_STATUS_OK) return status;
        workspace->scale_norm_b_org = fmax(workspace->scale_norm_b_org, rhs_max);
    }
    status = gpu_raw_max_abs(handle,
                             workspace->n,
                             gpu_problem->c.values,
                             &workspace->scale_norm_c_org);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    if (gpu_launch_fill_vector(&workspace->col_norm, 1.0, threads_n) != HPRQP_STATUS_OK) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    if (!workspace->empty_A && gpu_launch_fill_vector(&workspace->row_norm, 1.0, threads_m) != HPRQP_STATUS_OK) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    if (params->use_scaling) {
        for (iter = 0; iter < 10; ++iter) {
            if (threads_n > 0U) {
                gpu_compute_row_max_abs_with_q_kernel<<<blocks_n, threads_n>>>(workspace->reduce_tmp2.values,
                                                                                gpu_problem->AT.row_ptr,
                                                                                gpu_problem->AT.values,
                                                                                gpu_problem->Q.row_ptr,
                                                                                gpu_problem->Q.values,
                                                                                workspace->n);
            }
            if (!workspace->empty_A && threads_m > 0U) {
                gpu_compute_row_max_abs_kernel<<<blocks_m, threads_m>>>(workspace->reduce_tmp1.values,
                                                                        gpu_problem->A.row_ptr,
                                                                        gpu_problem->A.values,
                                                                        workspace->m);
            }
            if (gpu_soc_scaling_blocks(gpu_problem, workspace) != HPRQP_STATUS_OK)
                return HPRQP_STATUS_NOT_IMPLEMENTED;
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                return HPRQP_STATUS_NOT_IMPLEMENTED;
            }

            if (threads_n > 0U) {
                gpu_pointwise_multiply_kernel<<<blocks_n, threads_n>>>(workspace->col_norm.values,
                                                                       workspace->col_norm.values,
                                                                       workspace->reduce_tmp2.values,
                                                                       workspace->n);
                gpu_scale_rows_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->Q.row_ptr,
                                                                   gpu_problem->Q.values,
                                                                   workspace->reduce_tmp2.values,
                                                                   workspace->n);
                gpu_scale_cols_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->Q.row_ptr,
                                                                   gpu_problem->Q.col_ind,
                                                                   gpu_problem->Q.values,
                                                                   workspace->reduce_tmp2.values,
                                                                   workspace->n);
                gpu_scale_vector_div_kernel<<<blocks_n, threads_n>>>(gpu_problem->c.values,
                                                                     workspace->reduce_tmp2.values,
                                                                     workspace->n);
                gpu_scale_vector_mul_kernel<<<blocks_n, threads_n>>>(gpu_problem->l.values,
                                                                     workspace->reduce_tmp2.values,
                                                                     workspace->n);
                gpu_scale_vector_mul_kernel<<<blocks_n, threads_n>>>(gpu_problem->u.values,
                                                                     workspace->reduce_tmp2.values,
                                                                     workspace->n);
            }

            if (!workspace->empty_A && threads_m > 0U) {
                gpu_pointwise_multiply_kernel<<<blocks_m, threads_m>>>(workspace->row_norm.values,
                                                                       workspace->row_norm.values,
                                                                       workspace->reduce_tmp1.values,
                                                                       workspace->m);
                gpu_scale_rows_csr_kernel<<<blocks_m, threads_m>>>(gpu_problem->A.row_ptr,
                                                                   gpu_problem->A.values,
                                                                   workspace->reduce_tmp1.values,
                                                                   workspace->m);
                gpu_scale_cols_csr_kernel<<<blocks_m, threads_m>>>(gpu_problem->A.row_ptr,
                                                                   gpu_problem->A.col_ind,
                                                                   gpu_problem->A.values,
                                                                   workspace->reduce_tmp2.values,
                                                                   workspace->m);
                gpu_scale_rows_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->AT.row_ptr,
                                                                   gpu_problem->AT.values,
                                                                   workspace->reduce_tmp2.values,
                                                                   workspace->n);
                gpu_scale_cols_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->AT.row_ptr,
                                                                   gpu_problem->AT.col_ind,
                                                                   gpu_problem->AT.values,
                                                                   workspace->reduce_tmp1.values,
                                                                   workspace->n);
                gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->AL.values,
                                                                     workspace->reduce_tmp1.values,
                                                                     workspace->m);
                gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->AU.values,
                                                                     workspace->reduce_tmp1.values,
                                                                     workspace->m);
                if (gpu_problem->num_row_cones)
                    gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->soc_rhs.values,
                        workspace->reduce_tmp1.values, workspace->m);
            }

            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                return HPRQP_STATUS_NOT_IMPLEMENTED;
            }
        }
        if (threads_n > 0U) {
            gpu_compute_row_sum_abs_with_q_kernel<<<blocks_n, threads_n>>>(workspace->reduce_tmp2.values,
                                                                            gpu_problem->AT.row_ptr,
                                                                            gpu_problem->AT.values,
                                                                            gpu_problem->Q.row_ptr,
                                                                            gpu_problem->Q.values,
                                                                            workspace->n);
        }
        if (!workspace->empty_A && threads_m > 0U) {
            gpu_compute_row_sum_abs_kernel<<<blocks_m, threads_m>>>(workspace->reduce_tmp1.values,
                                                                    gpu_problem->A.row_ptr,
                                                                    gpu_problem->A.values,
                                                                    workspace->m);
        }
        if (gpu_soc_scaling_blocks(gpu_problem, workspace) != HPRQP_STATUS_OK)
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }

        if (threads_n > 0U) {
            gpu_pointwise_multiply_kernel<<<blocks_n, threads_n>>>(workspace->col_norm.values,
                                                                   workspace->col_norm.values,
                                                                   workspace->reduce_tmp2.values,
                                                                   workspace->n);
            gpu_scale_rows_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->Q.row_ptr,
                                                               gpu_problem->Q.values,
                                                               workspace->reduce_tmp2.values,
                                                               workspace->n);
            gpu_scale_cols_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->Q.row_ptr,
                                                               gpu_problem->Q.col_ind,
                                                               gpu_problem->Q.values,
                                                               workspace->reduce_tmp2.values,
                                                               workspace->n);
            gpu_scale_vector_div_kernel<<<blocks_n, threads_n>>>(gpu_problem->c.values,
                                                                 workspace->reduce_tmp2.values,
                                                                 workspace->n);
            gpu_scale_vector_mul_kernel<<<blocks_n, threads_n>>>(gpu_problem->l.values,
                                                                 workspace->reduce_tmp2.values,
                                                                 workspace->n);
            gpu_scale_vector_mul_kernel<<<blocks_n, threads_n>>>(gpu_problem->u.values,
                                                                 workspace->reduce_tmp2.values,
                                                                 workspace->n);
        }

        if (!workspace->empty_A && threads_m > 0U) {
            gpu_pointwise_multiply_kernel<<<blocks_m, threads_m>>>(workspace->row_norm.values,
                                                                   workspace->row_norm.values,
                                                                   workspace->reduce_tmp1.values,
                                                                   workspace->m);
            gpu_scale_rows_csr_kernel<<<blocks_m, threads_m>>>(gpu_problem->A.row_ptr,
                                                               gpu_problem->A.values,
                                                               workspace->reduce_tmp1.values,
                                                               workspace->m);
            gpu_scale_cols_csr_kernel<<<blocks_m, threads_m>>>(gpu_problem->A.row_ptr,
                                                               gpu_problem->A.col_ind,
                                                               gpu_problem->A.values,
                                                               workspace->reduce_tmp2.values,
                                                               workspace->m);
            gpu_scale_rows_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->AT.row_ptr,
                                                               gpu_problem->AT.values,
                                                               workspace->reduce_tmp2.values,
                                                               workspace->n);
            gpu_scale_cols_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->AT.row_ptr,
                                                               gpu_problem->AT.col_ind,
                                                               gpu_problem->AT.values,
                                                               workspace->reduce_tmp1.values,
                                                               workspace->n);
            gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->AL.values,
                                                                 workspace->reduce_tmp1.values,
                                                                 workspace->m);
            gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->AU.values,
                                                                 workspace->reduce_tmp1.values,
                                                                 workspace->m);
            if (gpu_problem->num_row_cones)
                gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->soc_rhs.values,
                    workspace->reduce_tmp1.values, workspace->m);
        }

        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
    }

    return gpu_download_scaling_to_host(workspace, gpu_problem, host_scaling);
}
