#include "cuda_common.h"
#include "gpu/row_metric.h"
#include "gpu/qp_row_metric.h"

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

__global__ static void gpu_free_variable_mask_kernel(double *output,
                                                     const double *l,
                                                     const double *u,
                                                     size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    output[i] = (isinf(l[i]) && isinf(u[i])) ? 1.0 : 0.0;
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

__global__ static void gpu_vector_diff_kernel(double *output,
                                              const double *lhs,
                                              const double *rhs,
                                              size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }
    output[i] = lhs[i] - rhs[i];
}

__global__ static void gpu_sigma_diag_eval_kernel(double *output,
                                                  const double *diag_q,
                                                  const double *q_atdy,
                                                  double sigma,
                                                  size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }
    output[i] = q_atdy[i] / (1.0 + sigma * diag_q[i]);
}

__global__ static void gpu_update_diag_factors_kernel(double *fact1,
                                                      double *fact2,
                                                      double *fact,
                                                      double *fact_m,
                                                      const double *diag_q,
                                                      double sigma,
                                                      size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double temp = 1.0 + sigma * diag_q[i];
    double inv = 1.0 / temp;
    fact2[i] = inv;
    fact[i] = sigma * inv;
    fact1[i] = sigma * diag_q[i] * inv;
    fact_m[i] = sigma * sigma * inv;
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

static HPRQPStatus gpu_compute_noC_local(cublasHandle_t handle,
                                         HPRQPGPUWorkspace *workspace,
                                         const HPRQPGPUProblem *model,
                                         int *noC)
{
    HPRQPStatus status;
    double free_count = 0.0;

    if (noC == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    gpu_free_variable_mask_kernel<<<workspace->blocks_n, workspace->threads_n>>>(workspace->reduce_tmp1.values,
                                                                                  model->l.values,
                                                                                  model->u.values,
                                                                                  workspace->n);
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    status = gpu_raw_dot(handle,
                         workspace->n,
                         workspace->reduce_tmp1.values,
                         workspace->reduce_tmp1.values,
                         &free_count);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    *noC = (5.0 * free_count > 4.0 * (double)workspace->n) ? 1 : 0;
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_setup(HPRQPGPUWorkspace *workspace,
                                const HPRQPGPUProblem *model,
                                const HPRQPParams *params)
{
    cublasHandle_t handle;
    HPRQPStatus status;
    double norm_b = 0.0;
    double norm_c = 0.0;

    if (workspace == NULL || model == NULL || params == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    handle = gpu_blas_handle(workspace);
    if (handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    workspace->to_check = 1;

    status = gpu_raw_norm2(handle, workspace->n, model->c.values, &norm_c);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_compute_bound_norm2_local(handle,
                                           workspace,
                                           &model->AL,
                                           &model->AU,
                                           &norm_b);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    if (model->num_row_cones) {
        double rhs_norm;
        status = gpu_raw_norm2(handle, workspace->m, model->soc_rhs.values, &rhs_norm);
        if (status != HPRQP_STATUS_OK) return status;
        norm_b = fmax(norm_b, rhs_norm);
    }

    if (norm_b > 1e-16 && norm_c > 1e-16 && norm_b < 1e16 && norm_c < 1e16) {
        workspace->sigma = norm_b / norm_c;
    } else {
        workspace->sigma = 1.0;
    }

    if (workspace->sigma <= 0.0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_compute_noC_local(handle, workspace, model, &workspace->noC);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    status = gpu_estimate_eigenvalues(workspace, model);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    if (model->num_var_cones || model->num_row_cones) {
        workspace->noC = 0;
        /* A constant affine cone is meaningful, even if A is identically zero. */
        if (!workspace->empty_A && workspace->lambda_max_A <= 0.0) workspace->lambda_max_A = 1.0;
    }

    status = gpu_qp_metric_setup(workspace, model, params);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_row_metric_setup(workspace, model, params);
    if (status != HPRQP_STATUS_OK) return status;
    if (workspace->Q_is_diag && workspace->threads_n > 0U) {
        gpu_update_diag_factors_kernel<<<workspace->blocks_n, workspace->threads_n>>>(workspace->fact1.values,
                                                                                       workspace->fact2.values,
                                                                                       workspace->fact.values,
                                                                                       workspace->fact_M.values,
                                                                                       workspace->diag_Q.values,
                                                                                       workspace->sigma,
                                                                                       workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_compute_m_norm_value(HPRQPGPUWorkspace *gpu_workspace,
                                     const HPRQPGPUProblem *gpu_problem,
                                     double *out_value)
{
    double m1 = 0.0;
    double m2;
    double m3 = 0.0;
    double dx_sq;
    double dqw_dx;

    cublasHandle_t blas_handle = (cublasHandle_t)gpu_workspace->blas_handle;
    if (blas_handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    (void)gpu_vector_dot(blas_handle, &gpu_workspace->dx, &gpu_workspace->dx, &dx_sq);

    m2 = (1.0 / gpu_workspace->sigma) * dx_sq;

    (void)gpu_spmv_apply_q_dw(gpu_workspace, gpu_problem);

    (void)gpu_vector_dot(blas_handle, &gpu_workspace->dQw, &gpu_workspace->dx, &dqw_dx);

    m2 -= 2.0 * dqw_dx;

    if (!gpu_workspace->empty_A) {
        double dy_sq;
        double dqw_atdy;
        double atdy_dx;

        (void)gpu_vector_dot(blas_handle, &gpu_workspace->dy, &gpu_workspace->dy, &dy_sq);

        if (gpu_qp_metric_enabled(gpu_workspace) || gpu_row_metric_enabled(gpu_workspace)) {
            HPRQPStatus st=gpu_qp_metric_enabled(gpu_workspace) ? gpu_qp_metric_quadratic(gpu_workspace,&dy_sq) : gpu_row_metric_quadratic(gpu_workspace,&dy_sq);
            if(st!=HPRQP_STATUS_OK)return st;
            m1=gpu_workspace->sigma*dy_sq;
        } else m1 = gpu_workspace->sigma * gpu_workspace->lambda_max_A * dy_sq;

        (void)gpu_spmv_apply_at_dy(gpu_workspace, gpu_problem);

        (void)gpu_spmv_apply_q_atdy(gpu_workspace, gpu_problem);

        (void)gpu_vector_dot(blas_handle, &gpu_workspace->dQw, &gpu_workspace->ATdy, &dqw_atdy);

        m1 -= 2.0 * gpu_workspace->sigma * dqw_atdy;

        (void)gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->dx, &atdy_dx);

        m2 += 2.0 * atdy_dx;

        if (gpu_workspace->Q_is_diag) {
            double dqw_sq;
            double diag_term;
            unsigned int threads_n;
            unsigned int blocks_n;

            threads_n = gpu_workspace->threads_n;
            blocks_n = gpu_workspace->blocks_n;
            gpu_pointwise_multiply_kernel<<<blocks_n, threads_n>>>(gpu_workspace->diag_tmp.values,
                                                                   gpu_workspace->ATdy.values,
                                                                   gpu_workspace->fact_M.values,
                                                                   gpu_workspace->n);

            gpu_vector_dot(blas_handle, &gpu_workspace->diag_tmp, &gpu_workspace->QATdy, &diag_term);
            m3 = diag_term;

            gpu_vector_dot(blas_handle, &gpu_workspace->dQw, &gpu_workspace->dQw, &dqw_sq);
            m1 += gpu_workspace->sigma * dqw_sq;
        } else {
            double atdy_qatdy;
            double dw_dqw;

            gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->QATdy, &atdy_qatdy);
            m3 = (gpu_workspace->sigma * gpu_workspace->sigma) /
                 (1.0 + gpu_workspace->sigma * gpu_workspace->lambda_max_Q) * atdy_qatdy;

            gpu_vector_dot(blas_handle, &gpu_workspace->dw, &gpu_workspace->dQw, &dw_dqw);
            m1 += gpu_workspace->sigma * gpu_workspace->lambda_max_Q * dw_dqw;
        }
    } else if (!gpu_workspace->Q_is_diag) {
        double dw_dqw;

        gpu_vector_dot(blas_handle, &gpu_workspace->dw, &gpu_workspace->dQw, &dw_dqw);
        m1 = gpu_workspace->sigma * gpu_workspace->lambda_max_Q * dw_dqw;
    }

    m2 += m1 > 0.0 ? m1 : 0.0;
    *out_value = sqrt((m2 > 0.0 ? m2 : 0.0) + (m3 > 0.0 ? m3 : 0.0));
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_sigma_update_value(HPRQPGPUWorkspace *gpu_workspace,
                                   const HPRQPGPUProblem *gpu_problem,
                                   const HPRQPParams *params,
                                   HPRQPRestart *restart_info,
                                   const HPRQPResiduals *residuals,
                                   double *out_sigma)
{
    cublasHandle_t blas_handle;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;
    double sigma_old;
    double a = 0.0;
    double b;
    double sigma_estimation;
    double sigma_cand;
    double fact;
    double temp_1;
    double kappa = 1.0;
    double ratio_infeas_org;
    double c_term = 0.0;
    double d_term = 0.0;
    double dx_sq;
    double dy_sq = 0.0;
    double dqw_atdy = 0.0;
    double dqw_sq_or_dw_dqw = 0.0;

    if (restart_info->restart_flag < 1) {
        *out_sigma = gpu_workspace->sigma;
        return HPRQP_STATUS_OK;
    }

    blas_handle = (cublasHandle_t)gpu_workspace->blas_handle;

    threads_n = gpu_workspace->threads_n;
    blocks_n = gpu_workspace->blocks_n;
    threads_m = gpu_workspace->threads_m;
    blocks_m = gpu_workspace->blocks_m;
    sigma_old = gpu_workspace->sigma;

    gpu_vector_diff_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dx.values,
                                                    gpu_workspace->x_bar.values,
                                                    gpu_workspace->last_x.values,
                                                    gpu_workspace->n);
    gpu_vector_diff_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dw.values,
                                                    gpu_workspace->w_bar.values,
                                                    gpu_workspace->last_w.values,
                                                    gpu_workspace->n);

    gpu_vector_dot(blas_handle, &gpu_workspace->dx, &gpu_workspace->dx, &dx_sq);

    b = dx_sq;

    gpu_spmv_apply_q_dw(gpu_workspace, gpu_problem);

    if (!gpu_workspace->empty_A) {
        gpu_vector_diff_kernel<<<blocks_m, threads_m>>>(gpu_workspace->dy.values,
                                                        gpu_workspace->y_bar.values,
                                                        gpu_workspace->last_y.values,
                                                        gpu_workspace->m);
        gpu_vector_diff_kernel<<<blocks_n, threads_n>>>(gpu_workspace->ATdy.values,
                                                        gpu_workspace->ATy_bar.values,
                                                        gpu_workspace->last_ATy.values,
                                                        gpu_workspace->n);

        gpu_vector_dot(blas_handle, &gpu_workspace->dy, &gpu_workspace->dy, &dy_sq);
        if (gpu_qp_metric_enabled(gpu_workspace) || gpu_row_metric_enabled(gpu_workspace)) {
            HPRQPStatus st=gpu_qp_metric_enabled(gpu_workspace) ? gpu_qp_metric_quadratic(gpu_workspace,&dy_sq) : gpu_row_metric_quadratic(gpu_workspace,&dy_sq);
            if(st!=HPRQP_STATUS_OK)return st;
            a=dy_sq;
        } else a = gpu_workspace->lambda_max_A * dy_sq;

        gpu_vector_dot(blas_handle, &gpu_workspace->dQw, &gpu_workspace->ATdy, &dqw_atdy);
        a -= 2.0 * dqw_atdy;

        gpu_spmv_apply_q_atdy(gpu_workspace, gpu_problem);
    }

    if (gpu_workspace->Q_is_diag) {
        gpu_vector_dot(blas_handle, &gpu_workspace->dQw, &gpu_workspace->dQw, &dqw_sq_or_dw_dqw);
        a += dqw_sq_or_dw_dqw;
    } else {
        gpu_vector_dot(blas_handle, &gpu_workspace->dw, &gpu_workspace->dQw, &dqw_sq_or_dw_dqw);
        a += gpu_workspace->lambda_max_Q * dqw_sq_or_dw_dqw;
        if (!gpu_workspace->empty_A) {
            gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->QATdy, &c_term);
            d_term = gpu_workspace->lambda_max_Q;
        }
    }

    double floor = 1e-12;
    a = fmax(a, floor);
    b = fmax(b, floor);

    if (gpu_workspace->Q_is_diag && !gpu_workspace->empty_A) {
        double left = 1e-12;
        double right = 1e12;
        double phi = (sqrt(5.0) - 1.0) / 2.0;
        double x1 = right - phi * (right - left);
        double x2 = left + phi * (right - left);
        double f1;
        double f2;
        size_t iter;

        gpu_sigma_diag_eval_kernel<<<blocks_n, threads_n>>>(gpu_workspace->diag_tmp.values,
                                                            gpu_workspace->diag_Q.values,
                                                            gpu_workspace->QATdy.values,
                                                            x1,
                                                            gpu_workspace->n);
        gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->diag_tmp, &c_term);

        f1 = a * x1 + b / x1 + x1 * x1 * c_term;

        gpu_sigma_diag_eval_kernel<<<blocks_n, threads_n>>>(gpu_workspace->diag_tmp.values,
                                                            gpu_workspace->diag_Q.values,
                                                            gpu_workspace->QATdy.values,
                                                            x2,
                                                            gpu_workspace->n);
        gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->diag_tmp, &c_term);
        f2 = a * x2 + b / x2 + x2 * x2 * c_term;

        for (iter = 0; iter < 200; ++iter) {
            if (fabs(right - left) <= 1e-12 * fmax(1.0, fabs(left))) {
                break;
            }
            if (f1 > f2) {
                left = x1;
                x1 = x2;
                f1 = f2;
                x2 = left + phi * (right - left);

                gpu_sigma_diag_eval_kernel<<<blocks_n, threads_n>>>(gpu_workspace->diag_tmp.values,
                                                                    gpu_workspace->diag_Q.values,
                                                                    gpu_workspace->QATdy.values,
                                                                    x2,
                                                                    gpu_workspace->n);
                gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->diag_tmp, &c_term);
                f2 = a * x2 + b / x2 + x2 * x2 * c_term;
            } else {
                right = x2;
                x2 = x1;
                f2 = f1;
                x1 = right - phi * (right - left);

                gpu_sigma_diag_eval_kernel<<<blocks_n, threads_n>>>(gpu_workspace->diag_tmp.values,
                                                                    gpu_workspace->diag_Q.values,
                                                                    gpu_workspace->QATdy.values,
                                                                    x1,
                                                                    gpu_workspace->n);
                gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->diag_tmp, &c_term);
                f1 = a * x1 + b / x1 + x1 * x1 * c_term;
            }
        }

        sigma_estimation = f1 < f2 ? x1 : x2;
    } else if (!gpu_workspace->Q_is_diag && !gpu_workspace->empty_A) {
        c_term = c_term > 0.0 ? c_term : 0.0;
        d_term = d_term > 0.0 ? d_term : 0.0;
        sigma_estimation = golden_search_sigma_local(a, b, c_term, d_term);
    } else {
        sigma_estimation = sqrt(b / a);
    }

    sigma_estimation = clamp_double_local(sigma_estimation, 1e-12, 1e12);
    fact = (restart_info->best_gap > 0.0 && isfinite(restart_info->best_gap))
               ? exp(-0.05 * (restart_info->current_gap / restart_info->best_gap))
               : 1.0;
    sigma_cand = exp(fact * log(sigma_estimation) +
                     (1.0 - fact) * log(restart_info->best_sigma > 0.0 ? restart_info->best_sigma : sigma_old));

    temp_1 = fmax(fmin(residuals->err_Rd_org_bar, residuals->err_Rp_org_bar),
                  fmin(residuals->rel_gap_bar, restart_info->current_gap));
    if (temp_1 <= 9e-10 && residuals->err_Rp_org_bar > 0.0) {
        ratio_infeas_org = residuals->err_Rd_org_bar / residuals->err_Rp_org_bar;
        if (temp_1 > 5e-10) {
            kappa = clamp_double_local(sqrt(ratio_infeas_org), 1e-2, 100.0);
        } else {
            kappa = clamp_double_local(ratio_infeas_org, 1e-2, 100.0);
        }
    }

    gpu_workspace->sigma = clamp_double_local(kappa * sigma_cand, 1e-12, 1e12);
    if (gpu_workspace->Q_is_diag && fabs(gpu_workspace->sigma - sigma_old) > 1e-15 && threads_n > 0U) {
        gpu_update_diag_factors_kernel<<<blocks_n, threads_n>>>(gpu_workspace->fact1.values,
                                                                gpu_workspace->fact2.values,
                                                                gpu_workspace->fact.values,
                                                                gpu_workspace->fact_M.values,
                                                                gpu_workspace->diag_Q.values,
                                                                gpu_workspace->sigma,
                                                                gpu_workspace->n);
    }

    *out_sigma = gpu_workspace->sigma;
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_restart_apply_value(HPRQPGPUWorkspace *gpu_workspace,
                                    const HPRQPGPUProblem *gpu_problem)
{
    cudaMemcpy(gpu_workspace->x.values,
               gpu_workspace->x_bar.values,
               gpu_workspace->n * sizeof(double),
               cudaMemcpyDeviceToDevice);

    cudaMemcpy(gpu_workspace->w.values,
               gpu_workspace->w_bar.values,
               gpu_workspace->n * sizeof(double),
               cudaMemcpyDeviceToDevice);

    cudaMemcpy(gpu_workspace->last_x.values,
               gpu_workspace->x_bar.values,
               gpu_workspace->n * sizeof(double),
               cudaMemcpyDeviceToDevice);

    cudaMemcpy(gpu_workspace->last_w.values,
               gpu_workspace->w_bar.values,
               gpu_workspace->n * sizeof(double),
               cudaMemcpyDeviceToDevice);

    if (!gpu_workspace->empty_A) {
        cudaMemcpy(gpu_workspace->y.values,
                   gpu_workspace->y_bar.values,
                   gpu_workspace->m * sizeof(double),
                   cudaMemcpyDeviceToDevice);

        cudaMemcpy(gpu_workspace->last_y.values,
                   gpu_workspace->y_bar.values,
                   gpu_workspace->m * sizeof(double),
                   cudaMemcpyDeviceToDevice);

        gpu_spmv_apply_at_y_bar(gpu_workspace, gpu_problem);

        cudaMemcpy(gpu_workspace->ATy.values,
                   gpu_workspace->ATy_bar.values,
                   gpu_workspace->n * sizeof(double),
                   cudaMemcpyDeviceToDevice);

        cudaMemcpy(gpu_workspace->last_ATy.values,
                   gpu_workspace->ATy_bar.values,
                   gpu_workspace->n * sizeof(double),
                   cudaMemcpyDeviceToDevice);
    }

    if (gpu_workspace->noC) {
        cudaMemcpy(gpu_workspace->Qw.values,
                   gpu_workspace->Qw_bar.values,
                   gpu_workspace->n * sizeof(double),
                   cudaMemcpyDeviceToDevice);

        cudaMemcpy(gpu_workspace->last_Qw.values,
                   gpu_workspace->Qw_bar.values,
                   gpu_workspace->n * sizeof(double),
                   cudaMemcpyDeviceToDevice);
    }

    return HPRQP_STATUS_OK;
}