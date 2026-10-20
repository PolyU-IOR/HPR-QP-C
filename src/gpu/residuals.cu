#include "cuda_common.h"

__global__ static void gpu_finalize_rd_kernel(double *rd,
                                              double *scaled_qx_abs,
                                              double *scaled_atdy_abs,
                                              const double *qx,
                                              const double *c,
                                              const double *atdy,
                                              const double *z_bar,
                                              const double *col_norm,
                                              double c_scale,
                                              size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double scale_fact = col_norm[i] * c_scale;
    double scaled_qx = qx[i] * scale_fact;
    double scaled_atdy = atdy[i] * scale_fact;

    rd[i] = (qx[i] + c[i] - atdy[i] - z_bar[i]) * scale_fact;
    scaled_qx_abs[i] = fabs(scaled_qx);
    scaled_atdy_abs[i] = fabs(scaled_atdy);
}

__global__ static void gpu_finalize_rp_kernel(double *rp,
                                              double *scaled_ax_abs,
                                              double *s,
                                              const double *ax,
                                              const double *al,
                                              const double *au,
                                              const double *row_norm,
                                              double b_scale,
                                              size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= m) {
        return;
    }

    double ax_i = ax[i];
    double projected = ax_i < al[i] ? al[i] : (ax_i > au[i] ? au[i] : ax_i);
    double scale_fact = row_norm[i] * b_scale;

    s[i] = projected;
    rp[i] = (projected - ax_i) * scale_fact;
    scaled_ax_abs[i] = fabs(ax_i * scale_fact);
}

__global__ static void gpu_bound_violation_kernel(double *bound_scaled_abs,
                                                  const double *x_bar,
                                                  const double *l,
                                                  const double *u,
                                                  const double *col_norm,
                                                  double b_scale,
                                                  size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double lower_violation = l[i] - x_bar[i];
    double upper_violation = x_bar[i] - u[i];
    double violation = 0.0;

    if (lower_violation > 0.0) {
        violation += lower_violation;
    }
    if (upper_violation > 0.0) {
        violation += upper_violation;
    }

    bound_scaled_abs[i] = fabs(violation * (b_scale / col_norm[i]));
}

#include "soc_residuals.cuh"

HPRQPStatus gpu_compute_residuals_value(HPRQPGPUWorkspace *gpu_workspace,
                                        const HPRQPGPUProblem *gpu_problem,
                                        HPRQPResiduals *residuals,
                                        size_t iter)
{
    cublasHandle_t blas_handle;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;
    double c_dot_x;
    double x_dot_qx;
    double z_dot_x;
    double y_dot_s = 0.0;
    double max_scaled_qx = 0.0;
    double max_scaled_atdy = 0.0;
    double max_scaled_ax = 0.0;
    double bound_violation = 0.0;

    if (gpu_problem->num_var_cones || gpu_problem->num_row_cones)
        return gpu_soc_compute_residuals(gpu_workspace, gpu_problem, residuals);

    memset(residuals, 0, sizeof(*residuals));
    blas_handle = (cublasHandle_t)gpu_workspace->blas_handle;
    threads_n = gpu_workspace->threads_n;
    blocks_n = gpu_workspace->blocks_n;
    threads_m = gpu_workspace->threads_m;
    blocks_m = gpu_workspace->blocks_m;

    (void)gpu_spmv_apply_q_x_bar(gpu_workspace, gpu_problem);

    if (!gpu_workspace->empty_A) {
        (void)gpu_spmv_apply_a_x_bar(gpu_workspace, gpu_problem);
        (void)gpu_spmv_apply_at_y_bar_residual(gpu_workspace, gpu_problem);
    }

    (void)gpu_vector_dot(blas_handle, &gpu_problem->c, &gpu_workspace->x_bar, &c_dot_x);
    (void)gpu_vector_dot(blas_handle, &gpu_workspace->x_bar, &gpu_workspace->Qx, &x_dot_qx);
    (void)gpu_vector_dot(blas_handle, &gpu_workspace->z_bar, &gpu_workspace->x_bar, &z_dot_x);

    gpu_finalize_rd_kernel<<<blocks_n, threads_n>>>(gpu_workspace->Rd.values,
                                                    gpu_workspace->reduce_tmp1.values,
                                                    gpu_workspace->reduce_tmp2.values,
                                                    gpu_workspace->Qx.values,
                                                    gpu_problem->c.values,
                                                    gpu_workspace->ATdy.values,
                                                    gpu_workspace->z_bar.values,
                                                    gpu_workspace->col_norm.values,
                                                    gpu_workspace->scale_c_scale,
                                                    gpu_workspace->n);

    (void)gpu_raw_max_abs(blas_handle, gpu_workspace->n, gpu_workspace->Rd.values, &residuals->err_Rd_org_bar);
    (void)gpu_raw_max_abs(blas_handle, gpu_workspace->n, gpu_workspace->reduce_tmp1.values, &max_scaled_qx);
    (void)gpu_raw_max_abs(blas_handle, gpu_workspace->n, gpu_workspace->reduce_tmp2.values, &max_scaled_atdy);

    residuals->primal_obj_bar = gpu_workspace->scale_b_scale * gpu_workspace->scale_c_scale *
                                (c_dot_x + 0.5 * x_dot_qx) + gpu_problem->obj_constant;

    if (!gpu_workspace->empty_A) {
        gpu_finalize_rp_kernel<<<blocks_m, threads_m>>>(gpu_workspace->Rp.values,
                                                        gpu_workspace->reduce_tmp1.values,
                                                        gpu_workspace->s.values,
                                                        gpu_workspace->Ax.values,
                                                        gpu_problem->AL.values,
                                                        gpu_problem->AU.values,
                                                        gpu_workspace->row_norm.values,
                                                        gpu_workspace->scale_b_scale,
                                                        gpu_workspace->m);

        (void)gpu_vector_dot(blas_handle, &gpu_workspace->y_bar, &gpu_workspace->s, &y_dot_s);
        (void)gpu_raw_max_abs(blas_handle, gpu_workspace->m, gpu_workspace->Rp.values, &residuals->err_Rp_org_bar);
        (void)gpu_raw_max_abs(blas_handle, gpu_workspace->m, gpu_workspace->reduce_tmp1.values, &max_scaled_ax);
    }

    residuals->dual_obj_bar = gpu_workspace->scale_b_scale * gpu_workspace->scale_c_scale *
                              (-0.5 * x_dot_qx + z_dot_x + y_dot_s) + gpu_problem->obj_constant;
    residuals->rel_gap_bar = fabs(residuals->primal_obj_bar - residuals->dual_obj_bar) /
                             (1.0 + fmax(fabs(residuals->primal_obj_bar), fabs(residuals->dual_obj_bar)));
    residuals->err_Rd_org_bar /= 1.0 + fmax(gpu_workspace->scale_norm_c_org, fmax(max_scaled_atdy, max_scaled_qx));

    if (!gpu_workspace->empty_A) {
        residuals->err_Rp_org_bar /= 1.0 + fmax(gpu_workspace->scale_norm_b_org, max_scaled_ax);
    }

    if (iter == 0) {
        gpu_bound_violation_kernel<<<blocks_n, threads_n>>>(gpu_workspace->reduce_tmp1.values,
                                                            gpu_workspace->x_bar.values,
                                                            gpu_problem->l.values,
                                                            gpu_problem->u.values,
                                                            gpu_workspace->col_norm.values,
                                                            gpu_workspace->scale_b_scale,
                                                            gpu_workspace->n);
        (void)gpu_raw_max_abs(blas_handle, gpu_workspace->n, gpu_workspace->reduce_tmp1.values, &bound_violation);
        if (bound_violation > residuals->err_Rp_org_bar) {
            residuals->err_Rp_org_bar = bound_violation;
        }
    }

    residuals->KKTx_and_gap_org_bar = fmax(residuals->err_Rp_org_bar,
                                           fmax(residuals->err_Rd_org_bar, residuals->rel_gap_bar));
    residuals->is_updated = 1;
    return HPRQP_STATUS_OK;
}