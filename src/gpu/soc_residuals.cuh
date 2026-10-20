#ifndef HPRQP_SOC_RESIDUALS_CUH
#define HPRQP_SOC_RESIDUALS_CUH
#include "soc_common.cuh"

__global__ static void gpu_soc_row_argument_kernel(HPRQPGPUWorkspace w, HPRQPGPUProblem p)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w.m) w.Rp.values[i] = w.Ax.values[i] - (p.num_row_cones ? p.soc_rhs.values[i] : 0.0);
}

__global__ static void gpu_soc_row_residual_kernel(HPRQPGPUWorkspace w)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w.m) return;
    double scale = w.row_norm.values[i] * w.scale_b_scale;
    w.Rp.values[i] = (w.reduce_tmp1.values[i] - w.Rp.values[i]) * scale;
    w.reduce_tmp2.values[i] = fabs(w.Ax.values[i] * scale);
}

__global__ static void gpu_soc_variable_residual_kernel(HPRQPGPUWorkspace w)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w.n) w.reduce_tmp1.values[i] = fabs((w.reduce_tmp1.values[i] - w.x_bar.values[i]) *
        w.scale_b_scale / w.col_norm.values[i]);
}

static HPRQPStatus gpu_soc_compute_residuals(HPRQPGPUWorkspace *w, const HPRQPGPUProblem *p,
                                             HPRQPResiduals *res)
{
    HPRQPStatus status;
    cublasHandle_t handle = (cublasHandle_t)w->blas_handle;
    double cx, xqx, zx, ys = 0.0, yb = 0.0, qmax, atmax, axmax = 0.0, violation;
    memset(res, 0, sizeof(*res));
#define SOC_TRY(call) do { status = (call); if (status != HPRQP_STATUS_OK) return status; } while (0)
    SOC_TRY(gpu_spmv_apply_q_x_bar(w, p));
    if (!w->empty_A) {
        SOC_TRY(gpu_spmv_apply_a_x_bar(w, p));
        SOC_TRY(gpu_spmv_apply_at_y_bar_residual(w, p));
        /* s is the iteration projection, paired with y_bar. Never replace it
         * with the independent feasibility projection used below. */
        SOC_TRY(gpu_vector_dot(handle, &w->y_bar, &w->s, &ys));
        if (p->num_row_cones) SOC_TRY(gpu_vector_dot(handle, &w->y_bar, &p->soc_rhs, &yb));
    }
    SOC_TRY(gpu_vector_dot(handle, &p->c, &w->x_bar, &cx));
    SOC_TRY(gpu_vector_dot(handle, &w->x_bar, &w->Qx, &xqx));
    SOC_TRY(gpu_vector_dot(handle, &w->z_bar, &w->x_bar, &zx));
    double scale = w->scale_b_scale * w->scale_c_scale;
    res->primal_obj_bar = scale * (cx + 0.5 * xqx) + p->obj_constant;
    res->dual_obj_bar = scale * (-0.5 * xqx + zx + ys + yb) + p->obj_constant;
    res->rel_gap_bar = fabs(res->primal_obj_bar - res->dual_obj_bar) /
        (1.0 + fmax(fabs(res->primal_obj_bar), fabs(res->dual_obj_bar)));
    gpu_finalize_rd_kernel<<<w->blocks_n, w->threads_n>>>(w->Rd.values,
        w->reduce_tmp1.values, w->reduce_tmp2.values, w->Qx.values, p->c.values,
        w->ATdy.values, w->z_bar.values, w->col_norm.values, w->scale_c_scale, w->n);
    SOC_TRY(gpu_raw_max_abs(handle, w->n, w->Rd.values, &res->err_Rd_org_bar));
    SOC_TRY(gpu_raw_max_abs(handle, w->n, w->reduce_tmp1.values, &qmax));
    SOC_TRY(gpu_raw_max_abs(handle, w->n, w->reduce_tmp2.values, &atmax));
    res->err_Rd_org_bar /= 1.0 + fmax(w->scale_norm_c_org, fmax(qmax, atmax));
    if (!w->empty_A) {
        gpu_soc_row_argument_kernel<<<w->blocks_m, w->threads_m>>>(*w, *p);
        SOC_TRY(gpu_soc_project(w->reduce_tmp1.values, w->Rp.values, p->AL.values, p->AU.values,
                               p->linear_m, p->row_cone_offsets, p->num_row_cones));
        gpu_soc_row_residual_kernel<<<w->blocks_m, w->threads_m>>>(*w);
        SOC_TRY(gpu_raw_max_abs(handle, w->m, w->Rp.values, &res->err_Rp_org_bar));
        SOC_TRY(gpu_raw_max_abs(handle, w->m, w->reduce_tmp2.values, &axmax));
        res->err_Rp_org_bar /= 1.0 + fmax(w->scale_norm_b_org, axmax);
    }
    /* Also audit domain membership on every cone-model check, including iteration0. */
    SOC_TRY(gpu_soc_project(w->reduce_tmp1.values, w->x_bar.values, p->l.values, p->u.values,
                           p->linear_n, p->var_cone_offsets, p->num_var_cones));
    gpu_soc_variable_residual_kernel<<<w->blocks_n, w->threads_n>>>(*w);
    SOC_TRY(gpu_raw_max_abs(handle, w->n, w->reduce_tmp1.values, &violation));
    res->err_Rp_org_bar = fmax(res->err_Rp_org_bar, violation);
    res->KKTx_and_gap_org_bar = fmax(res->err_Rp_org_bar, fmax(res->err_Rd_org_bar, res->rel_gap_bar));
    res->is_updated = 1;
#undef SOC_TRY
    return map_cuda_status(cudaGetLastError());
}
#endif
