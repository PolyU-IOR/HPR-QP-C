#ifndef HPRQP_SOC_ITERATION_CUH
#define HPRQP_SOC_ITERATION_CUH
#include "soc_common.cuh"
#include "gpu/row_metric.h"
#include "gpu/qp_row_metric.h"

__global__ static void gpu_soc_raw_x_kernel(HPRQPGPUWorkspace w, HPRQPGPUProblem p)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w.n) w.tempv.values[i] = w.x.values[i] + w.sigma *
        (-w.Qw.values[i] + w.ATy.values[i] - p.c.values[i]);
}

__global__ static void gpu_soc_finish_x_kernel(HPRQPGPUWorkspace w, double a, double b)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w.n) return;
    double x = w.x.values[i], xb = w.x_bar.values[i], xh = 2.0 * xb - x;
    double f2 = w.Q_is_diag ? w.fact2.values[i] : 1.0 / (1.0 + w.sigma * w.lambda_max_Q);
    double f1 = w.Q_is_diag ? w.fact1.values[i] : 1.0 - f2;
    w.dx.values[i] = xb - x;
    w.z_bar.values[i] = (xb - w.tempv.values[i]) / w.sigma;
    w.x_hat.values[i] = xh;
    w.x.values[i] = a * w.last_x.values[i] + b * xh;
    w.w_bar.values[i] = f1 * w.w.values[i] + f2 * xh;
}

__global__ static void gpu_soc_raw_y_kernel(HPRQPGPUWorkspace w, HPRQPGPUProblem p, double tau)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w.m) w.Rp.values[i] = w.Ax.values[i] -
        (p.num_row_cones ? p.soc_rhs.values[i] : 0.0) - tau * w.y.values[i];
}

__global__ static void gpu_soc_finish_y_kernel(HPRQPGPUWorkspace w, double tau, double a, double b)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w.m) return;
    double y = w.y.values[i], yb = (w.s.values[i] - w.Rp.values[i]) / tau;
    w.dy.values[i] = yb - y;
    w.y_bar.values[i] = yb;
    w.y.values[i] = a * w.last_y.values[i] + b * (2.0 * yb - y);
}

static HPRQPStatus gpu_soc_iteration_step(HPRQPGPUWorkspace *w, const HPRQPGPUProblem *p, size_t inner)
{
    HPRQPStatus status;
    double a = 1.0 / ((double)inner + 2.0), b = 1.0 - a;
    status = gpu_spmv_apply_q_w(w, p);
    if (status != HPRQP_STATUS_OK) return status;
    gpu_soc_raw_x_kernel<<<w->blocks_n, w->threads_n>>>(*w, *p);
    status = gpu_soc_project(w->x_bar.values, w->tempv.values, p->l.values, p->u.values,
                            p->linear_n, p->var_cone_offsets, p->num_var_cones);
    if (status != HPRQP_STATUS_OK) return status;
    gpu_soc_finish_x_kernel<<<w->blocks_n, w->threads_n>>>(*w, a, b);
    status = gpu_spmv_apply_q_w_bar(w, p);
    if (status != HPRQP_STATUS_OK) return status;
    gpu_compute_tempv_kernel<<<w->blocks_n, w->threads_n>>>(w->tempv.values, w->x_hat.values,
        w->Qw.values, w->Qw_bar.values, w->sigma, w->n);
    if (!w->empty_A) {
        double tau = w->sigma * w->lambda_max_A;
        status = gpu_spmv_apply_a_tempv(w, p);
        if (status != HPRQP_STATUS_OK) return status;
        gpu_soc_raw_y_kernel<<<w->blocks_m, w->threads_m>>>(*w, *p, tau);
        status = gpu_soc_project(w->s.values, w->Rp.values, p->AL.values, p->AU.values,
                                p->linear_m, p->row_cone_offsets, p->num_row_cones);
        if (status != HPRQP_STATUS_OK) return status;
        if(gpu_qp_metric_enabled(w) || gpu_row_metric_enabled(w)) {
            status=gpu_qp_metric_enabled(w) ? gpu_qp_metric_finish_y(w,a,b) : gpu_row_metric_finish_y(w,a,b);
            if(status!=HPRQP_STATUS_OK)return status;
        } else gpu_soc_finish_y_kernel<<<w->blocks_m, w->threads_m>>>(*w, tau, a, b);
        status = gpu_spmv_apply_at_y_bar(w, p);
        if (status != HPRQP_STATUS_OK) return status;
    }
    /* Full writes on the generic cone path keep iteration slack and check state
     * available independently of printing/check intervals. Zero-cone path is untouched. */
    double fact = w->sigma / (1.0 + w->sigma * w->lambda_max_Q);
    gpu_update_w2_full_kernel<<<w->blocks_n, w->threads_n>>>(w->dw.values, w->ATdy.values,
        w->w.values, w->w_bar.values, w->last_w.values, w->ATy_bar.values,
        w->last_ATy.values, w->ATy.values, w->fact.values, fact, a, b, w->Q_is_diag, w->n);
    return map_cuda_status(cudaGetLastError());
}
#endif
