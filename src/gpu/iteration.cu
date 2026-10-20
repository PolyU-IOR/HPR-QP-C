#include "packed_spmv.cuh"
#include "cuda_common.h"

static HPRQPStatus gpu_vector_copy_device_to_device(HPRQPGPUVector *dst, const HPRQPGPUVector *src)
{
    if (dst == NULL || src == NULL || dst->length != src->length) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (dst->length == 0) {
        return HPRQP_STATUS_OK;
    }

    return map_cuda_status(cudaMemcpy(dst->values,
                                      src->values,
                                      dst->length * sizeof(double),
                                      cudaMemcpyDeviceToDevice));
}

static HPRQPStatus gpu_workspace_copy_q_benchmark_state(HPRQPGPUWorkspace *dst,
                                                        const HPRQPGPUWorkspace *src)
{
    HPRQPStatus status;

    if (dst == NULL || src == NULL || dst->m != src->m || dst->n != src->n) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_vector_copy_device_to_device(&dst->z_bar, &src->z_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->x, &src->x);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->x_hat, &src->x_hat);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->x_bar, &src->x_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->w, &src->w);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->w_bar, &src->w_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->dx, &src->dx);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->Qw, &src->Qw);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->Qw_bar, &src->Qw_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->tempv, &src->tempv);
    if (status != HPRQP_STATUS_OK) return status;

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_workspace_copy_a_benchmark_state(HPRQPGPUWorkspace *dst,
                                                        const HPRQPGPUWorkspace *src)
{
    HPRQPStatus status;

    if (dst == NULL || src == NULL || dst->m != src->m || dst->n != src->n) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_vector_copy_device_to_device(&dst->y, &src->y);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->y_bar, &src->y_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->dy, &src->dy);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->s, &src->s);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->w, &src->w);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->w_bar, &src->w_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->dw, &src->dw);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->Ax, &src->Ax);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->ATy, &src->ATy);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->ATy_bar, &src->ATy_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->ATdy, &src->ATdy);
    if (status != HPRQP_STATUS_OK) return status;

    return HPRQP_STATUS_OK;
}

static __device__ __forceinline__ double interval_project_device(double value, double lower, double upper)
{
    return value < lower ? lower : (value > upper ? upper : value);
}

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

__global__ static void gpu_update_zxw1_full_kernel(double *dx,
                                                   double *z_bar,
                                                   double *w_bar,
                                                   double *x_bar,
                                                   double *x_hat,
                                                   double *x,
                                                   const double *last_x,
                                                   const double *w,
                                                   const double *Qw,
                                                   const double *ATy,
                                                   const double *c,
                                                   const double *l,
                                                   const double *u,
                                                   const double *fact1_vec,
                                                   const double *fact2_vec,
                                                   double sigma,
                                                   double fact1_scalar,
                                                   double fact2_scalar,
                                                   double halpern_fact1,
                                                   double halpern_fact2,
                                                   int q_is_diag,
                                                   size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double tmp = -Qw[i] + ATy[i] - c[i];
    double z_raw = x[i] + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x[i];
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w[i] + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w[i] + fact2_scalar * x_hat_i);

    x_bar[i] = x_bar_i;
    x_hat[i] = x_hat_i;
    z_bar[i] = (x_bar_i - z_raw) / sigma;
    dx[i] = x_bar_i - x[i];
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
}

__global__ static void gpu_update_zxw1_partial_kernel(double *w_bar,
                                                      double *x_hat,
                                                      double *x,
                                                      const double *last_x,
                                                      const double *w,
                                                      const double *Qw,
                                                      const double *ATy,
                                                      const double *c,
                                                      const double *l,
                                                      const double *u,
                                                      const double *fact1_vec,
                                                      const double *fact2_vec,
                                                      double sigma,
                                                      double fact1_scalar,
                                                      double fact2_scalar,
                                                      double halpern_fact1,
                                                      double halpern_fact2,
                                                      int q_is_diag,
                                                      size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double tmp = -Qw[i] + ATy[i] - c[i];
    double z_raw = x[i] + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x[i];
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w[i] + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w[i] + fact2_scalar * x_hat_i);

    x_hat[i] = x_hat_i;
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
}

__global__ static void gpu_update_zxw1_noC_full_kernel(double *dx,
                                                       double *dw,
                                                       double *z_bar,
                                                       double *w_bar,
                                                       double *x_bar,
                                                       double *x_hat,
                                                       double *x,
                                                       double *w,
                                                       const double *last_x,
                                                       const double *last_w,
                                                       const double *Qw,
                                                       const double *ATy,
                                                       const double *c,
                                                       const double *l,
                                                       const double *u,
                                                       const double *fact1_vec,
                                                       const double *fact2_vec,
                                                       double sigma,
                                                       double fact1_scalar,
                                                       double fact2_scalar,
                                                       double halpern_fact1,
                                                       double halpern_fact2,
                                                       int q_is_diag,
                                                       size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double x_i = x[i];
    double w_i = w[i];
    double tmp = -Qw[i] + ATy[i] - c[i];
    double z_raw = x_i + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x_i;
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w_i + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w_i + fact2_scalar * x_hat_i);
    double w_hat_i = 2.0 * w_bar_i - w_i;

    x_bar[i] = x_bar_i;
    x_hat[i] = x_hat_i;
    z_bar[i] = (x_bar_i - z_raw) / sigma;
    dx[i] = x_bar_i - x_i;
    dw[i] = w_bar_i - w_i;
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
    w[i] = halpern_fact2 * w_hat_i + halpern_fact1 * last_w[i];
}

__global__ static void gpu_update_zxw1_noC_partial_kernel(double *w_bar,
                                                          double *x_hat,
                                                          double *x,
                                                          double *w,
                                                          const double *last_x,
                                                          const double *last_w,
                                                          const double *Qw,
                                                          const double *ATy,
                                                          const double *c,
                                                          const double *l,
                                                          const double *u,
                                                          const double *fact1_vec,
                                                          const double *fact2_vec,
                                                          double sigma,
                                                          double fact1_scalar,
                                                          double fact2_scalar,
                                                          double halpern_fact1,
                                                          double halpern_fact2,
                                                          int q_is_diag,
                                                          size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double x_i = x[i];
    double w_i = w[i];
    double tmp = -Qw[i] + ATy[i] - c[i];
    double z_raw = x_i + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x_i;
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w_i + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w_i + fact2_scalar * x_hat_i);
    double w_hat_i = 2.0 * w_bar_i - w_i;

    x_hat[i] = x_hat_i;
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
    w[i] = halpern_fact2 * w_hat_i + halpern_fact1 * last_w[i];
}

__global__ static void gpu_update_zxw1_custom_full_kernel(double *dx,
                                                          double *z_bar,
                                                          double *w_bar,
                                                          double *x_bar,
                                                          double *x_hat,
                                                          double *x,
                                                          double *Qw,
                                                          const double *last_x,
                                                          const double *w,
                                                          const double *ATy,
                                                          const double *c,
                                                          const double *l,
                                                          const double *u,
                                                          const double *fact1_vec,
                                                          const double *fact2_vec,
                                                          const HPRQPGPUCsrIndex *row_ptr,
                                                          const HPRQPGPUCsrIndex *col_ind,
                                                          const double *values,
                                                          double sigma,
                                                          double fact1_scalar,
                                                          double fact2_scalar,
                                                          double halpern_fact1,
                                                          double halpern_fact2,
                                                          int q_is_diag,
                                                          size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double qw_i = csr_row_dot_device(row_ptr, col_ind, values, w, i);
    double tmp = -qw_i + ATy[i] - c[i];
    double z_raw = x[i] + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x[i];
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w[i] + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w[i] + fact2_scalar * x_hat_i);

    Qw[i] = qw_i;
    x_bar[i] = x_bar_i;
    x_hat[i] = x_hat_i;
    z_bar[i] = (x_bar_i - z_raw) / sigma;
    dx[i] = x_bar_i - x[i];
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
}

__global__ static void gpu_update_zxw1_custom_partial_kernel(double *w_bar,
                                                             double *x_hat,
                                                             double *x,
                                                             double *Qw,
                                                             const double *last_x,
                                                             const double *w,
                                                             const double *ATy,
                                                             const double *c,
                                                             const double *l,
                                                             const double *u,
                                                             const double *fact1_vec,
                                                             const double *fact2_vec,
                                                             const HPRQPGPUCsrIndex *row_ptr,
                                                             const HPRQPGPUCsrIndex *col_ind,
                                                             const double *values,
                                                             double sigma,
                                                             double fact1_scalar,
                                                             double fact2_scalar,
                                                             double halpern_fact1,
                                                             double halpern_fact2,
                                                             int q_is_diag,
                                                             size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double qw_i = csr_row_dot_device(row_ptr, col_ind, values, w, i);
    double tmp = -qw_i + ATy[i] - c[i];
    double z_raw = x[i] + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x[i];
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w[i] + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w[i] + fact2_scalar * x_hat_i);

    Qw[i] = qw_i;
    x_hat[i] = x_hat_i;
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
}

__global__ static void gpu_compute_tempv_kernel(double *tempv,
                                                const double *x_hat,
                                                const double *Qw,
                                                const double *Qw_bar,
                                                double sigma,
                                                size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }
    tempv[i] = x_hat[i] + sigma * (Qw[i] - Qw_bar[i]);
}

__global__ static void gpu_compute_tempv_custom_kernel(double *tempv,
                                                       const double *x_hat,
                                                       const double *w_bar,
                                                       const double *Qw,
                                                       const HPRQPGPUCsrIndex *row_ptr,
                                                       const HPRQPGPUCsrIndex *col_ind,
                                                       const double *values,
                                                       double sigma,
                                                       size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double qw_bar_i = csr_row_dot_device(row_ptr, col_ind, values, w_bar, i);
    tempv[i] = x_hat[i] + sigma * (Qw[i] - qw_bar_i);
}

__global__ static void gpu_compute_tempv_custom_noc_kernel(double *tempv,
                                                           double *Qw_bar,
                                                           const double *x_hat,
                                                           const double *w_bar,
                                                           const double *Qw,
                                                           const HPRQPGPUCsrIndex *row_ptr,
                                                           const HPRQPGPUCsrIndex *col_ind,
                                                           const double *values,
                                                           double sigma,
                                                           size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double qw_bar_i = csr_row_dot_device(row_ptr, col_ind, values, w_bar, i);
    Qw_bar[i] = qw_bar_i;
    tempv[i] = x_hat[i] + sigma * (Qw[i] - qw_bar_i);
}

__global__ static void gpu_finalize_qw_noC_kernel(double *tempv,
                                                  double *Qw,
                                                  const double *Qw_bar,
                                                  const double *last_Qw,
                                                  const double *x_hat,
                                                  double sigma,
                                                  double halpern_fact1,
                                                  double halpern_fact2,
                                                  size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double qw_i = Qw[i];
    double qw_bar_i = Qw_bar[i];
    double qw_hat_i = 2.0 * qw_bar_i - qw_i;

    Qw[i] = halpern_fact2 * qw_hat_i + halpern_fact1 * last_Qw[i];
    tempv[i] = x_hat[i] + sigma * (qw_i - qw_bar_i);
}

__global__ static void gpu_update_y_full_kernel(double *dy,
                                                double *y_bar,
                                                double *y,
                                                const double *last_y,
                                                double *s,
                                                const double *AL,
                                                const double *AU,
                                                const double *Ax,
                                                double fact1,
                                                double fact2,
                                                double halpern_fact1,
                                                double halpern_fact2,
                                                size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= m) {
        return;
    }

    double s_raw = Ax[i] - fact1 * y[i];
    double s_proj = interval_project_device(s_raw, AL[i], AU[i]);
    double y_bar_i = fact2 * (s_proj - s_raw);
    double y_hat_i = 2.0 * y_bar_i - y[i];

    y_bar[i] = y_bar_i;
    dy[i] = y_bar_i - y[i];
    s[i] = s_proj;
    y[i] = halpern_fact2 * y_hat_i + halpern_fact1 * last_y[i];
}

__global__ static void gpu_update_y_partial_kernel(double *y_bar,
                                                   double *y,
                                                   const double *last_y,
                                                   const double *AL,
                                                   const double *AU,
                                                   const double *Ax,
                                                   double fact1,
                                                   double fact2,
                                                   double halpern_fact1,
                                                   double halpern_fact2,
                                                   size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= m) {
        return;
    }

    double s_raw = Ax[i] - fact1 * y[i];
    double s_proj = interval_project_device(s_raw, AL[i], AU[i]);
    double y_bar_i = fact2 * (s_proj - s_raw);
    double y_hat_i = 2.0 * y_bar_i - y[i];

    y_bar[i] = y_bar_i;
    y[i] = halpern_fact2 * y_hat_i + halpern_fact1 * last_y[i];
}

__global__ static void gpu_update_y_custom_full_kernel(double *dy,
                                                       double *y_bar,
                                                       double *y,
                                                       double *s,
                                                       double *Ax,
                                                       const double *last_y,
                                                       const double *tempv,
                                                       const double *AL,
                                                       const double *AU,
                                                       const HPRQPGPUCsrIndex *row_ptr,
                                                       const HPRQPGPUCsrIndex *col_ind,
                                                       const double *values,
                                                       HPRQPGPUELL ell,
                                                       double fact1,
                                                       double fact2,
                                                       double halpern_fact1,
                                                       double halpern_fact2,
                                                       size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= m) {
        return;
    }

    double ax_i = ell.width ? packed_spmv_dot(ell, tempv, i) : csr_row_dot_device(row_ptr, col_ind, values, tempv, i);
    double s_raw = ax_i - fact1 * y[i];
    double s_proj = interval_project_device(s_raw, AL[i], AU[i]);
    double y_bar_i = fact2 * (s_proj - s_raw);
    double y_hat_i = 2.0 * y_bar_i - y[i];

    Ax[i] = ax_i;
    y_bar[i] = y_bar_i;
    dy[i] = y_bar_i - y[i];
    s[i] = s_proj;
    y[i] = halpern_fact2 * y_hat_i + halpern_fact1 * last_y[i];
}

__global__ static void gpu_update_y_custom_partial_kernel(double *y_bar,
                                                          double *y,
                                                          const double *last_y,
                                                          const double *tempv,
                                                          const double *AL,
                                                          const double *AU,
                                                          const HPRQPGPUCsrIndex *row_ptr,
                                                          const HPRQPGPUCsrIndex *col_ind,
                                                          const double *values,
                                                       HPRQPGPUELL ell,
                                                          double fact1,
                                                          double fact2,
                                                          double halpern_fact1,
                                                          double halpern_fact2,
                                                          size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= m) {
        return;
    }

    double ax_i = ell.width ? packed_spmv_dot(ell, tempv, i) : csr_row_dot_device(row_ptr, col_ind, values, tempv, i);
    double s_raw = ax_i - fact1 * y[i];
    double s_proj = interval_project_device(s_raw, AL[i], AU[i]);
    double y_bar_i = fact2 * (s_proj - s_raw);
    double y_hat_i = 2.0 * y_bar_i - y[i];

    y_bar[i] = y_bar_i;
    y[i] = halpern_fact2 * y_hat_i + halpern_fact1 * last_y[i];
}

__global__ static void gpu_update_w2_full_kernel(double *dw,
                                                 double *ATdy,
                                                 double *w,
                                                 double *w_bar,
                                                 const double *last_w,
                                                 const double *ATy_bar,
                                                 const double *last_ATy,
                                                 double *ATy,
                                                 const double *fact_vec,
                                                 double fact_scalar,
                                                 double halpern_fact1,
                                                 double halpern_fact2,
                                                 int q_is_diag,
                                                 size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double fact;
    double w_bar_new;

    if (i >= n) {
        return;
    }

    fact = q_is_diag ? fact_vec[i] : fact_scalar;
    w_bar_new = w_bar[i] + fact * (ATy_bar[i] - ATy[i]);
    dw[i] = w_bar_new - w[i];
    ATdy[i] = ATy_bar[i] - ATy[i];
    w[i] = halpern_fact1 * last_w[i] + halpern_fact2 * (2.0 * w_bar_new - w[i]);
    ATy[i] = halpern_fact1 * last_ATy[i] + halpern_fact2 * (2.0 * ATy_bar[i] - ATy[i]);
    w_bar[i] = w_bar_new;
}

__global__ static void gpu_update_w2_partial_kernel(double *w,
                                                    const double *w_bar,
                                                    const double *last_w,
                                                    const double *ATy_bar,
                                                    const double *last_ATy,
                                                    double *ATy,
                                                    const double *fact_vec,
                                                    double fact_scalar,
                                                    double halpern_fact1,
                                                    double halpern_fact2,
                                                    int q_is_diag,
                                                    size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double fact;
    double w_bar_new;

    if (i >= n) {
        return;
    }

    fact = q_is_diag ? fact_vec[i] : fact_scalar;
    w_bar_new = w_bar[i] + fact * (ATy_bar[i] - ATy[i]);
    w[i] = halpern_fact1 * last_w[i] + halpern_fact2 * (2.0 * w_bar_new - w[i]);
    ATy[i] = halpern_fact1 * last_ATy[i] + halpern_fact2 * (2.0 * ATy_bar[i] - ATy[i]);
}

__global__ static void gpu_update_w2_custom_full_kernel(double *dw,
                                                        double *ATdy,
                                                        double *w,
                                                        double *w_bar,
                                                        double *ATy,
                                                        double *ATy_bar,
                                                        const double *last_w,
                                                        const double *last_ATy,
                                                        const double *y_bar,
                                                        const double *fact_vec,
                                                        const HPRQPGPUCsrIndex *row_ptr,
                                                        const HPRQPGPUCsrIndex *col_ind,
                                                        const double *values,
                                                       HPRQPGPUELL ell,
                                                        double fact_scalar,
                                                        double halpern_fact1,
                                                        double halpern_fact2,
                                                        int q_is_diag,
                                                        size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double aty_bar_i = ell.width ? packed_spmv_dot(ell, y_bar, i) : csr_row_dot_device(row_ptr, col_ind, values, y_bar, i);
    double fact = q_is_diag ? fact_vec[i] : fact_scalar;
    double w_bar_new = w_bar[i] + fact * (aty_bar_i - ATy[i]);

    ATy_bar[i] = aty_bar_i;
    dw[i] = w_bar_new - w[i];
    ATdy[i] = aty_bar_i - ATy[i];
    w[i] = halpern_fact1 * last_w[i] + halpern_fact2 * (2.0 * w_bar_new - w[i]);
    ATy[i] = halpern_fact1 * last_ATy[i] + halpern_fact2 * (2.0 * aty_bar_i - ATy[i]);
    w_bar[i] = w_bar_new;
}

__global__ static void gpu_update_w2_custom_partial_kernel(double *w,
                                                           double *ATy,
                                                           const double *w_bar,
                                                           const double *last_w,
                                                           const double *last_ATy,
                                                           const double *y_bar,
                                                           const double *fact_vec,
                                                           const HPRQPGPUCsrIndex *row_ptr,
                                                           const HPRQPGPUCsrIndex *col_ind,
                                                           const double *values,
                                                       HPRQPGPUELL ell,
                                                           double fact_scalar,
                                                           double halpern_fact1,
                                                           double halpern_fact2,
                                                           int q_is_diag,
                                                           size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double aty_bar_i = ell.width ? packed_spmv_dot(ell, y_bar, i) : csr_row_dot_device(row_ptr, col_ind, values, y_bar, i);
    double fact = q_is_diag ? fact_vec[i] : fact_scalar;
    double w_bar_new = w_bar[i] + fact * (aty_bar_i - ATy[i]);

    w[i] = halpern_fact1 * last_w[i] + halpern_fact2 * (2.0 * w_bar_new - w[i]);
    ATy[i] = halpern_fact1 * last_ATy[i] + halpern_fact2 * (2.0 * aty_bar_i - ATy[i]);
}

static HPRQPStatus gpu_update_zxw1_step(HPRQPGPUWorkspace *gpu_workspace,
                                        const HPRQPGPUProblem *gpu_problem,
                                        double halpern_fact1,
                                        double halpern_fact2,
                                        unsigned int threads_n,
                                        unsigned int blocks_n)
{
    double fact2_scalar = 1.0 / (1.0 + gpu_workspace->sigma * gpu_workspace->lambda_max_Q);
    double fact1_scalar = 1.0 - fact2_scalar;

    if (gpu_workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        if (gpu_workspace->to_check) {
            gpu_update_zxw1_custom_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dx.values,
                                                                        gpu_workspace->z_bar.values,
                                                                        gpu_workspace->w_bar.values,
                                                                        gpu_workspace->x_bar.values,
                                                                        gpu_workspace->x_hat.values,
                                                                        gpu_workspace->x.values,
                                                                        gpu_workspace->Qw.values,
                                                                        gpu_workspace->last_x.values,
                                                                        gpu_workspace->w.values,
                                                                        gpu_workspace->ATy.values,
                                                                        gpu_problem->c.values,
                                                                        gpu_problem->l.values,
                                                                        gpu_problem->u.values,
                                                                        gpu_workspace->fact1.values,
                                                                        gpu_workspace->fact2.values,
                                                                        gpu_problem->Q.row_ptr,
                                                                        gpu_problem->Q.col_ind,
                                                                        gpu_problem->Q.values,
                                                                        gpu_workspace->sigma,
                                                                        fact1_scalar,
                                                                        fact2_scalar,
                                                                        halpern_fact1,
                                                                        halpern_fact2,
                                                                        gpu_workspace->Q_is_diag,
                                                                        gpu_workspace->n);
        } else {
            gpu_update_zxw1_custom_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w_bar.values,
                                                                           gpu_workspace->x_hat.values,
                                                                           gpu_workspace->x.values,
                                                                           gpu_workspace->Qw.values,
                                                                           gpu_workspace->last_x.values,
                                                                           gpu_workspace->w.values,
                                                                           gpu_workspace->ATy.values,
                                                                           gpu_problem->c.values,
                                                                           gpu_problem->l.values,
                                                                           gpu_problem->u.values,
                                                                           gpu_workspace->fact1.values,
                                                                           gpu_workspace->fact2.values,
                                                                           gpu_problem->Q.row_ptr,
                                                                           gpu_problem->Q.col_ind,
                                                                           gpu_problem->Q.values,
                                                                           gpu_workspace->sigma,
                                                                           fact1_scalar,
                                                                           fact2_scalar,
                                                                           halpern_fact1,
                                                                           halpern_fact2,
                                                                           gpu_workspace->Q_is_diag,
                                                                           gpu_workspace->n);
        }
    } else {
        (void)gpu_spmv_apply_q_w(gpu_workspace, gpu_problem);
        if (gpu_workspace->to_check) {
            gpu_update_zxw1_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dx.values,
                                                                 gpu_workspace->z_bar.values,
                                                                 gpu_workspace->w_bar.values,
                                                                 gpu_workspace->x_bar.values,
                                                                 gpu_workspace->x_hat.values,
                                                                 gpu_workspace->x.values,
                                                                 gpu_workspace->last_x.values,
                                                                 gpu_workspace->w.values,
                                                                 gpu_workspace->Qw.values,
                                                                 gpu_workspace->ATy.values,
                                                                 gpu_problem->c.values,
                                                                 gpu_problem->l.values,
                                                                 gpu_problem->u.values,
                                                                 gpu_workspace->fact1.values,
                                                                 gpu_workspace->fact2.values,
                                                                 gpu_workspace->sigma,
                                                                 fact1_scalar,
                                                                 fact2_scalar,
                                                                 halpern_fact1,
                                                                 halpern_fact2,
                                                                 gpu_workspace->Q_is_diag,
                                                                 gpu_workspace->n);
        } else {
            gpu_update_zxw1_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w_bar.values,
                                                                    gpu_workspace->x_hat.values,
                                                                    gpu_workspace->x.values,
                                                                    gpu_workspace->last_x.values,
                                                                    gpu_workspace->w.values,
                                                                    gpu_workspace->Qw.values,
                                                                    gpu_workspace->ATy.values,
                                                                    gpu_problem->c.values,
                                                                    gpu_problem->l.values,
                                                                    gpu_problem->u.values,
                                                                    gpu_workspace->fact1.values,
                                                                    gpu_workspace->fact2.values,
                                                                    gpu_workspace->sigma,
                                                                    fact1_scalar,
                                                                    fact2_scalar,
                                                                    halpern_fact1,
                                                                    halpern_fact2,
                                                                    gpu_workspace->Q_is_diag,
                                                                    gpu_workspace->n);
        }
    }

    if (gpu_workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        gpu_compute_tempv_custom_kernel<<<blocks_n, threads_n>>>(gpu_workspace->tempv.values,
                                                                 gpu_workspace->x_hat.values,
                                                                 gpu_workspace->w_bar.values,
                                                                 gpu_workspace->Qw.values,
                                                                 gpu_problem->Q.row_ptr,
                                                                 gpu_problem->Q.col_ind,
                                                                 gpu_problem->Q.values,
                                                                 gpu_workspace->sigma,
                                                                 gpu_workspace->n);
    } else {
        (void)gpu_spmv_apply_q_w_bar(gpu_workspace, gpu_problem);
        gpu_compute_tempv_kernel<<<blocks_n, threads_n>>>(gpu_workspace->tempv.values,
                                                          gpu_workspace->x_hat.values,
                                                          gpu_workspace->Qw.values,
                                                          gpu_workspace->Qw_bar.values,
                                                          gpu_workspace->sigma,
                                                          gpu_workspace->n);
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_update_zxw1_noC_step(HPRQPGPUWorkspace *gpu_workspace,
                                            const HPRQPGPUProblem *gpu_problem,
                                            double halpern_fact1,
                                            double halpern_fact2,
                                            unsigned int threads_n,
                                            unsigned int blocks_n)
{
    double fact2_scalar = 1.0 / (1.0 + gpu_workspace->sigma * gpu_workspace->lambda_max_Q);
    double fact1_scalar = 1.0 - fact2_scalar;

    if (!gpu_workspace->empty_A) {
        (void)gpu_spmv_apply_at_y(gpu_workspace, gpu_problem);
    } else if (gpu_workspace->ATy.length > 0) {
        cudaMemset(gpu_workspace->ATy.values, 0, gpu_workspace->ATy.length * sizeof(double));
    }

    if (gpu_workspace->to_check) {
        gpu_update_zxw1_noC_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dx.values,
                                                                 gpu_workspace->dw.values,
                                                                 gpu_workspace->z_bar.values,
                                                                 gpu_workspace->w_bar.values,
                                                                 gpu_workspace->x_bar.values,
                                                                 gpu_workspace->x_hat.values,
                                                                 gpu_workspace->x.values,
                                                                 gpu_workspace->w.values,
                                                                 gpu_workspace->last_x.values,
                                                                 gpu_workspace->last_w.values,
                                                                 gpu_workspace->Qw.values,
                                                                 gpu_workspace->ATy.values,
                                                                 gpu_problem->c.values,
                                                                 gpu_problem->l.values,
                                                                 gpu_problem->u.values,
                                                                 gpu_workspace->fact1.values,
                                                                 gpu_workspace->fact2.values,
                                                                 gpu_workspace->sigma,
                                                                 fact1_scalar,
                                                                 fact2_scalar,
                                                                 halpern_fact1,
                                                                 halpern_fact2,
                                                                 gpu_workspace->Q_is_diag,
                                                                 gpu_workspace->n);
    } else {
        gpu_update_zxw1_noC_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w_bar.values,
                                                                    gpu_workspace->x_hat.values,
                                                                    gpu_workspace->x.values,
                                                                    gpu_workspace->w.values,
                                                                    gpu_workspace->last_x.values,
                                                                    gpu_workspace->last_w.values,
                                                                    gpu_workspace->Qw.values,
                                                                    gpu_workspace->ATy.values,
                                                                    gpu_problem->c.values,
                                                                    gpu_problem->l.values,
                                                                    gpu_problem->u.values,
                                                                    gpu_workspace->fact1.values,
                                                                    gpu_workspace->fact2.values,
                                                                    gpu_workspace->sigma,
                                                                    fact1_scalar,
                                                                    fact2_scalar,
                                                                    halpern_fact1,
                                                                    halpern_fact2,
                                                                    gpu_workspace->Q_is_diag,
                                                                    gpu_workspace->n);
    }

    if (gpu_workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        gpu_compute_tempv_custom_noc_kernel<<<blocks_n, threads_n>>>(gpu_workspace->tempv.values,
                                                                     gpu_workspace->Qw_bar.values,
                                                                     gpu_workspace->x_hat.values,
                                                                     gpu_workspace->w_bar.values,
                                                                     gpu_workspace->Qw.values,
                                                                     gpu_problem->Q.row_ptr,
                                                                     gpu_problem->Q.col_ind,
                                                                     gpu_problem->Q.values,
                                                                     gpu_workspace->sigma,
                                                                     gpu_workspace->n);
    } else {
        (void)gpu_spmv_apply_q_w_bar(gpu_workspace, gpu_problem);
    }

    gpu_finalize_qw_noC_kernel<<<blocks_n, threads_n>>>(gpu_workspace->tempv.values,
                                                        gpu_workspace->Qw.values,
                                                        gpu_workspace->Qw_bar.values,
                                                        gpu_workspace->last_Qw.values,
                                                        gpu_workspace->x_hat.values,
                                                        gpu_workspace->sigma,
                                                        halpern_fact1,
                                                        halpern_fact2,
                                                        gpu_workspace->n);
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_update_y_step(HPRQPGPUWorkspace *gpu_workspace,
                                     const HPRQPGPUProblem *gpu_problem,
                                     double halpern_fact1,
                                     double halpern_fact2,
                                     unsigned int threads_m,
                                     unsigned int blocks_m)
{
    double fact1;
    double fact2;

    if (gpu_workspace->empty_A || threads_m == 0U) {
        return HPRQP_STATUS_OK;
    }

    fact1 = gpu_workspace->lambda_max_A * gpu_workspace->sigma;
    fact2 = 1.0 / fact1;

    if (gpu_problem->ell_A.width || gpu_workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        if (gpu_workspace->to_check) {
            gpu_update_y_custom_full_kernel<<<blocks_m, threads_m>>>(gpu_workspace->dy.values,
                                                                     gpu_workspace->y_bar.values,
                                                                     gpu_workspace->y.values,
                                                                     gpu_workspace->s.values,
                                                                     gpu_workspace->Ax.values,
                                                                     gpu_workspace->last_y.values,
                                                                     gpu_workspace->tempv.values,
                                                                     gpu_problem->AL.values,
                                                                     gpu_problem->AU.values,
                                                                     gpu_problem->A.row_ptr,
                                                                     gpu_problem->A.col_ind,
                                                                     gpu_problem->A.values,
                                                                         gpu_problem->ell_A,
                                                                     fact1,
                                                                     fact2,
                                                                     halpern_fact1,
                                                                     halpern_fact2,
                                                                     gpu_workspace->m);
        } else {
            gpu_update_y_custom_partial_kernel<<<blocks_m, threads_m>>>(gpu_workspace->y_bar.values,
                                                                        gpu_workspace->y.values,
                                                                        gpu_workspace->last_y.values,
                                                                        gpu_workspace->tempv.values,
                                                                        gpu_problem->AL.values,
                                                                        gpu_problem->AU.values,
                                                                        gpu_problem->A.row_ptr,
                                                                        gpu_problem->A.col_ind,
                                                                        gpu_problem->A.values,
                                                                         gpu_problem->ell_A,
                                                                        fact1,
                                                                        fact2,
                                                                        halpern_fact1,
                                                                        halpern_fact2,
                                                                        gpu_workspace->m);
        }
    } else {
        (void)gpu_spmv_apply_a_tempv(gpu_workspace, gpu_problem);
        if (gpu_workspace->to_check) {
            gpu_update_y_full_kernel<<<blocks_m, threads_m>>>(gpu_workspace->dy.values,
                                                              gpu_workspace->y_bar.values,
                                                              gpu_workspace->y.values,
                                                              gpu_workspace->last_y.values,
                                                              gpu_workspace->s.values,
                                                              gpu_problem->AL.values,
                                                              gpu_problem->AU.values,
                                                              gpu_workspace->Ax.values,
                                                              fact1,
                                                              fact2,
                                                              halpern_fact1,
                                                              halpern_fact2,
                                                              gpu_workspace->m);
        } else {
            gpu_update_y_partial_kernel<<<blocks_m, threads_m>>>(gpu_workspace->y_bar.values,
                                                                 gpu_workspace->y.values,
                                                                 gpu_workspace->last_y.values,
                                                                 gpu_problem->AL.values,
                                                                 gpu_problem->AU.values,
                                                                 gpu_workspace->Ax.values,
                                                                 fact1,
                                                                 fact2,
                                                                 halpern_fact1,
                                                                 halpern_fact2,
                                                                 gpu_workspace->m);
        }
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_update_w2_step(HPRQPGPUWorkspace *gpu_workspace,
                                      const HPRQPGPUProblem *gpu_problem,
                                      double halpern_fact1,
                                      double halpern_fact2,
                                      unsigned int threads_n,
                                      unsigned int blocks_n)
{
    double fact_scalar;

    fact_scalar = gpu_workspace->sigma / (1.0 + gpu_workspace->sigma * gpu_workspace->lambda_max_Q);

    if (gpu_workspace->empty_A) {
        if (gpu_workspace->to_check) {
            gpu_update_w2_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dw.values,
                                                               gpu_workspace->ATdy.values,
                                                               gpu_workspace->w.values,
                                                               gpu_workspace->w_bar.values,
                                                               gpu_workspace->last_w.values,
                                                               gpu_workspace->ATy_bar.values,
                                                               gpu_workspace->last_ATy.values,
                                                               gpu_workspace->ATy.values,
                                                               gpu_workspace->fact.values,
                                                               fact_scalar,
                                                               halpern_fact1,
                                                               halpern_fact2,
                                                               gpu_workspace->Q_is_diag,
                                                               gpu_workspace->n);
        } else {
            gpu_update_w2_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w.values,
                                                                  gpu_workspace->w_bar.values,
                                                                  gpu_workspace->last_w.values,
                                                                  gpu_workspace->ATy_bar.values,
                                                                  gpu_workspace->last_ATy.values,
                                                                  gpu_workspace->ATy.values,
                                                                  gpu_workspace->fact.values,
                                                                  fact_scalar,
                                                                  halpern_fact1,
                                                                  halpern_fact2,
                                                                  gpu_workspace->Q_is_diag,
                                                                  gpu_workspace->n);
        }
        return HPRQP_STATUS_OK;
    }

    if (gpu_problem->ell_A.width || gpu_workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        if (gpu_workspace->to_check) {
            gpu_update_w2_custom_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dw.values,
                                                                      gpu_workspace->ATdy.values,
                                                                      gpu_workspace->w.values,
                                                                      gpu_workspace->w_bar.values,
                                                                      gpu_workspace->ATy.values,
                                                                      gpu_workspace->ATy_bar.values,
                                                                      gpu_workspace->last_w.values,
                                                                      gpu_workspace->last_ATy.values,
                                                                      gpu_workspace->y_bar.values,
                                                                      gpu_workspace->fact.values,
                                                                      gpu_problem->AT.row_ptr,
                                                                      gpu_problem->AT.col_ind,
                                                                      gpu_problem->AT.values,
                                                                         gpu_problem->ell_AT,
                                                                      fact_scalar,
                                                                      halpern_fact1,
                                                                      halpern_fact2,
                                                                      gpu_workspace->Q_is_diag,
                                                                      gpu_workspace->n);
        } else {
            gpu_update_w2_custom_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w.values,
                                                                         gpu_workspace->ATy.values,
                                                                         gpu_workspace->w_bar.values,
                                                                         gpu_workspace->last_w.values,
                                                                         gpu_workspace->last_ATy.values,
                                                                         gpu_workspace->y_bar.values,
                                                                         gpu_workspace->fact.values,
                                                                         gpu_problem->AT.row_ptr,
                                                                         gpu_problem->AT.col_ind,
                                                                         gpu_problem->AT.values,
                                                                         gpu_problem->ell_AT,
                                                                         fact_scalar,
                                                                         halpern_fact1,
                                                                         halpern_fact2,
                                                                         gpu_workspace->Q_is_diag,
                                                                         gpu_workspace->n);
        }
    } else {
        (void)gpu_spmv_apply_at_y_bar(gpu_workspace, gpu_problem);
        if (gpu_workspace->to_check) {
            gpu_update_w2_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dw.values,
                                                               gpu_workspace->ATdy.values,
                                                               gpu_workspace->w.values,
                                                               gpu_workspace->w_bar.values,
                                                               gpu_workspace->last_w.values,
                                                               gpu_workspace->ATy_bar.values,
                                                               gpu_workspace->last_ATy.values,
                                                               gpu_workspace->ATy.values,
                                                               gpu_workspace->fact.values,
                                                               fact_scalar,
                                                               halpern_fact1,
                                                               halpern_fact2,
                                                               gpu_workspace->Q_is_diag,
                                                               gpu_workspace->n);
        } else {
            gpu_update_w2_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w.values,
                                                                  gpu_workspace->w_bar.values,
                                                                  gpu_workspace->last_w.values,
                                                                  gpu_workspace->ATy_bar.values,
                                                                  gpu_workspace->last_ATy.values,
                                                                  gpu_workspace->ATy.values,
                                                                  gpu_workspace->fact.values,
                                                                  fact_scalar,
                                                                  halpern_fact1,
                                                                  halpern_fact2,
                                                                  gpu_workspace->Q_is_diag,
                                                                  gpu_workspace->n);
        }
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus benchmark_q_mode(double *elapsed_ms,
                                    HPRQPGPUWorkspace *gpu_workspace,
                                    const HPRQPGPUProblem *gpu_problem,
                                    HPRQPGPUWorkspace *benchmark_workspace,
                                    HPRQPSpmvMode mode)
{
    cudaEvent_t start_event = NULL;
    cudaEvent_t stop_event = NULL;
    HPRQPSpmvMode saved_mode = gpu_workspace->spmv_mode_q;
    HPRQPStatus status;
    float elapsed_ms_float;
    size_t rep;
    unsigned int threads_n;
    unsigned int blocks_n;

    if (elapsed_ms == NULL || gpu_workspace == NULL || gpu_problem == NULL || benchmark_workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_workspace_copy_q_benchmark_state(benchmark_workspace, gpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    gpu_workspace->spmv_mode_q = mode;
    threads_n = gpu_workspace->n == 0 ? 0U : 256U;
    blocks_n = gpu_workspace->n == 0 ? 0U : (unsigned int)((gpu_workspace->n + threads_n - 1) / threads_n);

    if (cudaEventCreate(&start_event) != cudaSuccess || cudaEventCreate(&stop_event) != cudaSuccess) {
        if (start_event != NULL) {
            cudaEventDestroy(start_event);
        }
        if (stop_event != NULL) {
            cudaEventDestroy(stop_event);
        }
        (void)gpu_workspace_copy_q_benchmark_state(gpu_workspace, benchmark_workspace);
        gpu_workspace->spmv_mode_q = saved_mode;
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    cudaEventRecord(start_event);
    for (rep = 0; rep < kSpmvBenchmarkRepetitions; ++rep) {
        status = gpu_update_zxw1_step(gpu_workspace, gpu_problem, 0.5, 0.5, threads_n, blocks_n);
        if (status != HPRQP_STATUS_OK) {
            cudaEventDestroy(stop_event);
            cudaEventDestroy(start_event);
            (void)gpu_workspace_copy_q_benchmark_state(gpu_workspace, benchmark_workspace);
            gpu_workspace->spmv_mode_q = saved_mode;
            return status;
        }
    }
    cudaEventRecord(stop_event);
    if (cudaEventSynchronize(stop_event) != cudaSuccess || cudaEventElapsedTime(&elapsed_ms_float, start_event, stop_event) != cudaSuccess) {
        cudaEventDestroy(stop_event);
        cudaEventDestroy(start_event);
        (void)gpu_workspace_copy_q_benchmark_state(gpu_workspace, benchmark_workspace);
        gpu_workspace->spmv_mode_q = saved_mode;
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    cudaEventDestroy(stop_event);
    cudaEventDestroy(start_event);
    status = gpu_workspace_copy_q_benchmark_state(gpu_workspace, benchmark_workspace);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace->spmv_mode_q = saved_mode;
        return status;
    }
    gpu_workspace->spmv_mode_q = saved_mode;
    *elapsed_ms = (double)elapsed_ms_float;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus benchmark_a_mode(double *elapsed_ms,
                                    HPRQPGPUWorkspace *gpu_workspace,
                                    const HPRQPGPUProblem *gpu_problem,
                                    HPRQPGPUWorkspace *benchmark_workspace,
                                    HPRQPSpmvMode mode)
{
    cudaEvent_t start_event = NULL;
    cudaEvent_t stop_event = NULL;
    HPRQPSpmvMode saved_mode = gpu_workspace->spmv_mode_a;
    HPRQPStatus status;
    float elapsed_ms_float;
    size_t rep;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;

    if (elapsed_ms == NULL || gpu_workspace == NULL || gpu_problem == NULL || benchmark_workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (gpu_workspace->empty_A) {
        *elapsed_ms = 0.0;
        return HPRQP_STATUS_OK;
    }

    status = gpu_workspace_copy_a_benchmark_state(benchmark_workspace, gpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    gpu_workspace->spmv_mode_a = mode;
    threads_n = gpu_workspace->n == 0 ? 0U : 256U;
    blocks_n = gpu_workspace->n == 0 ? 0U : (unsigned int)((gpu_workspace->n + threads_n - 1) / threads_n);
    threads_m = gpu_workspace->threads_m;
    blocks_m = gpu_workspace->blocks_m;

    if (cudaEventCreate(&start_event) != cudaSuccess || cudaEventCreate(&stop_event) != cudaSuccess) {
        if (start_event != NULL) {
            cudaEventDestroy(start_event);
        }
        if (stop_event != NULL) {
            cudaEventDestroy(stop_event);
        }
        (void)gpu_workspace_copy_a_benchmark_state(gpu_workspace, benchmark_workspace);
        gpu_workspace->spmv_mode_a = saved_mode;
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    cudaEventRecord(start_event);
    for (rep = 0; rep < kSpmvBenchmarkRepetitions; ++rep) {
        status = gpu_update_y_step(gpu_workspace, gpu_problem, 0.5, 0.5, threads_m, blocks_m);
        if (status != HPRQP_STATUS_OK) {
            cudaEventDestroy(stop_event);
            cudaEventDestroy(start_event);
            (void)gpu_workspace_copy_a_benchmark_state(gpu_workspace, benchmark_workspace);
            gpu_workspace->spmv_mode_a = saved_mode;
            return status;
        }
        status = gpu_update_w2_step(gpu_workspace, gpu_problem, 0.5, 0.5, threads_n, blocks_n);
        if (status != HPRQP_STATUS_OK) {
            cudaEventDestroy(stop_event);
            cudaEventDestroy(start_event);
            (void)gpu_workspace_copy_a_benchmark_state(gpu_workspace, benchmark_workspace);
            gpu_workspace->spmv_mode_a = saved_mode;
            return status;
        }
    }
    cudaEventRecord(stop_event);
    if (cudaEventSynchronize(stop_event) != cudaSuccess || cudaEventElapsedTime(&elapsed_ms_float, start_event, stop_event) != cudaSuccess) {
        cudaEventDestroy(stop_event);
        cudaEventDestroy(start_event);
        (void)gpu_workspace_copy_a_benchmark_state(gpu_workspace, benchmark_workspace);
        gpu_workspace->spmv_mode_a = saved_mode;
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    cudaEventDestroy(stop_event);
    cudaEventDestroy(start_event);
    status = gpu_workspace_copy_a_benchmark_state(gpu_workspace, benchmark_workspace);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace->spmv_mode_a = saved_mode;
        return status;
    }
    gpu_workspace->spmv_mode_a = saved_mode;
    *elapsed_ms = (double)elapsed_ms_float;
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_select_spmv_modes(HPRQPGPUWorkspace *workspace,
                                            const HPRQPGPUProblem *model,
                                            const HPRQPParams *params)
{
    HPRQPStatus status;
    HPRQPGPUWorkspace benchmark_workspace;
    HPRQPSpmvMode selected_mode_q;
    HPRQPSpmvMode selected_mode_a;
    double q_cusparse_ms;
    double q_custom_ms;
    double a_cusparse_ms;
    double a_custom_ms;
    int benchmark_q;
    int benchmark_a;
    int need_benchmark;

    if (workspace == NULL || model == NULL || params == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    gpu_workspace_init(&benchmark_workspace);

    workspace->spmv_mode_q = params->spmv_mode_q == HPRQP_SPMV_MODE_AUTO ? HPRQP_SPMV_MODE_CUSPARSE : params->spmv_mode_q;
    workspace->spmv_mode_a = params->spmv_mode_a == HPRQP_SPMV_MODE_AUTO ? HPRQP_SPMV_MODE_CUSPARSE : params->spmv_mode_a;
    /* Existing microbenchmarks update interval-only iterates. Do not run them
     * on cone data; the first generic cone path defaults to cuSPARSE. */
    if (model->num_var_cones || model->num_row_cones) return HPRQP_STATUS_OK;
    selected_mode_q = workspace->spmv_mode_q;
    selected_mode_a = workspace->spmv_mode_a;

    benchmark_q = params->spmv_mode_q == HPRQP_SPMV_MODE_AUTO && model->Q.nnz > 0;
    benchmark_a = !workspace->empty_A && params->spmv_mode_a == HPRQP_SPMV_MODE_AUTO;
    need_benchmark = benchmark_q || benchmark_a;
    if (need_benchmark) {
        status = gpu_workspace_allocate(&benchmark_workspace, model, params);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    if (benchmark_q) {
        status = benchmark_q_mode(&q_cusparse_ms, workspace, model, &benchmark_workspace, HPRQP_SPMV_MODE_CUSPARSE);
        if (status != HPRQP_STATUS_OK) {
            gpu_workspace_free(&benchmark_workspace);
            return status;
        }
        status = benchmark_q_mode(&q_custom_ms, workspace, model, &benchmark_workspace, HPRQP_SPMV_MODE_CUSTOMIZED);
        if (status != HPRQP_STATUS_OK) {
            gpu_workspace_free(&benchmark_workspace);
            return status;
        }
        selected_mode_q = q_cusparse_ms <= q_custom_ms ? HPRQP_SPMV_MODE_CUSPARSE : HPRQP_SPMV_MODE_CUSTOMIZED;
        workspace->spmv_mode_q = selected_mode_q;
        // if (params->verbose) {
        //     std::printf("    Q: CUSPARSE=%.3fms, customized=%.3fms -> %s\n",
        //                 q_cusparse_ms,
        //                 q_custom_ms,
        //                 spmv_mode_name_local(workspace->spmv_mode_q));
        // }
    }

    if (benchmark_a) {
        status = benchmark_a_mode(&a_cusparse_ms, workspace, model, &benchmark_workspace, HPRQP_SPMV_MODE_CUSPARSE);
        if (status != HPRQP_STATUS_OK) {
            gpu_workspace_free(&benchmark_workspace);
            return status;
        }
        status = benchmark_a_mode(&a_custom_ms, workspace, model, &benchmark_workspace, HPRQP_SPMV_MODE_CUSTOMIZED);
        if (status != HPRQP_STATUS_OK) {
            gpu_workspace_free(&benchmark_workspace);
            return status;
        }
        selected_mode_a = a_cusparse_ms <= a_custom_ms ? HPRQP_SPMV_MODE_CUSPARSE : HPRQP_SPMV_MODE_CUSTOMIZED;
        workspace->spmv_mode_a = selected_mode_a;
        // if (params->verbose) {
        //     std::printf("    A: CUSPARSE=%.3fms, customized=%.3fms -> %s\n",
        //                 a_cusparse_ms,
        //                 a_custom_ms,
        //                 spmv_mode_name_local(workspace->spmv_mode_a));
        // }
    }

    if (need_benchmark) {
        gpu_workspace_free(&benchmark_workspace);
        workspace->spmv_mode_q = selected_mode_q;
        workspace->spmv_mode_a = selected_mode_a;
    }

    // if (params->verbose) {
    //     std::printf("  Selected: spmv_mode_q=%s, spmv_mode_a=%s\n",
    //                 spmv_mode_name_local(workspace->spmv_mode_q),
    //                 spmv_mode_name_local(workspace->spmv_mode_a));
    // }

    return HPRQP_STATUS_OK;
}

#include "soc_iteration.cuh"

HPRQPStatus gpu_iteration_step(HPRQPGPUWorkspace *gpu_workspace,
                               const HPRQPGPUProblem *gpu_problem,
                               size_t inner_iter)
{
    double halpern_fact1;
    double halpern_fact2;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;

    if (gpu_qp_metric_enabled(gpu_workspace) || gpu_problem->num_var_cones || gpu_problem->num_row_cones)
        return gpu_soc_iteration_step(gpu_workspace, gpu_problem, inner_iter);

    halpern_fact1 = 1.0 / ((double)inner_iter + 2.0);
    halpern_fact2 = 1.0 - halpern_fact1;
    threads_n = gpu_workspace->threads_n;
    blocks_n = gpu_workspace->blocks_n;
    threads_m = gpu_workspace->threads_m;
    blocks_m = gpu_workspace->blocks_m;

    if (gpu_workspace->noC) {
        (void)gpu_update_zxw1_noC_step(gpu_workspace,
                                       gpu_problem,
                                       halpern_fact1,
                                       halpern_fact2,
                                       threads_n,
                                       blocks_n);
    } else {
        (void)gpu_update_zxw1_step(gpu_workspace,
                                   gpu_problem,
                                   halpern_fact1,
                                   halpern_fact2,
                                   threads_n,
                                   blocks_n);
    }

    (void)gpu_update_y_step(gpu_workspace,
                            gpu_problem,
                            halpern_fact1,
                            halpern_fact2,
                            threads_m,
                            blocks_m);

    if (gpu_workspace->noC) {
        return HPRQP_STATUS_OK;
    }

    (void)gpu_update_w2_step(gpu_workspace,
                             gpu_problem,
                             halpern_fact1,
                             halpern_fact2,
                             threads_n,
                             blocks_n);

    return HPRQP_STATUS_OK;
}
