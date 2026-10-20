#ifndef HPRQP_SOC_COMMON_CUH
#define HPRQP_SOC_COMMON_CUH

/* Generic correctness path: one thread per cone, no dimension limit.
 * Scaled compensated accumulation avoids squared overflow and long-cone drift.
 * Cone blocks are disjoint. */
__global__ static void gpu_soc_project_kernel(double *out, const double *in,
                                              const size_t *offsets, size_t count)
{
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    size_t begin = offsets[k], end = offsets[k + 1];
    double t = in[begin], maximum = 0.0, sum = 0.0, correction = 0.0;
    for (size_t j = begin + 1; j < end; ++j) maximum = fmax(maximum, fabs(in[j]));
    if (maximum > 0.0) {
        for (size_t j = begin + 1; j < end; ++j) {
            double scaled = in[j] / maximum;
            double term = scaled * scaled - correction;
            double next = sum + term;
            correction = (next - sum) - term;
            sum = next;
        }
    }
    double norm = maximum * sqrt(sum);
    if (norm <= t) {
        for (size_t j = begin; j < end; ++j) out[j] = in[j];
    } else if (norm <= -t) {
        for (size_t j = begin; j < end; ++j) out[j] = 0.0;
    } else {
        double alpha = 0.5 + 0.5 * (t / norm);
        out[begin] = 0.5 * norm + 0.5 * t;
        for (size_t j = begin + 1; j < end; ++j) out[j] = alpha * in[j];
    }
}

__global__ static void gpu_soc_interval_kernel(double *out, const double *in,
                                               const double *lower, const double *upper,
                                               size_t length)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) out[i] = fmin(fmax(in[i], lower[i]), upper[i]);
}

static HPRQPStatus gpu_soc_project(double *out, const double *in,
                                   const double *lower, const double *upper,
                                   size_t linear_length, const size_t *offsets, size_t count)
{
    if (linear_length)
        gpu_soc_interval_kernel<<<(unsigned int)((linear_length + 255) / 256), 256>>>(out, in, lower, upper, linear_length);
    if (count)
        gpu_soc_project_kernel<<<(unsigned int)((count + 127) / 128), 128>>>(out, in, offsets, count);
    return map_cuda_status(cudaGetLastError());
}

/* Positive scalar scaling on each cone leaves the standard Lorentz cone fixed.
 * Use the conservative block maximum; other Julia aggregation policies are deferred. */
__global__ static void gpu_soc_block_scale_kernel(double *norms, const size_t *offsets, size_t count)
{
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    double value = 0.0;
    for (size_t j = offsets[k]; j < offsets[k + 1]; ++j) value = fmax(value, norms[j]);
    for (size_t j = offsets[k]; j < offsets[k + 1]; ++j) norms[j] = value;
}

static HPRQPStatus gpu_soc_scaling_blocks(const HPRQPGPUProblem *p, HPRQPGPUWorkspace *w)
{
    if (p->num_var_cones)
        gpu_soc_block_scale_kernel<<<(unsigned int)((p->num_var_cones + 127) / 128), 128>>>(
            w->reduce_tmp2.values, p->var_cone_offsets, p->num_var_cones);
    if (p->num_row_cones)
        gpu_soc_block_scale_kernel<<<(unsigned int)((p->num_row_cones + 127) / 128), 128>>>(
            w->reduce_tmp1.values, p->row_cone_offsets, p->num_row_cones);
    return map_cuda_status(cudaGetLastError());
}
#endif
