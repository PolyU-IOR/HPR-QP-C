#ifndef HPRQP_CUDA_COMMON_H
#define HPRQP_CUDA_COMMON_H

#include "gpu/context.h"
#include "gpu/control.h"
#include "gpu/eigs.h"
#include "gpu/gpu.h"
#include "gpu/iteration.h"
#include "gpu/residuals.h"
#include "gpu/scale.h"
#include "gpu/spmv.h"
#include "gpu/workspace.h"
#include "cpu/scale.h"

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

static const double kSpmvAlpha = 1.0;
static const double kSpmvBeta = 0.0;
static const size_t kSpmvBenchmarkRepetitions = 30;

static inline HPRQPStatus convert_host_indices_to_gpu(const size_t *host_values,
                                                      size_t count,
                                                      HPRQPGPUCsrIndex *gpu_values)
{
    size_t i;

    if (count == 0) {
        return HPRQP_STATUS_OK;
    }
    if (host_values == NULL || gpu_values == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < count; ++i) {
        if (host_values[i] > (size_t)INT32_MAX) {
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
        gpu_values[i] = (HPRQPGPUCsrIndex)host_values[i];
    }

    return HPRQP_STATUS_OK;
}

static inline HPRQPStatus map_cuda_status(cudaError_t cuda_status)
{
    return cuda_status == cudaSuccess ? HPRQP_STATUS_OK : HPRQP_STATUS_NOT_IMPLEMENTED;
}

static inline HPRQPStatus map_cusparse_status(cusparseStatus_t cusparse_status)
{
    switch (cusparse_status) {
    case CUSPARSE_STATUS_SUCCESS:
        return HPRQP_STATUS_OK;
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return HPRQP_STATUS_ALLOCATION_FAILED;
    case CUSPARSE_STATUS_INVALID_VALUE:
        return HPRQP_STATUS_INVALID_ARGUMENT;
    default:
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
}

static inline HPRQPStatus map_cublas_status(cublasStatus_t cublas_status)
{
    switch (cublas_status) {
    case CUBLAS_STATUS_SUCCESS:
        return HPRQP_STATUS_OK;
    case CUBLAS_STATUS_ALLOC_FAILED:
        return HPRQP_STATUS_ALLOCATION_FAILED;
    case CUBLAS_STATUS_INVALID_VALUE:
        return HPRQP_STATUS_INVALID_ARGUMENT;
    default:
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
}

static inline const char *spmv_mode_name_local(HPRQPSpmvMode mode)
{
    switch (mode) {
    case HPRQP_SPMV_MODE_AUTO:
        return "auto";
    case HPRQP_SPMV_MODE_CUSPARSE:
        return "cusparse";
    case HPRQP_SPMV_MODE_CUSTOMIZED:
        return "customized";
    default:
        return "unknown";
    }
}

static inline double clamp_double_local(double value, double lower, double upper)
{
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static inline double sigma_objective_local(double x, double a, double b, double c, double d)
{
    return a * x + b / x + c * x * x / (1.0 + d * x);
}

static inline double golden_search_sigma_local(double a, double b, double c, double d)
{
    double left = 1e-12;
    double right = 1e12;
    double inv_phi = (sqrt(5.0) - 1.0) / 2.0;
    double inv_phi_sq = (3.0 - sqrt(5.0)) / 2.0;
    double x1 = left + inv_phi_sq * (right - left);
    double x2 = left + inv_phi * (right - left);
    double f1 = sigma_objective_local(x1, a, b, c, d);
    double f2 = sigma_objective_local(x2, a, b, c, d);
    size_t iter;

    for (iter = 0; iter < 256; ++iter) {
        if (fabs(right - left) <= 1e-13 * fmax(1.0, fmax(fabs(left), fabs(right)))) {
            break;
        }
        if (f1 > f2) {
            left = x1;
            x1 = x2;
            f1 = f2;
            x2 = left + inv_phi * (right - left);
            f2 = sigma_objective_local(x2, a, b, c, d);
        } else {
            right = x2;
            x2 = x1;
            f2 = f1;
            x1 = left + inv_phi_sq * (right - left);
            f1 = sigma_objective_local(x1, a, b, c, d);
        }
    }

    return 0.5 * (left + right);
}

static inline cublasHandle_t gpu_blas_handle(const HPRQPGPUWorkspace *workspace)
{
    return workspace == NULL ? NULL : (cublasHandle_t)workspace->blas_handle;
}

static inline HPRQPStatus gpu_raw_dot(cublasHandle_t handle,
                                      size_t length,
                                      const double *lhs,
                                      const double *rhs,
                                      double *result)
{
    cublasDdot(handle, (int)length, lhs, 1, rhs, 1, result);
    return HPRQP_STATUS_OK;
}

static inline HPRQPStatus gpu_vector_dot(cublasHandle_t handle,
                                         const HPRQPGPUVector *lhs,
                                         const HPRQPGPUVector *rhs,
                                         double *result)
{
    return gpu_raw_dot(handle, lhs->length, lhs->values, rhs->values, result);
}

static inline HPRQPStatus gpu_raw_norm2(cublasHandle_t handle,
                                        size_t length,
                                        const double *values,
                                        double *result)
{
    double norm_sq = 0.0;
    HPRQPStatus status;

    status = gpu_raw_dot(handle, length, values, values, &norm_sq);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    *result = sqrt(norm_sq > 0.0 ? norm_sq : 0.0);
    return HPRQP_STATUS_OK;
}

static inline HPRQPStatus gpu_raw_max_abs(cublasHandle_t handle,
                                          size_t length,
                                          const double *values,
                                          double *result)
{
    int index = 0;
    double value = 0.0;

    cublasIdamax(handle, (int)length, values, 1, &index);
    cudaMemcpy(&value, values + (size_t)(index - 1), sizeof(double), cudaMemcpyDeviceToHost);

    *result = fabs(value);
    return HPRQP_STATUS_OK;
}

#endif