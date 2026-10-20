#ifndef HPRQP_GPU_SPMV_OPERATION_H
#define HPRQP_GPU_SPMV_OPERATION_H

// Include before cuda_common.h: CUDA 13.3 exposes SpMVOp as an experimental API.
#include <cuda_runtime_api.h>
#if CUDART_VERSION >= 13030
#ifndef CUSPARSE_ENABLE_EXPERIMENTAL_API
#define CUSPARSE_ENABLE_EXPERIMENTAL_API
#endif
#define HPRQP_HAS_SPMVOP 1
#else
#define HPRQP_HAS_SPMVOP 0
#endif
#include <cusparse.h>
#include <cstddef>

// Owns only the plan and scratch allocation. The workspace owns the handle,
// matrix and vector descriptors; they must outlive this object.
class HPRQPSpMVOperation {
public:
    HPRQPSpMVOperation() = default;
    HPRQPSpMVOperation(const HPRQPSpMVOperation &) = delete;
    HPRQPSpMVOperation &operator=(const HPRQPSpMVOperation &) = delete;
    ~HPRQPSpMVOperation() { reset(); }

    void reset() {
#if HPRQP_HAS_SPMVOP
        if (plan_) cusparseSpMVOp_destroyPlan(plan_);
        if (descriptor_) cusparseSpMVOp_destroyDescr(descriptor_);
        plan_ = nullptr;
        descriptor_ = nullptr;
#endif
        if (buffer_) cudaFree(buffer_);
        buffer_ = nullptr;
        buffer_size_ = 0;
        matrix_ = nullptr;
        spmvop_ = false;
    }

    cusparseStatus_t prepare(cusparseHandle_t handle, cusparseSpMatDescr_t matrix,
                             cusparseDnVecDescr_t input, cusparseDnVecDescr_t output,
                             bool allow_spmvop = true) {
        reset();
        fallback_status_ = CUSPARSE_STATUS_SUCCESS;
#if HPRQP_HAS_SPMVOP
        if (allow_spmvop) {
            cusparseStatus_t status = prepare_spmvop(handle, matrix, input, output);
            if (status == CUSPARSE_STATUS_SUCCESS) {
                matrix_ = matrix;
                spmvop_ = true;
                return status;
            }
            // Release partial plans before trying the smaller legacy workspace.
            // Execution errors are never retried through another backend.
            fallback_status_ = status;
            reset();
            if (status != CUSPARSE_STATUS_ALLOC_FAILED &&
                status != CUSPARSE_STATUS_NOT_SUPPORTED &&
                status != CUSPARSE_STATUS_ARCH_MISMATCH)
                return status;
        }
#else
        (void)allow_spmvop;
#endif
        const double one = 1.0, zero = 0.0;
        cusparseStatus_t status = cusparseSpMV_bufferSize(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matrix, input, &zero, output,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &buffer_size_);
        if (status == CUSPARSE_STATUS_SUCCESS) status = allocate();
#if CUDART_VERSION >= 12040
        if (status == CUSPARSE_STATUS_SUCCESS) {
            status = cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, matrix, input, &zero, output, CUDA_R_64F,
                CUSPARSE_SPMV_CSR_ALG2, buffer_);
        }
#endif
        if (status != CUSPARSE_STATUS_SUCCESS) { reset(); return status; }
        matrix_ = matrix;
        return status;
    }

    cusparseStatus_t apply(cusparseHandle_t handle, cusparseSpMatDescr_t matrix,
                           cusparseDnVecDescr_t input, cusparseDnVecDescr_t output) const {
        if (!matrix_ || matrix != matrix_) return CUSPARSE_STATUS_NOT_INITIALIZED;
        const double one = 1.0, zero = 0.0;
#if HPRQP_HAS_SPMVOP
        if (spmvop_) return cusparseSpMVOp(handle, plan_, &one, &zero, input, output, output);
#endif
        return cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
            matrix_, input, &zero, output, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, buffer_);
    }

    bool uses_spmvop() const { return spmvop_; }
    void *buffer() const { return buffer_; }
    std::size_t buffer_size() const { return buffer_size_; }
    cusparseStatus_t fallback_status() const { return fallback_status_; }

private:
    cusparseStatus_t allocate() {
        if (!buffer_size_) return CUSPARSE_STATUS_SUCCESS;
        cudaError_t error = cudaMalloc(&buffer_, buffer_size_);
        if (error == cudaSuccess) return CUSPARSE_STATUS_SUCCESS;
        // A handled allocation failure must not poison a subsequent kernel check.
        if (error == cudaErrorMemoryAllocation) cudaGetLastError();
        return error == cudaErrorMemoryAllocation ? CUSPARSE_STATUS_ALLOC_FAILED
                                                 : CUSPARSE_STATUS_EXECUTION_FAILED;
    }
#if HPRQP_HAS_SPMVOP
    cusparseStatus_t prepare_spmvop(cusparseHandle_t handle, cusparseSpMatDescr_t matrix,
                                   cusparseDnVecDescr_t input, cusparseDnVecDescr_t output) {
        cusparseStatus_t status = cusparseSpMVOp_bufferSize(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, matrix, input, output, output,
            CUDA_R_64F, CUSPARSE_SPMVOP_ALG1, &buffer_size_);
        if (status == CUSPARSE_STATUS_SUCCESS) status = allocate();
        if (status == CUSPARSE_STATUS_SUCCESS) {
            status = cusparseSpMVOp_createDescr(handle, &descriptor_,
                CUSPARSE_OPERATION_NON_TRANSPOSE, matrix, input, output, output,
                CUDA_R_64F, CUSPARSE_SPMVOP_ALG1, buffer_);
        }
        if (status == CUSPARSE_STATUS_SUCCESS) {
            status = cusparseSpMVOp_createPlan(handle, descriptor_, &plan_, nullptr, 0);
        }
        return status;
    }
    cusparseSpMVOpDescr_t descriptor_ = nullptr;
    cusparseSpMVOpPlan_t plan_ = nullptr;
#endif
    cusparseSpMatDescr_t matrix_ = nullptr;
    void *buffer_ = nullptr;
    std::size_t buffer_size_ = 0;
    bool spmvop_ = false;
    cusparseStatus_t fallback_status_ = CUSPARSE_STATUS_SUCCESS;
};

#endif
