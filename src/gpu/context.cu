#include "packed_spmv.cuh"
#include "gpu/gpu.h"

#include "cuda_common.h"
#include "shared/build.h"

int gpu_backend_available(void)
{
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void gpu_context_init(HPRQPGPUContext *context)
{
    if (context == NULL) {
        return;
    }
    std::memset(context, 0, sizeof(*context));
}

void gpu_context_free(HPRQPGPUContext *context)
{
    gpu_context_init(context);
}

HPRQPStatus gpu_context_open(HPRQPGPUContext *context, int device_number)
{
    int device_count = 0;

    if (context == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    if (device_number < 0 || device_number >= device_count) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device_number) != cudaSuccess) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    context->requested_device = device_number;
    context->initialized = 1;
    return HPRQP_STATUS_OK;
}

void gpu_vector_init(HPRQPGPUVector *vector)
{
    if (vector == NULL) {
        return;
    }
    std::memset(vector, 0, sizeof(*vector));
}

void gpu_vector_free(HPRQPGPUVector *vector)
{
    if (vector == NULL) {
        return;
    }
    if (vector->values != NULL) {
        cudaFree(vector->values);
    }
    gpu_vector_init(vector);
}

HPRQPStatus gpu_vector_alloc(HPRQPGPUVector *vector, size_t length)
{
    if (vector == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    gpu_vector_init(vector);
    vector->length = length;
    if (length == 0) {
        vector->allocated = 1;
        return HPRQP_STATUS_OK;
    }
    if (cudaMalloc((void **)&vector->values, length * sizeof(double)) != cudaSuccess) {
        gpu_vector_free(vector);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    if (cudaMemset(vector->values, 0, length * sizeof(double)) != cudaSuccess) {
        gpu_vector_free(vector);
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    vector->allocated = 1;
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_vector_copy_from_host(HPRQPGPUVector *vector, const double *host_values, size_t length)
{
    if (vector == NULL || (length > 0 && host_values == NULL) || vector->length != length) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (length == 0) {
        return HPRQP_STATUS_OK;
    }
    return map_cuda_status(cudaMemcpy(vector->values, host_values, length * sizeof(double), cudaMemcpyHostToDevice));
}

HPRQPStatus gpu_vector_copy_to_host(const HPRQPGPUVector *vector, double *host_values, size_t length)
{
    if (vector == NULL || (length > 0 && host_values == NULL) || vector->length != length) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (length == 0) {
        return HPRQP_STATUS_OK;
    }
    return map_cuda_status(cudaMemcpy(host_values, vector->values, length * sizeof(double), cudaMemcpyDeviceToHost));
}

void gpu_matrix_init(HPRQPGPUMatrixCSR *matrix)
{
    if (matrix == NULL) {
        return;
    }
    std::memset(matrix, 0, sizeof(*matrix));
}

void gpu_matrix_free(HPRQPGPUMatrixCSR *matrix)
{
    if (matrix == NULL) {
        return;
    }
    if (matrix->row_ptr != NULL) {
        cudaFree(matrix->row_ptr);
    }
    if (matrix->col_ind != NULL) {
        cudaFree(matrix->col_ind);
    }
    if (matrix->values != NULL) {
        cudaFree(matrix->values);
    }
    gpu_matrix_init(matrix);
}

HPRQPStatus gpu_matrix_upload(HPRQPGPUMatrixCSR *gpu_matrix, const HPRQPMatrixCSR *cpu_matrix)
{
    HPRQPGPUCsrIndex *host_row_ptr = NULL;
    HPRQPGPUCsrIndex *host_col_ind = NULL;
    HPRQPStatus status;

    if (gpu_matrix == NULL || cpu_matrix == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    gpu_matrix_init(gpu_matrix);
    gpu_matrix->nrow = cpu_matrix->nrow;
    gpu_matrix->ncol = cpu_matrix->ncol;
    gpu_matrix->nnz = cpu_matrix->nnz;

    host_row_ptr = (HPRQPGPUCsrIndex *)malloc((cpu_matrix->nrow + 1) * sizeof(HPRQPGPUCsrIndex));
    if (host_row_ptr == NULL) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    status = convert_host_indices_to_gpu(cpu_matrix->row_ptr, cpu_matrix->nrow + 1, host_row_ptr);
    if (status != HPRQP_STATUS_OK) {
        free(host_row_ptr);
        return status;
    }

    if (cpu_matrix->nnz > 0) {
        host_col_ind = (HPRQPGPUCsrIndex *)malloc(cpu_matrix->nnz * sizeof(HPRQPGPUCsrIndex));
        if (host_col_ind == NULL) {
            free(host_row_ptr);
            return HPRQP_STATUS_ALLOCATION_FAILED;
        }
        status = convert_host_indices_to_gpu(cpu_matrix->col_ind, cpu_matrix->nnz, host_col_ind);
        if (status != HPRQP_STATUS_OK) {
            free(host_col_ind);
            free(host_row_ptr);
            return status;
        }
    }

    if (cudaMalloc((void **)&gpu_matrix->row_ptr,
                   (cpu_matrix->nrow + 1) * sizeof(HPRQPGPUCsrIndex)) != cudaSuccess) {
        free(host_col_ind);
        free(host_row_ptr);
        gpu_matrix_free(gpu_matrix);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    if (cpu_matrix->nnz > 0) {
        if (cudaMalloc((void **)&gpu_matrix->col_ind, cpu_matrix->nnz * sizeof(HPRQPGPUCsrIndex)) != cudaSuccess ||
            cudaMalloc((void **)&gpu_matrix->values, cpu_matrix->nnz * sizeof(double)) != cudaSuccess) {
            free(host_col_ind);
            free(host_row_ptr);
            gpu_matrix_free(gpu_matrix);
            return HPRQP_STATUS_ALLOCATION_FAILED;
        }
    }

    if (map_cuda_status(cudaMemcpy(gpu_matrix->row_ptr,
                                   host_row_ptr,
                                   (cpu_matrix->nrow + 1) * sizeof(HPRQPGPUCsrIndex),
                                   cudaMemcpyHostToDevice)) != HPRQP_STATUS_OK) {
        free(host_col_ind);
        free(host_row_ptr);
        gpu_matrix_free(gpu_matrix);
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    if (cpu_matrix->nnz > 0) {
        if (map_cuda_status(cudaMemcpy(gpu_matrix->col_ind,
                                       host_col_ind,
                                       cpu_matrix->nnz * sizeof(HPRQPGPUCsrIndex),
                                       cudaMemcpyHostToDevice)) != HPRQP_STATUS_OK ||
            map_cuda_status(cudaMemcpy(gpu_matrix->values,
                                       cpu_matrix->values,
                                       cpu_matrix->nnz * sizeof(double),
                                       cudaMemcpyHostToDevice)) != HPRQP_STATUS_OK) {
            free(host_col_ind);
            free(host_row_ptr);
            gpu_matrix_free(gpu_matrix);
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
    }

    free(host_col_ind);
    free(host_row_ptr);
    gpu_matrix->uploaded = 1;
    return HPRQP_STATUS_OK;
}

void gpu_problem_init(HPRQPGPUProblem *model)
{
    if (model == NULL) {
        return;
    }
    std::memset(model, 0, sizeof(*model));
}

void gpu_problem_free(HPRQPGPUProblem *model)
{
    if (model) { packed_spmv_free(&model->ell_A); packed_spmv_free(&model->ell_AT); }
    if (model == NULL) {
        return;
    }
    gpu_matrix_free(&model->Q);
    gpu_matrix_free(&model->A);
    gpu_matrix_free(&model->AT);
    gpu_vector_free(&model->c);
    gpu_vector_free(&model->AL);
    gpu_vector_free(&model->AU);
    gpu_vector_free(&model->l);
    gpu_vector_free(&model->u);
    gpu_vector_free(&model->soc_rhs);
    cudaFree(model->var_cone_offsets);
    cudaFree(model->row_cone_offsets);
    gpu_problem_init(model);
}

HPRQPStatus gpu_problem_upload(HPRQPGPUProblem *gpu_problem, const HPRQPModel *cpu_problem)
{
    HPRQPStatus status;

    if (gpu_problem == NULL || cpu_problem == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (model_validate_soc(cpu_problem) != HPRQP_STATUS_OK || cpu_problem->Q.ncol == 0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    gpu_problem->m = cpu_problem->A.nrow;
    gpu_problem->n = cpu_problem->Q.ncol;
    gpu_problem->nnz_q = cpu_problem->Q.nnz;
    gpu_problem->nnz_a = cpu_problem->A.nnz;
    gpu_problem->empty_A = cpu_problem->A.nrow == 0;
    gpu_problem->obj_constant = cpu_problem->obj_constant;

    status = gpu_matrix_upload(&gpu_problem->Q, &cpu_problem->Q);
    if (status != HPRQP_STATUS_OK) {
        gpu_problem_free(gpu_problem);
        return status;
    }
    status = gpu_matrix_upload(&gpu_problem->A, &cpu_problem->A);
    if (status != HPRQP_STATUS_OK) {
        gpu_problem_free(gpu_problem);
        return status;
    }
    status = gpu_matrix_upload(&gpu_problem->AT, &cpu_problem->AT);
    if (status != HPRQP_STATUS_OK) {
        gpu_problem_free(gpu_problem);
        return status;
    }

    status = gpu_vector_alloc(&gpu_problem->c, cpu_problem->Q.ncol);
    if (status != HPRQP_STATUS_OK) {
        gpu_problem_free(gpu_problem);
        return status;
    }
    status = gpu_vector_alloc(&gpu_problem->AL, cpu_problem->A.nrow);
    if (status != HPRQP_STATUS_OK) {
        gpu_problem_free(gpu_problem);
        return status;
    }
    status = gpu_vector_alloc(&gpu_problem->AU, cpu_problem->A.nrow);
    if (status != HPRQP_STATUS_OK) {
        gpu_problem_free(gpu_problem);
        return status;
    }
    status = gpu_vector_alloc(&gpu_problem->l, cpu_problem->Q.ncol);
    if (status != HPRQP_STATUS_OK) {
        gpu_problem_free(gpu_problem);
        return status;
    }
    status = gpu_vector_alloc(&gpu_problem->u, cpu_problem->Q.ncol);
    if (status != HPRQP_STATUS_OK) {
        gpu_problem_free(gpu_problem);
        return status;
    }

    if (cpu_problem->Q.ncol > 0) {
        if (cudaMemcpy(gpu_problem->c.values, cpu_problem->c, cpu_problem->Q.ncol * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(gpu_problem->l.values, cpu_problem->l, cpu_problem->Q.ncol * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(gpu_problem->u.values, cpu_problem->u, cpu_problem->Q.ncol * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
            gpu_problem_free(gpu_problem);
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
    }
    if (cpu_problem->A.nrow > 0) {
        if (cudaMemcpy(gpu_problem->AL.values, cpu_problem->AL, cpu_problem->A.nrow * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(gpu_problem->AU.values, cpu_problem->AU, cpu_problem->A.nrow * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
            gpu_problem_free(gpu_problem);
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
    }

    gpu_problem->linear_n = cpu_problem->num_var_cones ? cpu_problem->var_cone_offsets[0] : gpu_problem->n;
    gpu_problem->linear_m = cpu_problem->num_row_cones ? cpu_problem->row_cone_offsets[0] : gpu_problem->m;
    gpu_problem->num_var_cones = cpu_problem->num_var_cones;
    gpu_problem->num_row_cones = cpu_problem->num_row_cones;
    if (cpu_problem->num_var_cones) {
        size_t bytes = (cpu_problem->num_var_cones + 1) * sizeof(size_t);
        if (cudaMalloc((void **)&gpu_problem->var_cone_offsets, bytes) != cudaSuccess ||
            cudaMemcpy(gpu_problem->var_cone_offsets, cpu_problem->var_cone_offsets, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            gpu_problem_free(gpu_problem); return HPRQP_STATUS_ALLOCATION_FAILED;
        }
    }
    if (cpu_problem->num_row_cones) {
        size_t bytes = (cpu_problem->num_row_cones + 1) * sizeof(size_t);
        if (cudaMalloc((void **)&gpu_problem->row_cone_offsets, bytes) != cudaSuccess ||
            cudaMemcpy(gpu_problem->row_cone_offsets, cpu_problem->row_cone_offsets, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            gpu_problem_free(gpu_problem); return HPRQP_STATUS_ALLOCATION_FAILED;
        }
        status = gpu_vector_alloc(&gpu_problem->soc_rhs, gpu_problem->m);
        if (status == HPRQP_STATUS_OK)
            status = gpu_vector_copy_from_host(&gpu_problem->soc_rhs, cpu_problem->soc_rhs, gpu_problem->m);
        if (status != HPRQP_STATUS_OK) { gpu_problem_free(gpu_problem); return status; }
    }

    gpu_problem->uploaded = 1;
    // Optional iteration-only access; CSR remains authoritative for all products.
    const char *packed = getenv("HPRQP_PACKED_SPMV");
    if ((!packed || packed[0] != '0') && !cpu_problem->num_var_cones && !cpu_problem->num_row_cones) {
        packed_spmv_prepare_pair(gpu_problem, cpu_problem);
    }
    return HPRQP_STATUS_OK;
}
