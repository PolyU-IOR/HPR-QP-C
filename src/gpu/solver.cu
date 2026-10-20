#include "internal/hprqp_internal.h"
#include "gram_policy.h"
#include "gpu/row_metric.h"
#include "gpu/qp_row_metric.h"
#include <chrono>

#include "solver_backend.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

typedef struct HPRQPGPUSolveContext {
    HPRQPGPUProblem *gpu_problem;
    HPRQPGPUWorkspace *gpu_workspace;
} HPRQPGPUSolveContext;

static HPRQPStatus gpu_compute_residuals(HPRQPSolverBackendContext *context,
                                         HPRQPResiduals *residuals,
                                         size_t iter)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;

    return gpu_compute_residuals_value(gpu_context->gpu_workspace,
                                       gpu_context->gpu_problem,
                                       residuals,
                                       iter);
}

static HPRQPStatus gpu_update_sigma(HPRQPSolverBackendContext *context,
                                    HPRQPRestart *restart_info,
                                    const HPRQPResiduals *residuals)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;
    HPRQPWorkspace *workspace = context->workspace;
    double sigma_value;
    if (restart_info->restart_flag < 1) {
        return HPRQP_STATUS_OK;
    }

    HPRQPStatus status = gpu_sigma_update_value(gpu_context->gpu_workspace,
                                    gpu_context->gpu_problem,
                                    context->params,
                                    restart_info,
                                    residuals,
                                    &sigma_value);
    if (status != HPRQP_STATUS_OK) return status;

    workspace->sigma = sigma_value;
    workspace->lambda_max_A = gpu_context->gpu_workspace->lambda_max_A;
    workspace->lambda_max_Q = gpu_context->gpu_workspace->lambda_max_Q;
    workspace->Q_is_diag = gpu_context->gpu_workspace->Q_is_diag;

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_copy_output_iterates(HPRQPSolverBackendContext *context,
                                            double *x_bar,
                                            double *w_bar,
                                            double *z_bar,
                                            double *y_bar)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;
    HPRQPGPUWorkspace *gpu_workspace;


    gpu_workspace = gpu_context->gpu_workspace;
    return gpu_workspace_copy_output_unscaled_to_host(gpu_workspace, x_bar, w_bar, z_bar, y_bar);
}

static HPRQPStatus gpu_restart_apply(HPRQPSolverBackendContext *context,
                                     HPRQPRestart *restart_info)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;

    if (restart_info->restart_flag <= 0) {
        return HPRQP_STATUS_OK;
    }

    HPRQPStatus status = gpu_restart_apply_value(gpu_context->gpu_workspace,
                                  gpu_context->gpu_problem);
    if (status != HPRQP_STATUS_OK) return status;

    restart_info->last_gap = restart_info->current_gap;
    restart_info->save_gap = INFINITY;
    restart_info->times += 1;
    restart_info->inner = 0;
    return HPRQP_STATUS_OK;
}

static double gpu_compute_m_norm(HPRQPSolverBackendContext *context)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;
    double value;

    return gpu_compute_m_norm_value(gpu_context->gpu_workspace,
                                    gpu_context->gpu_problem,
                                    &value) == HPRQP_STATUS_OK
               ? value
               : INFINITY;
}

static HPRQPStatus gpu_iteration_step_backend(HPRQPSolverBackendContext *context, size_t inner_iter)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;

    gpu_context->gpu_workspace->to_check = context->workspace->to_check;

    return gpu_iteration_step(gpu_context->gpu_workspace,
                              gpu_context->gpu_problem,
                              inner_iter);
}

static const HPRQPSolverBackendOps GPU_BACKEND_OPS = {
    gpu_compute_residuals,
    gpu_copy_output_iterates,
    gpu_update_sigma,
    gpu_restart_apply,
    gpu_compute_m_norm,
    gpu_iteration_step_backend,
    1,
};

HPRQPStatus solve_gpu(const HPRQPModel *model, const HPRQPParams *params, HPRQPResults *results)
{
    auto begin = std::chrono::steady_clock::now();
    HPRQPParams selected_params;
    HPRQPScaling scaling;
    HPRQPWorkspace workspace;
    HPRQPGPUContext context;
    HPRQPGPUProblem gpu_problem;
    HPRQPGPUWorkspace gpu_workspace;
    HPRQPGPUSolveContext gpu_solve_context;
    HPRQPSolverBackendContext backend_context;
    HPRQPStatus status;
    HPRQPStatus return_status;

    if (model == NULL || params == NULL || results == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (params->gram_mode < HPRQP_GRAM_OFF || params->gram_mode > HPRQP_GRAM_REQUEST)
        return HPRQP_STATUS_INVALID_ARGUMENT;
    selected_params = *params;
    if (params->gram_mode == HPRQP_GRAM_AUTO) {
        int matched = hprqp_gram_auto_eligible(model);
        selected_params.gram_mode = matched ? HPRQP_GRAM_REQUEST : HPRQP_GRAM_OFF;
        if (params->verbose) fprintf(stderr, "GRAM_AUTO %s\n", matched ? "spatial_structure" : "declined_structure");
    }
    params = &selected_params;
    scaling_init(&scaling);
    workspace_init(&workspace);
    results_init(results);
    gpu_context_init(&context);
    gpu_problem_init(&gpu_problem);
    gpu_workspace_init(&gpu_workspace);
    return_status = HPRQP_STATUS_OK;

    if (params->verbose) {
        print_model_info(model);
        print_params_info(params);
    }

    status = gpu_context_open(&context, params->device_number);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    status = gpu_problem_upload(&gpu_problem, model);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    status = gpu_workspace_allocate(&gpu_workspace, &gpu_problem, params);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    status = gpu_problem_scale(&gpu_problem, &gpu_workspace, params, &scaling);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    status = gpu_workspace_prepare_spmv(&gpu_workspace, &gpu_problem, params->verbose);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    status = gpu_workspace_setup(&gpu_workspace, &gpu_problem, params);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    workspace.m = gpu_workspace.m;
    workspace.n = gpu_workspace.n;
    workspace.sigma = gpu_workspace.sigma;
    workspace.lambda_max_A = gpu_workspace.lambda_max_A;
    workspace.lambda_max_Q = gpu_workspace.lambda_max_Q;
    workspace.Q_is_diag = gpu_workspace.Q_is_diag;
    workspace.noC = gpu_workspace.noC;
    workspace.to_check = 1;

    status = gpu_workspace_select_spmv_modes(&gpu_workspace, &gpu_problem, params);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }

    gpu_solve_context.gpu_problem = &gpu_problem;
    gpu_solve_context.gpu_workspace = &gpu_workspace;

    backend_context.workspace = &workspace;
    backend_context.model = model;
    backend_context.scaling = &scaling;
    backend_context.params = params;
    backend_context.data = &gpu_solve_context;

    return_status = solve_with_backend(&backend_context, &GPU_BACKEND_OPS, "GPU mode", results);

cleanup:
    results->gram_active = gpu_row_metric_enabled(&gpu_workspace) || gpu_qp_metric_enabled(&gpu_workspace);
    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&context);
    workspace_free(&workspace);
    scaling_free(&scaling);
    results->total_time = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    return return_status;
}