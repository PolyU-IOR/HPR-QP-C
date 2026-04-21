#include "test_support.h"

#include <cstdio>
#include <cstring>

static void seed_workspace_state(HPRQPWorkspace *workspace)
{
    workspace->x[0] = 0.4;
    workspace->x[1] = 0.8;
    workspace->w[0] = -0.2;
    workspace->w[1] = 0.3;
    workspace->last_x[0] = 0.1;
    workspace->last_x[1] = -0.05;
    workspace->last_w[0] = -0.1;
    workspace->last_w[1] = 0.07;
    workspace->last_Qw[0] = 0.01;
    workspace->last_Qw[1] = -0.02;
    workspace->last_ATy[0] = 0.03;
    workspace->last_ATy[1] = -0.04;
    if (workspace->m > 0) {
        workspace->y[0] = 0.15;
        workspace->last_y[0] = 0.02;
    }
    workspace->to_check = 1;
}

static void copy_workspace_state(HPRQPWorkspace *dst, const HPRQPWorkspace *src)
{
    std::memcpy(dst->x, src->x, src->n * sizeof(double));
    std::memcpy(dst->w, src->w, src->n * sizeof(double));
    std::memcpy(dst->last_x, src->last_x, src->n * sizeof(double));
    std::memcpy(dst->last_w, src->last_w, src->n * sizeof(double));
    std::memcpy(dst->last_Qw, src->last_Qw, src->n * sizeof(double));
    std::memcpy(dst->last_ATy, src->last_ATy, src->n * sizeof(double));
    if (src->m > 0) {
        std::memcpy(dst->y, src->y, src->m * sizeof(double));
        std::memcpy(dst->last_y, src->last_y, src->m * sizeof(double));
    }
    dst->to_check = src->to_check;
}

static int compare_workspace_state(const HPRQPWorkspace *actual,
                                   const HPRQPWorkspace *expected,
                                   const char *label,
                                   int expect_qw_bar)
{
    for (size_t i = 0; i < actual->n; ++i) {
        if (!test_nearly_equal_abs(actual->x[i], expected->x[i], 1e-12) ||
            !test_nearly_equal_abs(actual->x_bar[i], expected->x_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->w[i], expected->w[i], 1e-12) ||
            !test_nearly_equal_abs(actual->w_bar[i], expected->w_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->Qw[i], expected->Qw[i], 1e-12) ||
            !test_nearly_equal_abs(actual->ATy[i], expected->ATy[i], 1e-12) ||
            !test_nearly_equal_abs(actual->ATy_bar[i], expected->ATy_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->dx[i], expected->dx[i], 1e-12) ||
            !test_nearly_equal_abs(actual->dw[i], expected->dw[i], 1e-12)) {
            std::fprintf(stderr, "GPU iteration mismatch at index %zu for %s\n", i, label);
            std::fprintf(stderr,
                         "  x=%g expected=%g x_bar=%g expected=%g w=%g expected=%g w_bar=%g expected=%g\n",
                         actual->x[i], expected->x[i], actual->x_bar[i], expected->x_bar[i],
                         actual->w[i], expected->w[i], actual->w_bar[i], expected->w_bar[i]);
            std::fprintf(stderr,
                         "  Qw=%g expected=%g Qw_bar=%g expected=%g ATy=%g expected=%g ATy_bar=%g expected=%g\n",
                         actual->Qw[i], expected->Qw[i], actual->Qw_bar[i], expected->Qw_bar[i],
                         actual->ATy[i], expected->ATy[i], actual->ATy_bar[i], expected->ATy_bar[i]);
            std::fprintf(stderr,
                         "  dx=%g expected=%g dw=%g expected=%g\n",
                         actual->dx[i], expected->dx[i], actual->dw[i], expected->dw[i]);
            return 1;
        }
        if (expect_qw_bar && !test_nearly_equal_abs(actual->Qw_bar[i], expected->Qw_bar[i], 1e-12)) {
            std::fprintf(stderr, "GPU iteration Qw_bar mismatch at index %zu for %s\n", i, label);
            std::fprintf(stderr,
                         "  Qw_bar=%g expected=%g\n",
                         actual->Qw_bar[i], expected->Qw_bar[i]);
            return 1;
        }
    }

    for (size_t i = 0; i < actual->m; ++i) {
        if (!test_nearly_equal_abs(actual->y[i], expected->y[i], 1e-12) ||
            !test_nearly_equal_abs(actual->y_bar[i], expected->y_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->Ax[i], expected->Ax[i], 1e-12) ||
            !test_nearly_equal_abs(actual->dy[i], expected->dy[i], 1e-12)) {
            std::fprintf(stderr, "GPU dual-state mismatch at index %zu for %s\n", i, label);
            return 1;
        }
    }

    return 0;
}

typedef struct ModeCase {
    HPRQPTestModelCase case_id;
    HPRQPSpmvMode mode_q;
    HPRQPSpmvMode mode_a;
    const char *label;
    int expect_qw_bar;
} ModeCase;

static int run_mode_case(const ModeCase *test_case)
{
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPParams params;
    HPRQPWorkspace cpu_workspace;
    HPRQPWorkspace expected_workspace;
    HPRQPGPUContext context;
    HPRQPGPUProblem gpu_problem;
    HPRQPGPUWorkspace gpu_workspace;
    HPRQPStatus status;
    int expect_qw_bar;

    model_init(&model);
    workspace_init(&cpu_workspace);
    workspace_init(&expected_workspace);
    gpu_context_init(&context);
    gpu_problem_init(&gpu_problem);
    gpu_workspace_init(&gpu_workspace);
    test_params_set_no_scaling(&params);
    params.verbose = 0;
    params.spmv_mode_q = test_case->mode_q;
    params.spmv_mode_a = test_case->mode_a;
    expect_qw_bar = test_case->expect_qw_bar;

    status = test_build_model_case(test_case->case_id, &Q, &A, &model);
    if (status != HPRQP_STATUS_OK) {
        return 1;
    }

    status = workspace_allocate(&model, &params, &cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = workspace_allocate(&model, &params, &expected_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }

    seed_workspace_state(&cpu_workspace);
    copy_workspace_state(&expected_workspace, &cpu_workspace);

    status = gpu_context_open(&context, 0);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = gpu_problem_upload(&gpu_problem, &model);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = gpu_workspace_allocate(&gpu_workspace, &gpu_problem, &params);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = gpu_workspace_prepare_spmv(&gpu_workspace, &gpu_problem, 0);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    gpu_workspace.spmv_mode_q = test_case->mode_q;
    gpu_workspace.spmv_mode_a = test_case->mode_a;
    status = gpu_workspace_sync_from_cpu(&gpu_workspace, &cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }

    status = iteration_step(&expected_workspace, 0);
    if (status != HPRQP_STATUS_OK) {
        std::fprintf(stderr, "cpu iteration_step failed for %s\n", test_case->label);
        goto fail;
    }

    status = gpu_iteration_step(&gpu_workspace, &gpu_problem, 0);
    if (status != HPRQP_STATUS_OK) {
        std::fprintf(stderr, "gpu_iteration_step failed with status=%d for %s\n", (int)status, test_case->label);
        goto fail;
    }
    status = gpu_workspace_copy_main_state_to_cpu(&gpu_workspace, &cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = gpu_workspace_copy_check_state_to_cpu(&gpu_workspace, &cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }

    if (compare_workspace_state(&cpu_workspace, &expected_workspace, test_case->label, expect_qw_bar) != 0) {
        goto fail;
    }

    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&context);
    workspace_free(&expected_workspace);
    workspace_free(&cpu_workspace);
    model_free(&model);
    matrix_csr_free(&A);
    matrix_csr_free(&Q);
    return 0;

fail:
    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&context);
    workspace_free(&expected_workspace);
    workspace_free(&cpu_workspace);
    model_free(&model);
    matrix_csr_free(&A);
    matrix_csr_free(&Q);
    return 1;
}

int main(void)
{
    static const ModeCase kCases[] = {
        {HPRQP_TEST_CASE_GPU_ITERATION_STANDARD, HPRQP_SPMV_MODE_CUSPARSE, HPRQP_SPMV_MODE_CUSPARSE, "standard-cusparse", 1},
        {HPRQP_TEST_CASE_GPU_ITERATION_STANDARD, HPRQP_SPMV_MODE_CUSTOMIZED, HPRQP_SPMV_MODE_CUSTOMIZED, "standard-customized", 0},
        {HPRQP_TEST_CASE_STANDARD_NOC, HPRQP_SPMV_MODE_CUSPARSE, HPRQP_SPMV_MODE_CUSPARSE, "noc-cusparse", 1},
        {HPRQP_TEST_CASE_STANDARD_NOC, HPRQP_SPMV_MODE_CUSTOMIZED, HPRQP_SPMV_MODE_CUSTOMIZED, "noc-customized", 1},
        {HPRQP_TEST_CASE_EMPTY_A, HPRQP_SPMV_MODE_CUSPARSE, HPRQP_SPMV_MODE_CUSPARSE, "empty-a", 1},
        {HPRQP_TEST_CASE_EMPTY_A, HPRQP_SPMV_MODE_CUSTOMIZED, HPRQP_SPMV_MODE_CUSPARSE, "empty-a-custom-q", 0},
    };

    if (!gpu_backend_available()) {
        std::printf("test_gpu_iteration skipped\n");
        return 0;
    }

    for (size_t i = 0; i < sizeof(kCases) / sizeof(kCases[0]); ++i) {
        if (run_mode_case(&kCases[i]) != 0) {
            return 1;
        }
    }

    std::printf("test_gpu_iteration passed\n");
    return 0;
}