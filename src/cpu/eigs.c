#include "cpu/eigs.h"

#include "shared/sparse.h"
#include "shared/vector.h"

#include <math.h>
#include <stdlib.h>

static const double HPRQP_EIGENVALUE_SCALE = 1.05;

static void init_probe(double *x, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        x[i] = 1.0 + ((double)i + 1.0) / ((double)n + 1.0);
    }
}

static HPRQPStatus power_iteration_a(const HPRQPWorkspace *workspace,
                                     size_t max_iterations,
                                     double tolerance,
                                     double *lambda_max)
{
    double *z;
    double *q;
    double *at_q;
    size_t iteration;

    if (workspace->m == 0) {
        *lambda_max = 0.0;
        return HPRQP_STATUS_OK;
    }

    z = (double *)malloc(workspace->m * sizeof(double));
    q = (double *)malloc(workspace->m * sizeof(double));
    at_q = (double *)malloc(workspace->n * sizeof(double));
    if (z == NULL || q == NULL || at_q == NULL) {
        free(z);
        free(q);
        free(at_q);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    init_probe(z, workspace->m);
    *lambda_max = 1.0;

    for (iteration = 0; iteration < max_iterations; ++iteration) {
        size_t i;
        double q_norm = vec_norm2(z, workspace->m);
        double z_norm;
        double error;

        if (q_norm <= 1e-16) {
            *lambda_max = 0.0;
            break;
        }

        for (i = 0; i < workspace->m; ++i) {
            q[i] = z[i] / q_norm;
        }

        if (matrix_csr_matvec(workspace->AT, q, at_q) != HPRQP_STATUS_OK ||
            matrix_csr_matvec(workspace->A, at_q, z) != HPRQP_STATUS_OK) {
            free(z);
            free(q);
            free(at_q);
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }

        *lambda_max = vec_dot(q, z, workspace->m);
        z_norm = vec_norm2(z, workspace->m);
        for (i = 0; i < workspace->m; ++i) {
            q[i] = z[i] - (*lambda_max) * q[i];
        }
        error = vec_norm2(q, workspace->m) / (z_norm + fabs(*lambda_max) + 1e-16);
        if (error < tolerance) {
            break;
        }
    }

    free(z);
    free(q);
    free(at_q);
    if (*lambda_max < 0.0) {
        *lambda_max = 0.0;
    }
    return HPRQP_STATUS_OK;
}

static HPRQPStatus power_iteration_q(const HPRQPWorkspace *workspace,
                                     size_t max_iterations,
                                     double tolerance,
                                     double *lambda_max)
{
    double *z;
    double *q;
    size_t iteration;

    if (workspace->n == 0) {
        *lambda_max = 0.0;
        return HPRQP_STATUS_OK;
    }

    z = (double *)malloc(workspace->n * sizeof(double));
    q = (double *)malloc(workspace->n * sizeof(double));
    if (z == NULL || q == NULL) {
        free(z);
        free(q);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    init_probe(z, workspace->n);
    *lambda_max = 1.0;

    for (iteration = 0; iteration < max_iterations; ++iteration) {
        size_t i;
        double q_norm = vec_norm2(z, workspace->n);
        double z_norm;
        double error;

        if (q_norm <= 1e-16) {
            *lambda_max = 0.0;
            break;
        }

        for (i = 0; i < workspace->n; ++i) {
            q[i] = z[i] / q_norm;
        }

        if (matrix_csr_matvec(workspace->Q, q, z) != HPRQP_STATUS_OK) {
            free(z);
            free(q);
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }

        *lambda_max = vec_dot(q, z, workspace->n);
        z_norm = vec_norm2(z, workspace->n);
        for (i = 0; i < workspace->n; ++i) {
            q[i] = z[i] - (*lambda_max) * q[i];
        }
        error = vec_norm2(q, workspace->n) / (z_norm + fabs(*lambda_max) + 1e-16);
        if (error < tolerance) {
            break;
        }
    }

    free(z);
    free(q);
    if (*lambda_max < 0.0) {
        *lambda_max = 0.0;
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus estimate_eigenvalues(HPRQPWorkspace *workspace)
{
    HPRQPStatus status;

    if (workspace == NULL || workspace->Q == NULL || workspace->A == NULL || workspace->AT == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = power_iteration_a(workspace, 5000, 1e-4, &workspace->lambda_max_A);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    workspace->lambda_max_A *= HPRQP_EIGENVALUE_SCALE;

    if (workspace->Q_is_diag) {
        size_t i;
        workspace->lambda_max_Q = 0.0;
        for (i = 0; i < workspace->n; ++i) {
            if (workspace->diag_Q[i] > workspace->lambda_max_Q) {
                workspace->lambda_max_Q = workspace->diag_Q[i];
            }
        }
    } else {
        status = power_iteration_q(workspace, 5000, 1e-4, &workspace->lambda_max_Q);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        workspace->lambda_max_Q *= HPRQP_EIGENVALUE_SCALE;
    }
    return HPRQP_STATUS_OK;
}