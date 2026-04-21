#include "hprqp.h"

#include <math.h>
#include <stdio.h>

static int nearly_equal(double a, double b)
{
    return fabs(a - b) < 1e-12;
}

int main(void)
{
    HPRQPModel model;
    HPRQPStatus status;
    const char *mps_path = HPRQP_SOURCE_DIR "/tests/data/simple_qp.mps";

    model_init(&model);
    status = build_from_mps(mps_path, &model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "build_from_mps failed with status %d\n", status);
        return 1;
    }

    if (model.A.nrow != 2 || model.A.ncol != 2 || model.A.nnz != 4) {
        fprintf(stderr, "A dimensions are wrong\n");
        model_free(&model);
        return 1;
    }

    if (model.Q.nrow != 2 || model.Q.ncol != 2 || model.Q.nnz != 4) {
        fprintf(stderr, "Q dimensions are wrong\n");
        model_free(&model);
        return 1;
    }

    if (!nearly_equal(model.Q.values[0], 2.0) || !nearly_equal(model.Q.values[1], 1.0) ||
        !nearly_equal(model.Q.values[2], 1.0) || !nearly_equal(model.Q.values[3], 4.0)) {
        fprintf(stderr, "Q values are wrong after symmetrization\n");
        model_free(&model);
        return 1;
    }

    if (model.A.nrow != 2) {
        fprintf(stderr, "unexpected row deletion in mps reader test\n");
        model_free(&model);
        return 1;
    }

    if (!nearly_equal(model.c[0], 1.0) || !nearly_equal(model.c[1], 2.0)) {
        fprintf(stderr, "objective coefficients are wrong\n");
        model_free(&model);
        return 1;
    }

    if (!nearly_equal(model.AL[0], 1.0) || !isinf(model.AL[1]) || model.AL[1] >= 0.0) {
        fprintf(stderr, "constraint lower bounds are wrong\n");
        model_free(&model);
        return 1;
    }

    if (!isinf(model.AU[0]) || model.AU[0] <= 0.0 || !nearly_equal(model.AU[1], 2.0)) {
        fprintf(stderr, "constraint upper bounds are wrong\n");
        model_free(&model);
        return 1;
    }

    if (!nearly_equal(model.l[0], 0.0) || !nearly_equal(model.u[0], 10.0) ||
        !nearly_equal(model.l[1], -5.0) || !nearly_equal(model.u[1], 5.0)) {
        fprintf(stderr, "variable bounds are wrong\n");
        model_free(&model);
        return 1;
    }

    if (model.AT.nrow != 2 || model.AT.ncol != 2 || model.AT.nnz != 4) {
        fprintf(stderr, "AT dimensions are wrong\n");
        model_free(&model);
        return 1;
    }

    model_free(&model);
    printf("test_mps_reader passed\n");
    return 0;
}
