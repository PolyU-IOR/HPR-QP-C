#include "test_support.h"

int main(void)
{
    int exit_code = 0;
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPScaling scaling;
    HPRQPParams params;
    double max_row_abs = 0.0;
    double max_col0_abs = 0.0;
    double max_col1_abs = 0.0;

    model_init(&model);
    scaling_init(&scaling);
    params_set_defaults(&params);

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_SCALE, &Q, &A, &model));
    TEST_ASSERT_OK(scale_problem(&model, &params, &scaling));

    TEST_ASSERT(scaling.row_norm[0] > 0.0 && scaling.col_norm[0] > 0.0 && scaling.col_norm[1] > 0.0,
                "scaling norms are non-positive");

    max_row_abs = fabs(model.A.values[0]) > fabs(model.A.values[1]) ? fabs(model.A.values[0]) : fabs(model.A.values[1]);
    max_col0_abs = fabs(model.A.values[0]) > fabs(model.Q.values[0]) ? fabs(model.A.values[0]) : fabs(model.Q.values[0]);
    max_col1_abs = fabs(model.A.values[1]) > fabs(model.Q.values[1]) ? fabs(model.A.values[1]) : fabs(model.Q.values[1]);

    TEST_ASSERT(max_row_abs <= 1.0 + 1e-10 && max_col0_abs <= 1.0 + 1e-10 && max_col1_abs <= 1.0 + 1e-10,
                "scaled matrix magnitudes are too large");

    TEST_ASSERT(matrix_csr_is_valid(&model.AT) && model.AT.nrow == 2 && model.AT.ncol == 1 && model.AT.nnz == 2,
                "scaled AT is wrong");

    TEST_ASSERT(scaling.norm_b > 0.0 && scaling.norm_c > 0.0 &&
                scaling.norm_b_org > 0.0 && scaling.norm_c_org > 0.0,
                "scaling norms metadata are wrong");

    TEST_ASSERT(test_nearly_equal_abs(scaling.l_org[0], 0.0, 1e-10) &&
                test_nearly_equal_abs(scaling.l_org[1], -1.0, 1e-10) &&
                test_nearly_equal_abs(scaling.u_org[0], 10.0, 1e-10) &&
                test_nearly_equal_abs(scaling.u_org[1], 3.0, 1e-10),
                "original bound copies are wrong");

cleanup:
    scaling_free(&scaling);
    model_free(&model);
    matrix_csr_free(&Q);
    matrix_csr_free(&A);
    if (exit_code == 0) {
        printf("test_scale passed\n");
    }
    return exit_code;
}