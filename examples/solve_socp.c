#include "hprqp.h"
#include <math.h>
#include <stdio.h>

int main(void)
{
    /* min t subject to (t-2, 3, 4) in Q^3. The optimum is t=7.
     * Zero rows encode constant affine cone components and must be retained. */
    size_t qptr[] = {0, 0}, qcol[] = {0}, aptr[] = {0, 1, 1, 1}, acol[] = {0};
    double qval[] = {0}, aval[] = {1}, c[] = {1}, l[] = {-INFINITY}, u[] = {INFINITY};
    double al[] = {-INFINITY, -INFINITY, -INFINITY}, au[] = {INFINITY, INFINITY, INFINITY};
    double rhs[] = {2, -3, -4};
    size_t row_offsets[] = {0, 3};
    HPRQPMatrixCSR Q = {1, 1, 0, qptr, qcol, qval, 0};
    HPRQPMatrixCSR A = {3, 1, 1, aptr, acol, aval, 0};
    HPRQPSOCSpec cones = {0, NULL, 1, row_offsets, rhs};
    HPRQPModel model;
    HPRQPParams params;
    HPRQPResults results;
    HPRQPStatus status;
    model_init(&model); results_init(&results); params_set_defaults(&params);
    status = build_from_QAbc_soc(&Q, &A, c, al, au, l, u, 0.0, &cones, &model);
    if (status == HPRQP_STATUS_OK) status = solve(&model, &params, &results);
    if (status == HPRQP_STATUS_OPTIMAL) printf("SOC optimum: t=%.12g (expected 7)\n", results.x[0]);
    else fprintf(stderr, "SOC solve failed: status=%d\n", (int)status);
    results_free(&results); model_free(&model);
    return status == HPRQP_STATUS_OPTIMAL ? 0 : 1;
}
