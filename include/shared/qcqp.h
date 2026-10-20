#ifndef HPRQP_QCQP_H
#define HPRQP_QCQP_H
#include "shared/types.h"
#ifdef __cplusplus
extern "C" {
#endif
/* h_i(x) = .5*||R_i*x||^2 + C_i*x + constant_i <= 0.
 * R stacks sparse factor rows; offsets (C.nrow+1) partition R's rows.
 * The lift has rows(R_i)+2 rows, no auxiliary variables, and never forms R_i'R_i.
 * A zero-rank factor is permitted. Factors need not be diagonal or full rank. */
HPRQPStatus build_from_QAbc_qc_factors(const HPRQPMatrixCSR *Q,
    const HPRQPMatrixCSR *A, const double *c, const double *AL, const double *AU,
    const double *l, const double *u, double obj_constant,
    const HPRQPMatrixCSR *C, const double *constant,
    const HPRQPMatrixCSR *R, const size_t *offsets, HPRQPModel *model);
/* Diagonal h_i: .5*sum_j D_ij*x_j^2+C_i*x+constant_i <= 0.
 * Finite nonnegative D only; each positive entry becomes one factor row. */
HPRQPStatus build_from_QAbc_qc_diagonal(const HPRQPMatrixCSR *Q,
    const HPRQPMatrixCSR *A, const double *c, const double *AL, const double *AU,
    const double *l, const double *u, double obj_constant,
    const HPRQPMatrixCSR *C, const double *constant,
    const HPRQPMatrixCSR *D, HPRQPModel *model);
#ifdef __cplusplus
}
#endif
#endif
