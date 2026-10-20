#ifndef HPRQP_POLISH_H
#define HPRQP_POLISH_H
#include "solver_backend.h"
#ifdef __cplusplus
extern "C" {
#endif
void *hprqp_polish_create(const HPRQPModel *, const HPRQPParams *);
int hprqp_polish_poll(void *, HPRQPSolverBackendContext *, const HPRQPSolverBackendOps *,
    const HPRQPResiduals *, size_t, double, double *, double *, double *, double *, HPRQPResiduals *);
void hprqp_polish_destroy(void *, HPRQPResults *);
#ifdef __cplusplus
}
#endif
#endif
