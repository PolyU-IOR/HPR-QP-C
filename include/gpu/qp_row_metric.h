#ifndef HPRQP_QP_ROW_METRIC_H
#define HPRQP_QP_ROW_METRIC_H
#include "gpu/types.h"
#ifdef __cplusplus
extern "C" {
#endif
int gpu_qp_metric_enabled(const HPRQPGPUWorkspace*);
HPRQPStatus gpu_qp_metric_setup(HPRQPGPUWorkspace*,const HPRQPGPUProblem*,const HPRQPParams*);
void gpu_qp_metric_free(HPRQPGPUWorkspace*);
HPRQPStatus gpu_qp_metric_finish_y(HPRQPGPUWorkspace*,double,double);
HPRQPStatus gpu_qp_metric_quadratic(HPRQPGPUWorkspace*,double*);
#ifdef __cplusplus
}
#endif
#endif
