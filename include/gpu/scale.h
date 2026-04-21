#ifndef HPRQP_GPU_SCALE_H
#define HPRQP_GPU_SCALE_H

#include "gpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

HPRQPStatus gpu_problem_scale(HPRQPGPUProblem *gpu_problem,
                              HPRQPGPUWorkspace *workspace,
                              const HPRQPParams *params,
                              HPRQPScaling *host_scaling);

#ifdef __cplusplus
}
#endif

#endif