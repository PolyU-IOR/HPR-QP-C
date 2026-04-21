#ifndef HPRQP_SCALE_H
#define HPRQP_SCALE_H

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

void scaling_init(HPRQPScaling *scaling);
void scaling_free(HPRQPScaling *scaling);
HPRQPStatus scale_problem(HPRQPModel *model, const HPRQPParams *params, HPRQPScaling *scaling);

#ifdef __cplusplus
}
#endif

#endif