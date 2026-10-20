#ifndef HPRQP_GRAM_POLICY_H
#define HPRQP_GRAM_POLICY_H
#include "shared/types.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Conservative structural heuristic, not a convergence or speedup guarantee. */
int hprqp_gram_auto_eligible(const HPRQPModel *model);
#ifdef __cplusplus
}
#endif
#endif
