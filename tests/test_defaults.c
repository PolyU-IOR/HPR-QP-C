#include "hprqp.h"
#include <stdio.h>
int main(void) {
    HPRQPParams p;
    params_set_defaults(&p);
    if (p.use_gpu != HPRQP_HAVE_CUDA || p.polishing != 1 ||
        p.gram_mode != HPRQP_GRAM_AUTO || p.polish_python || p.polish_worker ||
        p.spmv_mode_a != HPRQP_SPMV_MODE_AUTO || p.spmv_mode_q != HPRQP_SPMV_MODE_AUTO ||
        p.polish_max_attempts != 12 || p.polish_episode_seconds != 20 ||
        p.polish_total_seconds != 0 || p.stoptol != 1e-8) {
        fputs("Unexpected production defaults\n",stderr);return 1;
    }
    return 0;
}
