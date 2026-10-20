#include "hprqp.h"
#include "shared/diagnostics.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef HPRQP_SOURCE_DIR
#define HPRQP_SOURCE_DIR "."
#endif

static const char *status_name(HPRQPStatus status)
{
    switch (status) {
    case HPRQP_STATUS_OK:
        return "OK";
    case HPRQP_STATUS_OPTIMAL:
        return "OPTIMAL";
    case HPRQP_STATUS_INVALID_ARGUMENT:
        return "INVALID_ARGUMENT";
    case HPRQP_STATUS_ALLOCATION_FAILED:
        return "ALLOCATION_FAILED";
    case HPRQP_STATUS_NOT_IMPLEMENTED:
        return "NOT_IMPLEMENTED";
    case HPRQP_STATUS_CONTINUE:
        return "CONTINUE";
    case HPRQP_STATUS_MAX_ITER:
        return "MAX_ITER";
    case HPRQP_STATUS_TIME_LIMIT:
        return "TIME_LIMIT";
    default:
        return "UNKNOWN";
    }
}

static int parse_size_arg(const char *value, size_t *out)
{
    char *endptr;
    unsigned long long parsed;

    if (value == NULL || out == NULL) {
        return 0;
    }

    parsed = strtoull(value, &endptr, 10);
    if (*value == '\0' || *endptr != '\0') {
        return 0;
    }

    *out = (size_t)parsed;
    return 1;
}

static int parse_int_arg(const char *value, int *out)
{
    char *endptr;
    long parsed;

    if (value == NULL || out == NULL) {
        return 0;
    }

    parsed = strtol(value, &endptr, 10);
    if (*value == '\0' || *endptr != '\0') {
        return 0;
    }

    *out = (int)parsed;
    return 1;
}

static int parse_double_arg(const char *value, double *out)
{
    char *endptr;
    double parsed;

    if (value == NULL || out == NULL) {
        return 0;
    }

    parsed = strtod(value, &endptr);
    if (*value == '\0' || *endptr != '\0') {
        return 0;
    }

    *out = parsed;
    return 1;
}

static int parse_spmv_mode_arg(const char *value, HPRQPSpmvMode *out)
{
    if (value == NULL || out == NULL) {
        return 0;
    }
    if (strcmp(value, "auto") == 0) {
        *out = HPRQP_SPMV_MODE_AUTO;
        return 1;
    }
    if (strcmp(value, "cusparse") == 0) {
        *out = HPRQP_SPMV_MODE_CUSPARSE;
        return 1;
    }
    if (strcmp(value, "customized") == 0) {
        *out = HPRQP_SPMV_MODE_CUSTOMIZED;
        return 1;
    }
    return 0;
}

static void print_usage(const char *argv0)
{
    printf("usage: %s [mps_path] [--max-iter N] [--stoptol X] [--check-iter N] [--print-frequency N] [--time-limit X] [--verbose 0|1] [--use-gpu 0|1] [--device-number N] [--spmv-mode-q auto|cusparse|customized] [--spmv-mode-a auto|cusparse|customized] [--use-scaling 0|1] [--gram off|auto|request] [--polishing 0|1] [--polish-python PATH] [--polish-worker PATH] [--result-json PATH]\n", argv0);
}

/* Optional machine metadata keeps dataset validation independent of log verbosity. */
static int write_metadata(const char *path, const HPRQPModel *model,
                          const HPRQPResults *results)
{
    FILE *file = fopen(path, "w");
    int failed;
    if (file == NULL) return 0;
    fprintf(file, "{\"schema\":1,\"input_format\":\"mps_qcmatrix_v1\","
                  "\"row_cones\":%zu,\"gram_active\":%d,"
                  "\"polish_attempts\":%zu,\"polish_accepted\":%d,\"total_time\":",
            model->num_row_cones, results->gram_active,
            results->polish_attempts, results->polish_accepted);
    if (isfinite(results->total_time)) fprintf(file, "%.17g", results->total_time);
    else fputs("null", file);
    fputs("}\n", file);
    failed = ferror(file);
    if (fclose(file) != 0) failed = 1;
    return !failed;
}

int main(int argc, char **argv)
{
    HPRQPModel model;
    HPRQPParams params;
    HPRQPResults results;
    HPRQPStatus status;
    const char *mps_path = HPRQP_SOURCE_DIR "/model.mps";
    const char *result_json_path = NULL;
    size_t i;
    int argi = 1;
    int metadata_ok = 1;

    puts("HPR-QP-C version: v0.1.1");
    fflush(stdout);

    model_init(&model);
    results_init(&results);
    params_set_defaults(&params);
    // params.verbose = 0;

    while (argi < argc) {
        if (strcmp(argv[argi], "--max-iter") == 0 && argi + 1 < argc) {
            if (!parse_size_arg(argv[argi + 1], &params.max_iter)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--stoptol") == 0 && argi + 1 < argc) {
            if (!parse_double_arg(argv[argi + 1], &params.stoptol)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--check-iter") == 0 && argi + 1 < argc) {
            if (!parse_size_arg(argv[argi + 1], &params.check_iter)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--print-frequency") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.print_frequency)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--time-limit") == 0 && argi + 1 < argc) {
            if (!parse_double_arg(argv[argi + 1], &params.time_limit)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--verbose") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.verbose)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--use-gpu") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.use_gpu)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--device-number") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.device_number)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--spmv-mode-q") == 0 && argi + 1 < argc) {
            if (!parse_spmv_mode_arg(argv[argi + 1], &params.spmv_mode_q)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--spmv-mode-a") == 0 && argi + 1 < argc) {
            if (!parse_spmv_mode_arg(argv[argi + 1], &params.spmv_mode_a)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--use-scaling") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.use_scaling)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--gram") == 0 && argi + 1 < argc) {
            const char *mode = argv[argi + 1];
            if (strcmp(mode, "off") == 0) params.gram_mode = HPRQP_GRAM_OFF;
            else if (strcmp(mode, "auto") == 0) params.gram_mode = HPRQP_GRAM_AUTO;
            else if (strcmp(mode, "request") == 0) params.gram_mode = HPRQP_GRAM_REQUEST;
            else { print_usage(argv[0]); return 1; }
            argi += 2;
        } else if (strcmp(argv[argi], "--polishing") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.polishing) ||
                (params.polishing != 0 && params.polishing != 1)) {
                print_usage(argv[0]); return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--polish-python") == 0 && argi + 1 < argc) {
            params.polish_python = argv[argi + 1]; argi += 2;
        } else if (strcmp(argv[argi], "--polish-worker") == 0 && argi + 1 < argc) {
            params.polish_worker = argv[argi + 1]; argi += 2;
        } else if (strcmp(argv[argi], "--result-json") == 0 && argi + 1 < argc) {
            result_json_path = argv[argi + 1]; argi += 2;
        } else if (argv[argi][0] != '-') {
            mps_path = argv[argi];
            argi += 1;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    status = build_from_mps(mps_path, &model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "failed to load MPS file '%s' with status %s\n", mps_path, status_name(status));
        return 1;
    }

    status = solve(&model, &params, &results);
    printf("model: %s\n", mps_path);
    printf("status: %s\n", status == HPRQP_STATUS_OPTIMAL && results.polish_accepted
                               ? "polished_optimal" : status_name(status));
    printf("iter: %zu\n", results.iter);
    printf("time: %.6f\n", results.time);
    printf("objective: %.12g\n", results.primal_obj);
    printf("residual: %.12g\n", results.residuals);
    printf("gap: %.12g\n", results.gap);
    if (hprqp_diagnostics_enabled()) {
        printf("input_format: mps_qcmatrix_v1\n");
        printf("row_cones: %zu\n", model.num_row_cones);
        printf("total_time: %.6f\n", results.total_time);
        printf("gram_active: %d\n", results.gram_active);
        printf("polish_attempts: %zu\n", results.polish_attempts);
        printf("polish_accepted: %d\n", results.polish_accepted);
    }
    printf("iter_4: %zu\n", results.iter_4);
    printf("time_4: %.6f\n", results.time_4);
    printf("iter_6: %zu\n", results.iter_6);
    printf("time_6: %.6f\n", results.time_6);
    if (result_json_path != NULL) {
        metadata_ok = write_metadata(result_json_path, &model, &results);
        if (!metadata_ok)
            fprintf(stderr, "failed to write result metadata '%s'\n", result_json_path);
    }

    // if (results.x != NULL) {
    //     printf("x:");
    //     for (i = 0; i < model.Q.ncol; ++i) {
    //         printf(" %.12g", results.x[i]);
    //     }
    //     printf("\n");
    // }

    // if (results.y != NULL && model.A.nrow > 0) {
    //     printf("y:");
    //     for (i = 0; i < model.A.nrow; ++i) {
    //         printf(" %.12g", results.y[i]);
    //     }
    //     printf("\n");
    // }

    results_free(&results);
    model_free(&model);
    return metadata_ok && (status == HPRQP_STATUS_OPTIMAL || status == HPRQP_STATUS_MAX_ITER || status == HPRQP_STATUS_TIME_LIMIT) ? 0 : 1;
}
