#ifndef HPRQP_TYPES_H
#define HPRQP_TYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum HPRQPStatus {
    HPRQP_STATUS_OK = 0,
    HPRQP_STATUS_INVALID_ARGUMENT = 1,
    HPRQP_STATUS_ALLOCATION_FAILED = 2,
    HPRQP_STATUS_NOT_IMPLEMENTED = 3,
    HPRQP_STATUS_MAX_ITER = 4,
    HPRQP_STATUS_TIME_LIMIT = 5,
    HPRQP_STATUS_OPTIMAL = 6,
    HPRQP_STATUS_CONTINUE = 7
} HPRQPStatus;

typedef enum HPRQPSpmvMode {
    HPRQP_SPMV_MODE_AUTO = 0,
    HPRQP_SPMV_MODE_CUSPARSE = 1,
    HPRQP_SPMV_MODE_CUSTOMIZED = 2
} HPRQPSpmvMode;

typedef struct HPRQPMatrixCSR {
    size_t nrow;
    size_t ncol;
    size_t nnz;
    size_t *row_ptr;
    size_t *col_ind;
    double *values;
    int owns_data;
} HPRQPMatrixCSR;

typedef struct HPRQPModel {
    HPRQPMatrixCSR Q;
    HPRQPMatrixCSR A;
    HPRQPMatrixCSR AT;
    double *c;
    double *AL;
    double *AU;
    double *l;
    double *u;
    double obj_constant;
    int owns_data;
} HPRQPModel;

typedef struct HPRQPScaling {
    double *row_norm;
    double *col_norm;
    double *l_org;
    double *u_org;
    double b_scale;
    double c_scale;
    double norm_b;
    double norm_c;
    double norm_b_org;
    double norm_c_org;
} HPRQPScaling;

typedef struct HPRQPParams {
    double stoptol;
    size_t max_iter;
    int use_gpu;
    int device_number;
    double time_limit;
    size_t check_iter;
    int print_frequency;
    int use_scaling;
    int verbose;
    HPRQPSpmvMode spmv_mode_q;
    HPRQPSpmvMode spmv_mode_a;
} HPRQPParams;

typedef struct HPRQPWorkspace {
    size_t m;
    size_t n;
    double sigma;
    double lambda_max_A;
    double lambda_max_Q;
    int Q_is_diag;
    int noC;
    int to_check;
    const HPRQPMatrixCSR *Q;
    const HPRQPMatrixCSR *A;
    const HPRQPMatrixCSR *AT;
    const double *c;
    const double *AL;
    const double *AU;
    const double *l;
    const double *u;
    double *x;
    double *x_bar;
    double *x_hat;
    double *w;
    double *w_bar;
    double *w_hat;
    double *y;
    double *y_bar;
    double *y_hat;
    double *Ax;
    double *ATy;
    double *ATy_bar;
    double *ATdy;
    double *Qw;
    double *Qw_bar;
    double *Qx;
    double *dQw;
    double *QATdy;
    double *dx;
    double *dy;
    double *dw;
    double *Rp;
    double *Rd;
    double *s;
    double *z_bar;
    double *tempv;
    double *diag_Q;
    double *fact1;
    double *fact2;
    double *fact;
    double *fact_M;
    double *last_x;
    double *last_y;
    double *last_Qw;
    double *last_w;
    double *last_ATy;
} HPRQPWorkspace;

typedef struct HPRQPResiduals {
    int is_updated;
    double err_Rp_org_bar;
    double err_Rd_org_bar;
    double KKTx_and_gap_org_bar;
    double primal_obj_bar;
    double rel_gap_bar;
    double dual_obj_bar;
} HPRQPResiduals;

typedef struct HPRQPRestart {
    int restart_flag;
    int first_restart;
    double last_gap;
    double current_gap;
    double save_gap;
    size_t inner;
    size_t step;
    size_t sufficient;
    size_t necessary;
    size_t long_restart;
    size_t ratio;
    size_t times;
    double weighted_norm;
    double best_gap;
    double best_sigma;
} HPRQPRestart;

typedef struct HPRQPResults {
    size_t iter;
    size_t iter_4;
    size_t iter_6;
    double time;
    double time_4;
    double time_6;
    double primal_obj;
    double residuals;
    double gap;
    int status;
    double *x;
    double *y;
    double *z;
    double *w;
} HPRQPResults;

#ifdef __cplusplus
}
#endif

#endif
