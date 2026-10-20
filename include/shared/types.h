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

/* Standard Lorentz cones occupy contiguous suffix blocks. Offsets are zero-based,
 * length count+1, strictly increasing, and end at n (variables) or m (rows).
 * soc_rhs has m entries, zero on linear rows: A_soc*x - soc_rhs_soc in K.
 * Cone variables/rows must have (-INFINITY,+INFINITY) interval bounds.
 * The builder copies all arrays. A zero-initialized spec means an ordinary QP. */
typedef struct HPRQPSOCSpec {
    size_t num_var_cones;
    const size_t *var_offsets;
    size_t num_row_cones;
    const size_t *row_offsets;
    const double *rhs;
} HPRQPSOCSpec;

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
    size_t num_var_cones;
    size_t *var_cone_offsets;
    size_t num_row_cones;
    size_t *row_cone_offsets;
    double *soc_rhs;
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

/* Independent, cold-start row-coordinate policy; never switches mid-solve. */
typedef enum HPRQPGramMode {
    HPRQP_GRAM_OFF = 0,
    HPRQP_GRAM_AUTO = 1, /* conservative detected spatial SOC family */
    HPRQP_GRAM_REQUEST = 2 /* try every implemented eligible model; may decline */
} HPRQPGramMode;

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
    HPRQPGramMode gram_mode;
    int polishing; /* GPU owner, isolated CPU worker; terminal candidates only */
    const char *polish_python; /* NULL: python3 on PATH */
    const char *polish_worker; /* NULL: installed package or source-tree worker */
    const char *polish_directory; /* NULL: private directory under /tmp */
    int polish_keep_files;
    size_t polish_max_attempts; /* 0: no attempt-count limit */
    double polish_episode_seconds; /* 0: remaining global solve budget */
    double polish_total_seconds; /* 0: no cumulative worker budget */
    size_t polish_memory_bytes; /* 0: no address-space limit */
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
    size_t polish_attempts;
    int polish_accepted;
    int gram_active;
    double total_time; /* solve call including setup, CPU cleanup and output */
} HPRQPResults;

#ifdef __cplusplus
}
#endif

#endif
