#include "shared/build.h"
#include "shared/mps_reader.h"
#include "shared/sparse.h"
#include "shared/qcqp.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>
#include <vector>

static HPRQPStatus copy_csr_matrix(HPRQPMatrixCSR *dest, const CSRMatrix *src)
{
    HPRQPStatus status;
    int row;
    int idx;

    if (dest == nullptr || src == nullptr) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = matrix_csr_alloc(dest,
                              static_cast<size_t>(src->nrows),
                              static_cast<size_t>(src->ncols),
                              static_cast<size_t>(src->nnz));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    for (row = 0; row <= src->nrows; ++row) {
        dest->row_ptr[row] = static_cast<size_t>(src->row_ptr[row]);
    }
    for (idx = 0; idx < src->nnz; ++idx) {
        dest->col_ind[idx] = static_cast<size_t>(src->col_idx[idx]);
        dest->values[idx] = src->values[idx];
    }

    return HPRQP_STATUS_OK;
}

typedef struct MpsSymEntry {
    size_t row;
    size_t col;
    double value;
} MpsSymEntry;

static int compare_mps_sym_entry(const void *lhs, const void *rhs)
{
    const MpsSymEntry *a = static_cast<const MpsSymEntry *>(lhs);
    const MpsSymEntry *b = static_cast<const MpsSymEntry *>(rhs);
    if (a->row < b->row) return -1;
    if (a->row > b->row) return 1;
    if (a->col < b->col) return -1;
    if (a->col > b->col) return 1;
    return 0;
}

static HPRQPStatus symmetrize_q_for_mps(HPRQPMatrixCSR *dest, const HPRQPMatrixCSR *src)
{
    HPRQPStatus status;
    size_t row;
    size_t idx;
    size_t nnz = src->nnz * 2;
    MpsSymEntry *entries = static_cast<MpsSymEntry *>(malloc((nnz > 0 ? nnz : 1) * sizeof(MpsSymEntry)));
    if (entries == nullptr) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    nnz = 0;
    for (row = 0; row < src->nrow; ++row) {
        for (idx = src->row_ptr[row]; idx < src->row_ptr[row + 1]; ++idx) {
            entries[nnz].row = row;
            entries[nnz].col = src->col_ind[idx];
            entries[nnz].value = src->values[idx];
            nnz += 1;
            if (src->col_ind[idx] != row) {
                entries[nnz].row = src->col_ind[idx];
                entries[nnz].col = row;
                entries[nnz].value = src->values[idx];
                nnz += 1;
            }
        }
    }

    qsort(entries, nnz, sizeof(MpsSymEntry), compare_mps_sym_entry);

    {
        size_t write_idx = 0;
        for (idx = 0; idx < nnz; ++idx) {
            if (write_idx > 0 && entries[idx].row == entries[write_idx - 1].row && entries[idx].col == entries[write_idx - 1].col) {
                entries[write_idx - 1].value += entries[idx].value;
            } else {
                entries[write_idx++] = entries[idx];
            }
        }
        nnz = write_idx;
    }

    status = matrix_csr_alloc(dest, src->nrow, src->ncol, nnz);
    if (status != HPRQP_STATUS_OK) {
        free(entries);
        return status;
    }
    for (idx = 0; idx < nnz; ++idx) {
        dest->row_ptr[entries[idx].row + 1] += 1;
    }
    for (row = 0; row < src->nrow; ++row) {
        dest->row_ptr[row + 1] += dest->row_ptr[row];
    }
    for (idx = 0; idx < nnz; ++idx) {
        dest->col_ind[idx] = entries[idx].col;
        dest->values[idx] = entries[idx].value;
    }
    free(entries);
    return HPRQP_STATUS_OK;
}

/* Split quadratic rows out of A, preserving the order of the remaining rows.
 * QCMATRIX stores x' H x, whereas the factor API uses .5*x' D x; D = 2*H.
 * Only diagonal convex inequalities are supported by this file loader. */
static HPRQPStatus build_qc_mps(const QPSData *qps, const HPRQPMatrixCSR *Q,
                              const HPRQPMatrixCSR *A, HPRQPModel *model)
{
    std::vector<const MPSQuadraticRow *> rows(qps->ncon, nullptr);
    for (const auto &qc : qps->quadratic_rows) rows[qc.row] = &qc;
    std::vector<std::vector<std::pair<int, double>>> diag(qps->ncon);
    std::vector<double> signs(qps->ncon, 1.0), constant, al, au;
    size_t nq = 0, dnz = 0, cnz = 0, anz = 0;
    for (int r = 0; r < qps->ncon; ++r) {
        if (rows[r]) {
            const auto &qc = *rows[r];
            auto &d = diag[r];
            for (size_t k = 0; k < qc.values.size(); ++k)
                d.emplace_back(qc.columns[k], qc.values[k]);
            std::sort(d.begin(), d.end());
            size_t write = 0;
            for (const auto entry : d) {
                if (write && d[write-1].first == entry.first) d[write-1].second += entry.second;
                else d[write++] = entry;
            }
            d.resize(write);
            d.erase(std::remove_if(d.begin(), d.end(),
                                  [](const auto &x) { return x.second == 0.0; }), d.end());
            if (!d.empty()) {
                if (qps->lcon[r] == -INFINITY && std::isfinite(qps->ucon[r])) signs[r] = 1.0;
                else if (qps->ucon[r] == INFINITY && std::isfinite(qps->lcon[r])) signs[r] = -1.0;
                else {
                    fprintf(stderr, "Error: QCMATRIX row '%s' needs a one-sided inequality; "
                                    "quadratic equalities/ranges/free rows are unsupported.\n", qps->connames[r]);
                    return HPRQP_STATUS_INVALID_ARGUMENT;
                }
                for (auto &entry : d) {
                    entry.second *= 2.0 * signs[r];
                    if (!std::isfinite(entry.second) || entry.second < 0.0) {
                        fprintf(stderr, "Error: QCMATRIX row '%s' is not convex in its inequality sense.\n",
                                qps->connames[r]);
                        return HPRQP_STATUS_INVALID_ARGUMENT;
                    }
                }
                ++nq;
                dnz += d.size();
                cnz += A->row_ptr[r+1] - A->row_ptr[r];
                continue;
            }
        }
        anz += A->row_ptr[r+1] - A->row_ptr[r];
    }
    if (!nq) return build_model_from_arrays(Q, A, qps->c, qps->lcon, qps->ucon,
                                           qps->lvar, qps->uvar, qps->c0, model);
    /* RAII prevents leaking a partially constructed input if allocation fails. */
    struct Matrix {
        HPRQPMatrixCSR value{};
        ~Matrix() { matrix_csr_free(&value); }
    } linear, C, D;
    HPRQPStatus status = matrix_csr_alloc(&linear.value, qps->ncon-nq, qps->nvar, anz);
    if (status != HPRQP_STATUS_OK) return status;
    status = matrix_csr_alloc(&C.value, nq, qps->nvar, cnz);
    if (status != HPRQP_STATUS_OK) return status;
    status = matrix_csr_alloc(&D.value, nq, qps->nvar, dnz);
    if (status != HPRQP_STATUS_OK) return status;
    size_t lr = 0, qr = 0, ak = 0, ck = 0, dk = 0;
    for (int r = 0; r < qps->ncon; ++r) {
        if (diag[r].empty()) {
            for (size_t k = A->row_ptr[r]; k < A->row_ptr[r+1]; ++k) {
                linear.value.col_ind[ak] = A->col_ind[k];
                linear.value.values[ak++] = A->values[k];
            }
            linear.value.row_ptr[++lr] = ak;
            al.push_back(qps->lcon[r]); au.push_back(qps->ucon[r]);
        } else {
            for (size_t k = A->row_ptr[r]; k < A->row_ptr[r+1]; ++k) {
                C.value.col_ind[ck] = A->col_ind[k];
                C.value.values[ck++] = signs[r] * A->values[k];
            }
            for (const auto &entry : diag[r]) {
                D.value.col_ind[dk] = entry.first; D.value.values[dk++] = entry.second;
            }
            constant.push_back(signs[r] > 0.0 ? -qps->ucon[r] : qps->lcon[r]);
            C.value.row_ptr[qr+1] = ck; D.value.row_ptr[++qr] = dk;
        }
    }
    return build_from_QAbc_qc_diagonal(Q, &linear.value, qps->c, al.data(), au.data(),
        qps->lvar, qps->uvar, qps->c0, &C.value, constant.data(), &D.value, model);
}

extern "C" HPRQPStatus build_model_from_mps(const char *mps_path, HPRQPModel *model)
{
    QPSData *qps;
    CSRMatrix *csr_A;
    CSRMatrix *csr_Q;
    HPRQPMatrixCSR A;
    HPRQPMatrixCSR Q;
    HPRQPStatus status;

    if (mps_path == nullptr || model == nullptr) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    model_init(model);
    A = {};
    Q = {};

    qps = readqps(mps_path, MPS_FREE);
    if (qps == nullptr) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (qps->objsense == OBJSENSE_MAX) {
        fprintf(stderr, "Error: Maximization MPS input is unsupported; supply a convex minimization model.\n");
        qpsdata_free(qps);
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    for (int j = 0; j < qps->nvar; ++j) if (qps->vartypes[j] != VTYPE_CONTINUOUS) {
        fprintf(stderr, "Error: Discrete/semi-continuous MPS variables are unsupported.\n");
        qpsdata_free(qps);
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    csr_A = qpsdata_get_csr_matrix(qps);
    csr_Q = coo_to_csr(qps->nvar, qps->nvar, qps->qnnz, qps->qrows, qps->qcols, qps->qvals);
    if (csr_A == nullptr || csr_Q == nullptr) {
        if (csr_A != nullptr) {
            csr_free(csr_A);
        }
        if (csr_Q != nullptr) {
            csr_free(csr_Q);
        }
        qpsdata_free(qps);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    status = copy_csr_matrix(&A, csr_A);
    if (status != HPRQP_STATUS_OK) {
        csr_free(csr_A);
        csr_free(csr_Q);
        qpsdata_free(qps);
        return status;
    }

    status = copy_csr_matrix(&Q, csr_Q);
    if (status != HPRQP_STATUS_OK) {
        matrix_csr_free(&A);
        csr_free(csr_A);
        csr_free(csr_Q);
        qpsdata_free(qps);
        return status;
    }

    if (qps->full_objective_matrix) {
        for (size_t r = 0; r < Q.nrow; ++r) {
            for (size_t k = Q.row_ptr[r]; k < Q.row_ptr[r+1]; ++k) {
                size_t col = Q.col_ind[k];
                const size_t *begin = Q.col_ind + Q.row_ptr[col];
                const size_t *end = Q.col_ind + Q.row_ptr[col+1];
                const size_t *other = std::lower_bound(begin, end, r);
                double value = other != end && *other == r ? Q.values[other-Q.col_ind] : 0.0;
                if (value != Q.values[k]) {
                    fprintf(stderr, "Error: QMATRIX must contain a symmetric full objective matrix.\n");
                    matrix_csr_free(&A); matrix_csr_free(&Q);
                    csr_free(csr_A); csr_free(csr_Q); qpsdata_free(qps);
                    return HPRQP_STATUS_INVALID_ARGUMENT;
                }
            }
        }
    } else {
        HPRQPMatrixCSR sym_Q = {};
        status = symmetrize_q_for_mps(&sym_Q, &Q);
        if (status != HPRQP_STATUS_OK) {
            matrix_csr_free(&A);
            matrix_csr_free(&Q);
            csr_free(csr_A);
            csr_free(csr_Q);
            qpsdata_free(qps);
            return status;
        }
        matrix_csr_free(&Q);
        Q = sym_Q;
    }

    try {
    status = qps->quadratic_rows.empty() ? build_model_from_arrays(&Q,
                                     &A,
                                     qps->c,
                                     qps->lcon,
                                     qps->ucon,
                                     qps->lvar,
                                     qps->uvar,
                                     qps->c0,
                                     model) : build_qc_mps(qps, &Q, &A, model);
    } catch (const std::bad_alloc &) {
        status = HPRQP_STATUS_ALLOCATION_FAILED;
    }

    matrix_csr_free(&A);
    matrix_csr_free(&Q);
    csr_free(csr_A);
    csr_free(csr_Q);
    qpsdata_free(qps);
    return status;
}

extern "C" HPRQPStatus build_from_mps(const char *mps_path, HPRQPModel *model)
{
    return build_model_from_mps(mps_path, model);
}
