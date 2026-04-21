#include "shared/build.h"
#include "shared/mps_reader.h"
#include "shared/sparse.h"

#include <cstddef>
#include <cstdlib>

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

    {
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

    status = build_model_from_arrays(&Q,
                                     &A,
                                     qps->c,
                                     qps->lcon,
                                     qps->ucon,
                                     qps->lvar,
                                     qps->uvar,
                                     qps->c0,
                                     model);

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