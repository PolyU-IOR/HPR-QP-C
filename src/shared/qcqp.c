#include "hprqp.h"
#include "shared/build.h"
#include "shared/sparse.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int finite_matrix(const HPRQPMatrixCSR *a)
{
    if (!matrix_csr_is_valid(a)) return 0;
    for (size_t k=0;k<a->nnz;++k) if (!isfinite(a->values[k])) return 0;
    return 1;
}

HPRQPStatus build_from_QAbc_qc_factors(const HPRQPMatrixCSR *Q,
    const HPRQPMatrixCSR *A, const double *c, const double *AL, const double *AU,
    const double *l, const double *u, double obj_constant,
    const HPRQPMatrixCSR *C, const double *constant,
    const HPRQPMatrixCSR *R, const size_t *offsets, HPRQPModel *model)
{
    HPRQPMatrixCSR combined={0};
    HPRQPStatus status=HPRQP_STATUS_INVALID_ARGUMENT;
    double *al=NULL,*au=NULL,*rhs=NULL;
    size_t *cones=NULL;
    if (!Q || !A || !model || !finite_matrix(C) || !finite_matrix(R) ||
        !matrix_csr_is_valid(A) || C->ncol!=Q->ncol || R->ncol!=Q->ncol ||
        A->ncol!=Q->ncol) return status;
    size_t nq=C->nrow, m=A->nrow;
    if (!nq) return build_from_QAbc(Q,A,c,AL,AU,l,u,obj_constant,model);
    if (!constant || !offsets || offsets[0]!=0 || offsets[nq]!=R->nrow ||
        (m && (!AL || !AU))) return status;
    for (size_t i=0;i<nq;++i)
        if (!isfinite(constant[i]) || offsets[i]>offsets[i+1] || offsets[i+1]>R->nrow) return status;
    if (R->nrow>SIZE_MAX-m || R->nnz>SIZE_MAX-A->nnz ||
        nq>(SIZE_MAX-m-R->nrow)/2 || C->nnz>(SIZE_MAX-A->nnz-R->nnz)/2) return status;
    size_t rows=m+2*nq+R->nrow, nz=A->nnz+2*C->nnz+R->nnz;
    if (rows>SIZE_MAX/sizeof(double) || nz>SIZE_MAX/sizeof(double) ||
        nz>SIZE_MAX/sizeof(size_t) || nq+1>SIZE_MAX/sizeof(size_t)) return status;
    status=matrix_csr_alloc(&combined,rows,Q->ncol,nz);
    if (status!=HPRQP_STATUS_OK) return status;
    al=(double*)malloc(rows*sizeof(double));au=(double*)malloc(rows*sizeof(double));
    rhs=(double*)calloc(rows,sizeof(double));cones=(size_t*)malloc((nq+1)*sizeof(size_t));
    if (!al || !au || !rhs || !cones) {status=HPRQP_STATUS_ALLOCATION_FAILED;goto cleanup;}
    memcpy(combined.row_ptr,A->row_ptr,(m+1)*sizeof(size_t));
    if (A->nnz) {memcpy(combined.col_ind,A->col_ind,A->nnz*sizeof(size_t));memcpy(combined.values,A->values,A->nnz*sizeof(double));}
    if (m) {memcpy(al,AL,m*sizeof(double));memcpy(au,AU,m*sizeof(double));}
    size_t row=m,k=A->nnz;
    for (size_t i=0;i<nq;++i) {
        cones[i]=row;
        for (int head=0;head<2;++head) {
            combined.row_ptr[row]=k;
            for(size_t j=C->row_ptr[i];j<C->row_ptr[i+1];++j) {
                combined.col_ind[k]=C->col_ind[j];combined.values[k++]=-C->values[j];
            }
            rhs[row]=constant[i]+(head?1.0:-1.0);++row;
        }
        for (size_t r=offsets[i];r<offsets[i+1];++r) {
            combined.row_ptr[row++]=k;
            for(size_t j=R->row_ptr[r];j<R->row_ptr[r+1];++j) {
                combined.col_ind[k]=R->col_ind[j];combined.values[k++]=sqrt(2.0)*R->values[j];
            }
        }
    }
    cones[nq]=row;combined.row_ptr[row]=k;
    for(size_t i=m;i<rows;++i) {al[i]=-INFINITY;au[i]=INFINITY;}
    HPRQPSOCSpec spec={0,NULL,nq,cones,rhs};
    status=build_from_QAbc_soc(Q,&combined,c,al,au,l,u,obj_constant,&spec,model);
cleanup:
    matrix_csr_free(&combined);free(al);free(au);free(rhs);free(cones);
    return status;
}

HPRQPStatus build_from_QAbc_qc_diagonal(const HPRQPMatrixCSR *Q,
    const HPRQPMatrixCSR *A, const double *c, const double *AL, const double *AU,
    const double *l, const double *u, double obj_constant,
    const HPRQPMatrixCSR *C, const double *constant,
    const HPRQPMatrixCSR *D, HPRQPModel *model)
{
    if (!Q || !finite_matrix(D) || !C || C->nrow!=D->nrow || C->ncol!=D->ncol || D->ncol!=Q->ncol)
        return HPRQP_STATUS_INVALID_ARGUMENT;
    if (D->nrow>=SIZE_MAX/sizeof(size_t)) return HPRQP_STATUS_INVALID_ARGUMENT;
    size_t rank=0;
    for(size_t k=0;k<D->nnz;++k) {if(D->values[k]<0) return HPRQP_STATUS_INVALID_ARGUMENT; if(D->values[k]>0)++rank;}
    HPRQPMatrixCSR R={0};HPRQPStatus status=matrix_csr_alloc(&R,rank,D->ncol,rank);
    if(status!=HPRQP_STATUS_OK) return status;
    size_t *offsets=(size_t*)malloc((D->nrow+1)*sizeof(size_t));
    if(!offsets) {matrix_csr_free(&R);return HPRQP_STATUS_ALLOCATION_FAILED;}
    size_t row=0;R.row_ptr[0]=0;
    for(size_t i=0;i<D->nrow;++i) {
        offsets[i]=row;
        for(size_t k=D->row_ptr[i];k<D->row_ptr[i+1];++k) if(D->values[k]>0) {
            R.col_ind[row]=D->col_ind[k];R.values[row]=sqrt(D->values[k]);++row;R.row_ptr[row]=row;
        }
    }
    offsets[D->nrow]=row;
    status=build_from_QAbc_qc_factors(Q,A,c,AL,AU,l,u,obj_constant,C,constant,&R,offsets,model);
    free(offsets);matrix_csr_free(&R);return status;
}
