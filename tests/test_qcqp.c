#include "hprqp.h"
#include "shared/sparse.h"
#include "cpu/control.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define REQUIRE(c) do{if(!(c)){fprintf(stderr,"check failed %s:%d %s\n",__FILE__,__LINE__,#c);exit(1);}}while(0)
static double dotrow(HPRQPMatrixCSR *a,size_t row,double *x){double v=0;for(size_t j=a->row_ptr[row];j<a->row_ptr[row+1];++j)v+=a->values[j]*x[a->col_ind[j]];return v;}
int main(void){
    size_t zero[]={0,0,0,0},dummy[]={0},cp[]={0,2,3},cj[]={0,2,1},rp[]={0,2,4,5},rj[]={0,1,1,2,0},off[]={0,2,3};
    double none[]={0},cv[]={2,-1,.5},rv[]={1,2,-3,4,2},cc[]={0,0,0},b[]={-3,2},l[]={-INFINITY,-INFINITY,-INFINITY},u[]={INFINITY,INFINITY,INFINITY};
    HPRQPMatrixCSR Q={3,3,0,zero,dummy,none,0},A={0,3,0,zero,dummy,none,0},C={2,3,3,cp,cj,cv,0},R={3,3,5,rp,rj,rv,0};
    HPRQPModel model={0};
    REQUIRE(build_from_QAbc_qc_factors(&Q,&A,cc,NULL,NULL,l,u,7,&C,b,&R,off,&model)==HPRQP_STATUS_OK);
    REQUIRE(model.Q.ncol==3 && model.A.nrow==7 && model.A.nnz==2*C.nnz+R.nnz);
    REQUIRE(model.row_cone_offsets[0]==0&&model.row_cone_offsets[1]==4&&model.row_cone_offsets[2]==7);
    for(int t=-20;t<=20;++t){double x[]={t*.07,t*.03+.1,.2-t*.02};
        for(size_t i=0;i<2;++i){double h=dotrow(&C,i,x)+b[i];for(size_t j=off[i];j<off[i+1];++j){double v=dotrow(&R,j,x);h+=.5*v*v;}
            size_t first=model.row_cone_offsets[i],end=model.row_cone_offsets[i+1];double head=dotrow(&model.A,first,x)-model.soc_rhs[first],sq=0;
            for(size_t j=first+1;j<end;++j){double v=dotrow(&model.A,j,x)-model.soc_rhs[j];sq+=v*v;}
            REQUIRE(fabs((head*head-sq)+4*h)<1e-11*(1+fabs(h)));
            REQUIRE((h<=0)==(head>=sqrt(sq)));
        }}
    model_free(&model);
    rv[0]=NAN;REQUIRE(build_from_QAbc_qc_factors(&Q,&A,cc,NULL,NULL,l,u,0,&C,b,&R,off,&model)==HPRQP_STATUS_INVALID_ARGUMENT);rv[0]=1;
    puts("QC sparse-factor algebra, no dimension inflation and invalid data checks passed");return 0;
}
