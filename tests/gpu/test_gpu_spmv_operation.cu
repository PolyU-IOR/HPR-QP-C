#include <cuda_runtime_api.h>
#if CUDART_VERSION >= 13030
#define CUSPARSE_ENABLE_EXPERIMENTAL_API
#endif
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#if CUDART_VERSION >= 13030
static bool fail_plan = false;
template<class... Args> static cusparseStatus_t test_create_plan(Args... args) {
    if (fail_plan) return CUSPARSE_STATUS_ALLOC_FAILED;
    return cusparseSpMVOp_createPlan(args...);
}
#define cusparseSpMVOp_createPlan test_create_plan
#endif
#include "spmv_operation.h"
#if CUDART_VERSION >= 13030
#undef cusparseSpMVOp_createPlan
#endif
#define OK(call) do { if ((call) != 0) { std::fprintf(stderr,"failed %s at %d\n",#call,__LINE__); std::exit(1); } } while(0)
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"failed %s at %d\n",#x,__LINE__); std::exit(1); } } while(0)
int main() {
    int rp[]={0,2,3,5}, ci[]={0,2,1,0,2};
    double a[]={2,-1,3,4,5}, x[]={1,2,3}, x2[]={-2,1,4}, got[3];
    int *dr,*dc; double *da,*dx,*dx2,*dy;
    OK(cudaMalloc(&dr,sizeof(rp))); OK(cudaMalloc(&dc,sizeof(ci))); OK(cudaMalloc(&da,sizeof(a)));
    OK(cudaMalloc(&dx,sizeof(x))); OK(cudaMalloc(&dx2,sizeof(x2))); OK(cudaMalloc(&dy,sizeof(got)));
    OK(cudaMemcpy(dr,rp,sizeof(rp),cudaMemcpyHostToDevice)); OK(cudaMemcpy(dc,ci,sizeof(ci),cudaMemcpyHostToDevice));
    OK(cudaMemcpy(da,a,sizeof(a),cudaMemcpyHostToDevice)); OK(cudaMemcpy(dx,x,sizeof(x),cudaMemcpyHostToDevice)); OK(cudaMemcpy(dx2,x2,sizeof(x2),cudaMemcpyHostToDevice));
    cusparseHandle_t h; cusparseSpMatDescr_t m; cusparseDnVecDescr_t vx,vx2,vy;
    OK(cusparseCreate(&h)); OK(cusparseCreateCsr(&m,3,3,5,dr,dc,da,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
    OK(cusparseCreateDnVec(&vx,3,dx,CUDA_R_64F)); OK(cusparseCreateDnVec(&vx2,3,dx2,CUDA_R_64F)); OK(cusparseCreateDnVec(&vy,3,dy,CUDA_R_64F));
    cudaStream_t stream; OK(cudaStreamCreate(&stream)); OK(cusparseSetStream(h,stream));
    {
        HPRQPSpMVOperation op;
        REQUIRE(op.apply(h,m,vx,vy)==CUSPARSE_STATUS_NOT_INITIALIZED);
        for (int mode=0;mode<3;++mode) {
#if HPRQP_HAS_SPMVOP
            fail_plan = mode==2;
#endif
            OK(op.prepare(h,m,vx,vy,mode!=0));
            REQUIRE(op.uses_spmvop()==(HPRQP_HAS_SPMVOP && mode==1));
#if HPRQP_HAS_SPMVOP
            if(mode==2) REQUIRE(op.fallback_status()==CUSPARSE_STATUS_ALLOC_FAILED);
#endif
            OK(op.apply(h,m,vx,vy)); OK(cudaStreamSynchronize(stream));
            OK(cudaMemcpy(got,dy,sizeof(got),cudaMemcpyDeviceToHost));
            REQUIRE(got[0]==-1 && got[1]==6 && got[2]==19);
            // The production solver uses multiple vector descriptors per plan.
            OK(op.apply(h,m,vx2,vy)); OK(cudaStreamSynchronize(stream));
            OK(cudaMemcpy(got,dy,sizeof(got),cudaMemcpyDeviceToHost));
            REQUIRE(got[0]==-8 && got[1]==3 && got[2]==12);
            cudaGraph_t graph; cudaGraphExec_t exec;
            OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal)); OK(op.apply(h,m,vx,vy));
            OK(cudaStreamEndCapture(stream,&graph)); OK(cudaGraphInstantiate(&exec,graph,0));
            OK(cudaGraphLaunch(exec,stream)); OK(cudaStreamSynchronize(stream));
            OK(cudaMemcpy(got,dy,sizeof(got),cudaMemcpyDeviceToHost)); REQUIRE(got[0]==-1 && got[1]==6 && got[2]==19);
            OK(cudaGraphExecDestroy(exec)); OK(cudaGraphDestroy(graph));
        }
        op.reset(); op.reset(); REQUIRE(!op.buffer() && op.buffer_size()==0);
    }
    OK(cusparseDestroyDnVec(vx)); OK(cusparseDestroyDnVec(vx2)); OK(cusparseDestroyDnVec(vy)); OK(cusparseDestroySpMat(m)); OK(cusparseDestroy(h)); OK(cudaStreamDestroy(stream));
    OK(cudaFree(dr)); OK(cudaFree(dc)); OK(cudaFree(da)); OK(cudaFree(dx)); OK(cudaFree(dx2)); OK(cudaFree(dy));
    std::printf("SpMV operation: FP64, changed vectors, graph, reprepare, cleanup and fallback PASS; SpMVOp compiled=%d\n",HPRQP_HAS_SPMVOP);
    return 0;
}
