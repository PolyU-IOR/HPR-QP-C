#ifndef HPRQP_HAVE_CUDSS
#define HPRQP_HAVE_CUDSS 0
#endif
#if HPRQP_HAVE_CUDSS
#include "cuda_common.h"
#include "gpu/row_metric.h"
#include <cudss.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <climits>

struct SpatialGram;
struct RowMetric {
    SpatialGram* spatial=nullptr;
    int l=0,n=0,m=0;size_t nnz=0,calls=0,quadratics=0;
    double beta=0,setup_s=0,factor_s=0,backward=0,relative=0;
    int *rp=nullptr,*ci=nullptr,*trp=nullptr,*tci=nullptr;
    double *val=nullptr,*tv=nullptr,*rhs=nullptr,*sol=nullptr,*tmp=nullptr;
    cudssHandle_t handle=nullptr;cudssConfig_t config=nullptr;cudssData_t data=nullptr;
    cudssMatrix_t matrix=nullptr,rin=nullptr,sout=nullptr;
};
#define RM_TRY(x) do{HPRQPStatus s_=(x);if(s_!=HPRQP_STATUS_OK)return s_;}while(0)
#define RM_CUDA(x) RM_TRY(map_cuda_status(x))
#define RM_BLAS(x) RM_TRY(map_cublas_status(x))
#define RM_DSS(x) do{cudssStatus_t s_=(x);if(s_!=CUDSS_STATUS_SUCCESS){std::fprintf(stderr,"ROW_METRIC_CUDSS_ERROR line=%d status=%d\n",__LINE__,int(s_));return HPRQP_STATUS_NOT_IMPLEMENTED;}}while(0)
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
#include "spatial_gram.cuh"
static bool trace(const RowMetric*c){return c->n<=128&&std::getenv("HPRQP_GRAM_TRACE");}
static void vec(const char*name,const double*v,size_t n){std::vector<double>a(n);cudaMemcpy(a.data(),v,n*sizeof(double),cudaMemcpyDeviceToHost);std::fprintf(stderr,",\"%s\":[",name);for(size_t i=0;i<n;i++)std::fprintf(stderr,"%s%.17g",i?",":"",a[i]);std::fprintf(stderr,"]");}
int gpu_row_metric_enabled(const HPRQPGPUWorkspace*w){return w->row_metric_ctx!=nullptr;}
void gpu_row_metric_free(HPRQPGPUWorkspace*w){
    auto*c=(RowMetric*)w->row_metric_ctx;if(!c)return;cudaDeviceSynchronize();
    std::fprintf(stderr,"ROW_METRIC_COST setup_s=%.9g factor_s=%.9g iterations=%zu quadratics=%zu equality_rows=%d gram_nnz=%zu beta=%.17g backward=%.17g rhs_relative=%.17g\n",c->setup_s,c->factor_s,c->calls,c->quadratics,c->l,c->nnz,c->beta,c->backward,c->relative);
    sg_free(c->spatial);
    if(c->matrix)cudssMatrixDestroy(c->matrix);if(c->rin)cudssMatrixDestroy(c->rin);if(c->sout)cudssMatrixDestroy(c->sout);
    if(c->data)cudssDataDestroy(c->handle,c->data);if(c->config)cudssConfigDestroy(c->config);if(c->handle)cudssDestroy(c->handle);
    for(double*p:{c->val,c->tv,c->rhs,c->sol,c->tmp})cudaFree(p);
    for(int*p:{c->rp,c->ci,c->trp,c->tci})cudaFree(p);delete c;w->row_metric_ctx=nullptr;
}
__global__ static void finish_y(HPRQPGPUWorkspace w,const double*step,int l,double a,double b){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w.m)return;
    double y=w.y.values[i];double yb=i<l?y-step[i]/(2*w.sigma):(w.s.values[i]-w.Rp.values[i])/(w.sigma*w.lambda_max_A);
    w.dy.values[i]=yb-y;w.y_bar.values[i]=yb;w.y.values[i]=a*w.last_y.values[i]+b*(2*yb-y);
}
__global__ static void et_product(const int*rp,const int*ci,const double*v,const double*y,double*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double z=0;for(int j=rp[i];j<rp[i+1];j++)z+=v[j]*y[ci[j]];out[i]=z;
}
HPRQPStatus gpu_row_metric_finish_y(HPRQPGPUWorkspace*w,double a,double b){
    auto*c=(RowMetric*)w->row_metric_ctx;
    RM_CUDA(cudaMemcpy(c->rhs,w->Ax.values,c->l*sizeof(double),cudaMemcpyDeviceToDevice));
    if(c->spatial){RM_TRY(sg_solve(c->spatial,c->rhs,c->sol));}
    else {RM_DSS(cudssExecute(c->handle,CUDSS_PHASE_SOLVE,c->config,c->data,c->matrix,c->sout,c->rin));}
    if(trace(c)&&c->calls<4){std::fprintf(stderr,"HPRQP_GRAM_TRACE {\"stage\":\"row\",\"sigma\":%.17g,\"beta\":%.17g",w->sigma,c->beta);vec("rhs",c->rhs,c->l);vec("step",c->sol,c->l);vec("old_y",w->y.values,w->m);vec("slack",w->s.values,w->m);vec("raw",w->Rp.values,w->m);}
    finish_y<<<w->blocks_m,w->threads_m>>>(*w,c->sol,c->l,a,b);
    if(trace(c)&&c->calls<4){vec("yb",w->y_bar.values,w->m);std::fprintf(stderr,"}\n");}
    c->calls++;return map_cuda_status(cudaGetLastError());
}
HPRQPStatus gpu_row_metric_quadratic(HPRQPGPUWorkspace*w,double*out){
    auto*c=(RowMetric*)w->row_metric_ctx;double eq=0,cone=0;auto h=(cublasHandle_t)w->blas_handle;
    et_product<<<w->blocks_n,w->threads_n>>>(c->trp,c->tci,c->tv,w->dy.values,c->tmp,c->n);
    RM_BLAS(cublasDdot(h,c->n,c->tmp,1,c->tmp,1,&eq));
    RM_BLAS(cublasDdot(h,c->m-c->l,w->dy.values+c->l,1,w->dy.values+c->l,1,&cone));
    *out=2*eq+2*c->beta*cone;
    if(trace(c)&&c->quadratics<8){std::fprintf(stderr,"HPRQP_GRAM_TRACE {\"stage\":\"quadratic\",\"value\":%.17g",*out);vec("dy",w->dy.values,w->m);std::fprintf(stderr,"}\n");}
    c->quadratics++;return std::isfinite(*out)?HPRQP_STATUS_OK:HPRQP_STATUS_INVALID_ARGUMENT;
}
static HPRQPStatus row_metric_build(HPRQPGPUWorkspace*w,const HPRQPGPUProblem*p,const HPRQPParams*params){
    if(params->gram_mode==HPRQP_GRAM_OFF || !p->num_row_cones)return HPRQP_STATUS_OK;
    if(!p->num_row_cones||p->num_var_cones||!p->linear_m||!w->Q_is_diag||w->noC)return HPRQP_STATUS_NOT_IMPLEMENTED;
    double t=now();auto*c=new RowMetric;w->row_metric_ctx=c;c->l=p->linear_m;c->n=w->n;c->m=w->m;
    int l=c->l,n=c->n,m=c->m;
    std::vector<int>ar(m+1),ac(p->A.nnz);std::vector<double>av(p->A.nnz),al(l),au(l),rhs(l);
    RM_CUDA(cudaMemcpy(ar.data(),p->A.row_ptr,(m+1)*sizeof(int),cudaMemcpyDeviceToHost));
    RM_CUDA(cudaMemcpy(ac.data(),p->A.col_ind,ac.size()*sizeof(int),cudaMemcpyDeviceToHost));RM_CUDA(cudaMemcpy(av.data(),p->A.values,av.size()*sizeof(double),cudaMemcpyDeviceToHost));
    RM_CUDA(cudaMemcpy(al.data(),p->AL.values,l*sizeof(double),cudaMemcpyDeviceToHost));RM_CUDA(cudaMemcpy(au.data(),p->AU.values,l*sizeof(double),cudaMemcpyDeviceToHost));RM_CUDA(cudaMemcpy(rhs.data(),p->soc_rhs.values,l*sizeof(double),cudaMemcpyDeviceToHost));
    for(int i=0;i<l;i++)if(al[i]!=0||au[i]!=0||rhs[i]!=0)return HPRQP_STATUS_NOT_IMPLEMENTED;
    std::vector<double>cols(n,0);double rowmax=0;
    for(int i=l;i<m;i++){double row=0;for(int k=ar[i];k<ar[i+1];k++){row+=std::fabs(av[k]);cols[ac[k]]+=std::fabs(av[k]);}rowmax=std::max(rowmax,row);}
    c->beta=(1+1e-12)*rowmax*(*std::max_element(cols.begin(),cols.end()));if(!(c->beta>0))return HPRQP_STATUS_INVALID_ARGUMENT;
    std::vector<int>tr(n+1,0),tc(ar[l]);std::vector<double>tv(ar[l]);
    for(int k=0;k<ar[l];k++)tr[ac[k]+1]++;for(int j=0;j<n;j++)tr[j+1]+=tr[j];auto cursor=tr;
    for(int i=0;i<l;i++)for(int k=ar[i];k<ar[i+1];k++){int z=cursor[ac[k]]++;tc[z]=i;tv[z]=av[k];}
    unsigned long long work=0;
    for(int j=0;j<n;j++)work+=(unsigned long long)(tr[j+1]-tr[j])*(tr[j+1]-tr[j]);
    if(work>100000000ULL)return HPRQP_STATUS_NOT_IMPLEMENTED;
    std::vector<int>gr(l+1),gc;std::vector<double>gv;std::vector<std::pair<int,double>>pairs;
    for(int i=0;i<l;i++){
        gr[i]=gc.size();pairs.clear();
        for(int k=ar[i];k<ar[i+1];k++)for(int j=tr[ac[k]];j<tr[ac[k]+1];j++)pairs.emplace_back(tc[j],av[k]*tv[j]);
        std::sort(pairs.begin(),pairs.end(),[](auto&a,auto&b){return a.first<b.first;});
        for(size_t j=0;j<pairs.size();){int col=pairs[j].first;double sum=0;do{sum+=pairs[j++].second;}while(j<pairs.size()&&pairs[j].first==col);if(sum!=0){gc.push_back(col);gv.push_back(sum);}}
        if(gc.size()>INT_MAX)return HPRQP_STATUS_NOT_IMPLEMENTED;
    }gr[l]=gc.size();c->nnz=gc.size();
    for(auto pair:{std::make_pair(&c->rp,&gr),std::make_pair(&c->ci,&gc),std::make_pair(&c->trp,&tr),std::make_pair(&c->tci,&tc)}){RM_CUDA(cudaMalloc(pair.first,pair.second->size()*sizeof(int)));RM_CUDA(cudaMemcpy(*pair.first,pair.second->data(),pair.second->size()*sizeof(int),cudaMemcpyHostToDevice));}
    for(auto pair:{std::make_pair(&c->val,&gv),std::make_pair(&c->tv,&tv)}){RM_CUDA(cudaMalloc(pair.first,pair.second->size()*sizeof(double)));RM_CUDA(cudaMemcpy(*pair.first,pair.second->data(),pair.second->size()*sizeof(double),cudaMemcpyHostToDevice));}
    RM_CUDA(cudaMalloc(&c->rhs,l*sizeof(double)));RM_CUDA(cudaMalloc(&c->sol,l*sizeof(double)));RM_CUDA(cudaMalloc(&c->tmp,n*sizeof(double)));
    RM_DSS(cudssCreate(&c->handle));RM_DSS(cudssConfigCreate(&c->config));RM_DSS(cudssDataCreate(c->handle,&c->data));
    RM_DSS(cudssMatrixCreateCsr(&c->matrix,l,l,c->nnz,c->rp,nullptr,c->ci,c->val,CUDA_R_32I,CUDA_R_64F,CUDSS_MTYPE_SPD,CUDSS_MVIEW_UPPER,CUDSS_BASE_ZERO));
    RM_DSS(cudssMatrixCreateDn(&c->rin,l,1,l,c->rhs,CUDA_R_64F,CUDSS_LAYOUT_COL_MAJOR));RM_DSS(cudssMatrixCreateDn(&c->sout,l,1,l,c->sol,CUDA_R_64F,CUDSS_LAYOUT_COL_MAJOR));
    double ft=now();RM_DSS(cudssExecute(c->handle,CUDSS_PHASE_ANALYSIS,c->config,c->data,c->matrix,c->sout,c->rin));
    RM_DSS(cudssExecute(c->handle,CUDSS_PHASE_FACTORIZATION,c->config,c->data,c->matrix,c->sout,c->rin));RM_CUDA(cudaDeviceSynchronize());c->factor_s=now()-ft;
    int info=0;size_t written=0;RM_DSS(cudssDataGet(c->handle,c->data,CUDSS_DATA_INFO,&info,sizeof(info),&written));
    if(info)return HPRQP_STATUS_NOT_IMPLEMENTED;
    std::vector<double>test(l),sol(l);for(int i=0;i<l;i++)test[i]=std::sin(.173*(i+1))+.2*std::cos(.719*(i+1));
    RM_CUDA(cudaMemcpy(c->rhs,test.data(),l*sizeof(double),cudaMemcpyHostToDevice));RM_DSS(cudssExecute(c->handle,CUDSS_PHASE_SOLVE,c->config,c->data,c->matrix,c->sout,c->rin));
    RM_CUDA(cudaMemcpy(sol.data(),c->sol,l*sizeof(double),cudaMemcpyDeviceToHost));double err=0,an=0,xn=0,bn=0;
    for(int i=0;i<l;i++){if(!std::isfinite(sol[i]))return HPRQP_STATUS_INVALID_ARGUMENT;long double z=0;double rowsum=0;for(int k=gr[i];k<gr[i+1];k++){z+=(long double)gv[k]*sol[gc[k]];rowsum+=std::fabs(gv[k]);}err=std::max(err,(double)std::fabs(z-test[i]));an=std::max(an,rowsum);xn=std::max(xn,std::fabs(sol[i]));bn=std::max(bn,std::fabs(test[i]));}
    c->backward=err/(an*xn+bn);c->relative=err/bn;
    if(!(c->backward<=1e-10)||(l<=5000&&!(c->relative<=1e-8)))return HPRQP_STATUS_INVALID_ARGUMENT;
    // Equivalent inverse access: automatic after the same Gram eligibility gate.
    const char* spatial_flag=std::getenv("HPRQP_SPATIAL_GRAM");
    if(!spatial_flag||std::strcmp(spatial_flag,"0")){
        HPRQPStatus spatial_status=sg_setup(c,ar,ac,av);
        if(spatial_status==HPRQP_STATUS_OK)spatial_status=sg_validate_profile(c,gr,gc,gv);
        if(spatial_status!=HPRQP_STATUS_OK){
            sg_free(c->spatial);c->spatial=nullptr;
            cudaError_t err=cudaGetLastError();
            if(err!=cudaSuccess&&err!=cudaErrorMemoryAllocation)return map_cuda_status(err);
            std::fprintf(stderr,"S724_SPATIAL_SKIP status=%d retaining=direct_gram\n",int(spatial_status));
        }
    }
    w->sigma/=std::sqrt(c->beta);w->lambda_max_A=2*c->beta;
    c->setup_s=now()-t;std::fprintf(stderr,"ROW_METRIC_READY equality_rows=%d n=%d gram_nnz=%zu beta=%.17g sigma=%.17g setup_s=%.9g factor_s=%.9g backward=%.17g rhs_relative=%.17g\n",l,n,c->nnz,c->beta,w->sigma,c->setup_s,c->factor_s,c->backward,c->relative);
    if(trace(c)){
        std::fprintf(stderr,"HPRQP_GRAM_TRACE {\"stage\":\"setup\",\"l\":%d,\"n\":%d,\"m\":%d,\"beta\":%.17g,\"gram\":[",l,n,m,c->beta);
        for(int i=0;i<l;i++)for(int j=0;j<l;j++){double v=0;for(int k=gr[i];k<gr[i+1];k++)if(gc[k]==j)v=gv[k];std::fprintf(stderr,"%s%.17g",i||j?",":"",v);}std::fprintf(stderr,"]}\n");
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_row_metric_setup(HPRQPGPUWorkspace*w,const HPRQPGPUProblem*p,const HPRQPParams*params){
    if(params->gram_mode==HPRQP_GRAM_OFF || !p->num_row_cones)return HPRQP_STATUS_OK;
    const double sigma=w->sigma,lambda=w->lambda_max_A;const int noc=w->noC;
    HPRQPStatus status;
    try{status=row_metric_build(w,p,params);}catch(const std::bad_alloc&){status=HPRQP_STATUS_ALLOCATION_FAILED;}
    if(status!=HPRQP_STATUS_OK){
        gpu_row_metric_free(w);w->sigma=sigma;w->lambda_max_A=lambda;w->noC=noc;
        cudaError_t err=cudaGetLastError();
        std::fprintf(stderr,"GRAM_SKIP reason=soc_setup_or_inverse_rejected status=%d\n",int(status));
        if(err!=cudaSuccess && err!=cudaErrorMemoryAllocation)return status;
        return HPRQP_STATUS_OK;
    }
    return status;
}

#else
#include "gpu/row_metric.h"
#include <cstdio>
int gpu_row_metric_enabled(const HPRQPGPUWorkspace*){return 0;}
void gpu_row_metric_free(HPRQPGPUWorkspace*){}
HPRQPStatus gpu_row_metric_setup(HPRQPGPUWorkspace*,const HPRQPGPUProblem*p,const HPRQPParams*params){
    if(params->gram_mode!=HPRQP_GRAM_OFF && p->num_row_cones)
        std::fprintf(stderr,"GRAM_SKIP reason=cudss_not_built\n");
    return HPRQP_STATUS_OK;
}
HPRQPStatus gpu_row_metric_finish_y(HPRQPGPUWorkspace*,double,double){return HPRQP_STATUS_NOT_IMPLEMENTED;}
HPRQPStatus gpu_row_metric_quadratic(HPRQPGPUWorkspace*,double*){return HPRQP_STATUS_NOT_IMPLEMENTED;}
#endif
