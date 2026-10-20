#include "shared/diagnostics.h"
#ifndef HPRQP_HAVE_CUDSS
#define HPRQP_HAVE_CUDSS 0
#endif
#if HPRQP_HAVE_CUDSS
#include "cuda_common.h"
#include "gpu/qp_row_metric.h"
#include <cudss.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <climits>
#include <new>

struct QPMetric {
    int e=0,n=0,m=0; size_t nnz=0,calls=0,quadratics=0;
    bool ready=false,fatal=false; const char* reason="setup";
    double beta=1,setup_s=0,factor_s=0,backward=0,relative=0;
    int *rp=nullptr,*ci=nullptr,*trp=nullptr,*tci=nullptr,*eq=nullptr,*map=nullptr;
    double *val=nullptr,*tv=nullptr,*rhs=nullptr,*sol=nullptr,*tmp=nullptr,*rest=nullptr,*b=nullptr;
    cudssHandle_t handle=nullptr; cudssConfig_t config=nullptr; cudssData_t data=nullptr;
    cudssMatrix_t matrix=nullptr,rin=nullptr,sout=nullptr;
};
static double qp_now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
#define QP_CUDA(x) do{cudaError_t z=(x);if(z!=cudaSuccess){c->reason="cuda_setup_failure";c->fatal=(z!=cudaErrorMemoryAllocation);return map_cuda_status(z);}}while(0)
#define QP_DSS(x) do{cudssStatus_t z=(x);if(z!=CUDSS_STATUS_SUCCESS){c->reason="factor_or_solve_rejected";hprqp_diagnostic("QP_GRAM_CUDSS_ERROR line=%d status=%d\n",__LINE__,int(z));return HPRQP_STATUS_NOT_IMPLEMENTED;}}while(0)
static bool qp_trace(const QPMetric*c){return c->n<=128&&std::getenv("HPRQP_GRAM_TRACE");}
static void qp_vec(const char*name,const double*v,size_t n){std::vector<double>a(n);if(n)cudaMemcpy(a.data(),v,n*sizeof(double),cudaMemcpyDeviceToHost);std::fprintf(stderr,",\"%s\":[",name);for(size_t i=0;i<n;i++)std::fprintf(stderr,"%s%.17g",i?",":"",a[i]);std::fprintf(stderr,"]");}
static void qp_host(const char*name,const std::vector<double>&v){std::fprintf(stderr,",\"%s\":[",name);for(size_t i=0;i<v.size();i++){if(i)std::fprintf(stderr,",");if(std::isfinite(v[i]))std::fprintf(stderr,"%.17g",v[i]);else std::fprintf(stderr,"null");}std::fprintf(stderr,"]");}
int gpu_qp_metric_enabled(const HPRQPGPUWorkspace*w){auto*c=(QPMetric*)w->qp_row_metric_ctx;return c&&c->ready;}
void gpu_qp_metric_free(HPRQPGPUWorkspace*w){
    auto*c=(QPMetric*)w->qp_row_metric_ctx;if(!c)return;
    if(c->ready)hprqp_diagnostic("QP_GRAM_COST setup_s=%.9g factor_s=%.9g iterations=%zu quadratics=%zu equality_rows=%d gram_nnz=%zu beta=%.17g backward=%.17g rhs_relative=%.17g\n",c->setup_s,c->factor_s,c->calls,c->quadratics,c->e,c->nnz,c->beta,c->backward,c->relative);
    if(c->matrix)cudssMatrixDestroy(c->matrix);if(c->rin)cudssMatrixDestroy(c->rin);if(c->sout)cudssMatrixDestroy(c->sout);
    if(c->data)cudssDataDestroy(c->handle,c->data);if(c->config)cudssConfigDestroy(c->config);if(c->handle)cudssDestroy(c->handle);
    for(double*p:{c->val,c->tv,c->rhs,c->sol,c->tmp,c->rest,c->b})if(p)cudaFree(p);
    for(int*p:{c->rp,c->ci,c->trp,c->tci,c->eq,c->map})if(p)cudaFree(p);
    delete c;w->qp_row_metric_ctx=nullptr;
}
__global__ static void qp_gather_rhs(const double*ax,const int*eq,const double*b,double*rhs,int e){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<e)rhs[j]=ax[eq[j]]-b[j];}
__global__ static void qp_finish_y(HPRQPGPUWorkspace w,const int*map,const double*step,double a,double b){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w.m)return;int j=map[i];
    double y=w.y.values[i];double yb=j>=0?y-step[j]/(2*w.sigma):(w.s.values[i]-w.Rp.values[i])/(w.sigma*w.lambda_max_A);
    w.dy.values[i]=yb-y;w.y_bar.values[i]=yb;w.y.values[i]=a*w.last_y.values[i]+b*(2*yb-y);
}
__global__ static void qp_et(const int*rp,const int*ci,const double*v,const double*y,double*out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double z=0;for(int j=rp[i];j<rp[i+1];j++)z+=v[j]*y[ci[j]];out[i]=z;}
__global__ static void qp_rest(const double*dy,const int*map,double*out,int m){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m)out[i]=map[i]<0?dy[i]:0;}
HPRQPStatus gpu_qp_metric_finish_y(HPRQPGPUWorkspace*w,double a,double b){
    auto*c=(QPMetric*)w->qp_row_metric_ctx;
    qp_gather_rhs<<<(c->e+255)/256,256>>>(w->Ax.values,c->eq,c->b,c->rhs,c->e);
    QP_DSS(cudssExecute(c->handle,CUDSS_PHASE_SOLVE,c->config,c->data,c->matrix,c->sout,c->rin));
    if(qp_trace(c)&&c->calls<4){std::fprintf(stderr,"HPRQP_GRAM_TRACE {\"stage\":\"row\",\"sigma\":%.17g,\"beta\":%.17g",w->sigma,c->beta);qp_vec("rhs",c->rhs,c->e);qp_vec("step",c->sol,c->e);qp_vec("old_y",w->y.values,c->m);qp_vec("slack",w->s.values,c->m);qp_vec("raw",w->Rp.values,c->m);}
    qp_finish_y<<<w->blocks_m,w->threads_m>>>(*w,c->map,c->sol,a,b);
    if(qp_trace(c)&&c->calls<4){qp_vec("yb",w->y_bar.values,c->m);std::fprintf(stderr,"}\n");}
    c->calls++;return map_cuda_status(cudaGetLastError());
}
HPRQPStatus gpu_qp_metric_quadratic(HPRQPGPUWorkspace*w,double*out){
    auto*c=(QPMetric*)w->qp_row_metric_ctx;double eq=0,rest=0;auto h=(cublasHandle_t)w->blas_handle;
    qp_et<<<w->blocks_n,w->threads_n>>>(c->trp,c->tci,c->tv,w->dy.values,c->tmp,c->n);
    qp_rest<<<w->blocks_m,w->threads_m>>>(w->dy.values,c->map,c->rest,c->m);
    HPRQPStatus s=map_cublas_status(cublasDdot(h,c->n,c->tmp,1,c->tmp,1,&eq));if(s!=HPRQP_STATUS_OK)return s;
    s=map_cublas_status(cublasDdot(h,c->m,c->rest,1,c->rest,1,&rest));if(s!=HPRQP_STATUS_OK)return s;
    *out=2*eq+2*c->beta*rest;
    if(qp_trace(c)&&c->quadratics<8){std::fprintf(stderr,"HPRQP_GRAM_TRACE {\"stage\":\"quadratic\",\"value\":%.17g",*out);qp_vec("dy",w->dy.values,c->m);std::fprintf(stderr,"}\n");}
    c->quadratics++;return std::isfinite(*out)?HPRQP_STATUS_OK:HPRQP_STATUS_INVALID_ARGUMENT;
}

static HPRQPStatus qp_build(HPRQPGPUWorkspace*w,const HPRQPGPUProblem*p,QPMetric*c){
    int m=w->m,n=w->n;c->m=m;c->n=n;double tick=qp_now();
    std::vector<int>ar(m+1),ac(p->A.nnz),eq,map(m,-1);std::vector<double>av(p->A.nnz),al(m),au(m),b;
    QP_CUDA(cudaMemcpy(ar.data(),p->A.row_ptr,(m+1)*sizeof(int),cudaMemcpyDeviceToHost));
    if(ac.size()){QP_CUDA(cudaMemcpy(ac.data(),p->A.col_ind,ac.size()*sizeof(int),cudaMemcpyDeviceToHost));QP_CUDA(cudaMemcpy(av.data(),p->A.values,av.size()*sizeof(double),cudaMemcpyDeviceToHost));}
    QP_CUDA(cudaMemcpy(al.data(),p->AL.values,m*sizeof(double),cudaMemcpyDeviceToHost));QP_CUDA(cudaMemcpy(au.data(),p->AU.values,m*sizeof(double),cudaMemcpyDeviceToHost));
    for(int i=0;i<m;i++)if(std::isfinite(al[i])&&al[i]==au[i]){map[i]=eq.size();eq.push_back(i);b.push_back(al[i]);}
    c->e=eq.size();int e=c->e;
    if(!e){c->reason="no_equalities";return HPRQP_STATUS_NOT_IMPLEMENTED;}
    if(e>n){c->reason="more_equalities_than_variables";return HPRQP_STATUS_NOT_IMPLEMENTED;}
    std::vector<double>cols(n,0);double rowmax=0;long double bound_rest=0;
    for(int i=0;i<m;i++)if(map[i]<0){double row=0;for(int k=ar[i];k<ar[i+1];k++){row+=std::fabs(av[k]);cols[ac[k]]+=std::fabs(av[k]);}rowmax=std::max(rowmax,row);double v=std::max(std::isfinite(al[i])?std::fabs(al[i]):0.,std::isfinite(au[i])?std::fabs(au[i]):0.);bound_rest+=(long double)v*v;}
    c->beta=(1+1e-12)*rowmax*(*std::max_element(cols.begin(),cols.end()));if(c->beta==0)c->beta=1;
    if(!std::isfinite(c->beta)){c->reason="nonfinite_majorant";return HPRQP_STATUS_INVALID_ARGUMENT;}
    std::vector<int>tr(n+1,0);
    for(int i:eq)for(int k=ar[i];k<ar[i+1];k++)tr[ac[k]+1]++;
    unsigned long long work=0;for(int j=0;j<n;j++)work+=(unsigned long long)tr[j+1]*tr[j+1];
    if(work>100000000ULL){c->reason="gram_assembly_work_limit";return HPRQP_STATUS_NOT_IMPLEMENTED;}
    for(int j=0;j<n;j++)tr[j+1]+=tr[j];std::vector<int>tc(tr[n]);std::vector<double>tv(tr[n]);auto cursor=tr;
    for(int i:eq)for(int k=ar[i];k<ar[i+1];k++){int z=cursor[ac[k]]++;tc[z]=i;tv[z]=av[k];}
    std::vector<int>gr(e+1),gc;std::vector<double>gv;std::vector<std::pair<int,double>>pairs;
    for(int q=0;q<e;q++){
        int i=eq[q];gr[q]=gc.size();pairs.clear();
        for(int k=ar[i];k<ar[i+1];k++)for(int j=tr[ac[k]];j<tr[ac[k]+1];j++)pairs.emplace_back(map[tc[j]],av[k]*tv[j]);
        std::sort(pairs.begin(),pairs.end(),[](auto&a,auto&b){return a.first<b.first;});
        for(size_t j=0;j<pairs.size();){int col=pairs[j].first;double sum=0;do{sum+=pairs[j++].second;}while(j<pairs.size()&&pairs[j].first==col);if(sum!=0){gc.push_back(col);gv.push_back(sum);}}
    }gr[e]=gc.size();c->nnz=gc.size();
    for(auto pair:{std::make_pair(&c->rp,&gr),std::make_pair(&c->ci,&gc),std::make_pair(&c->trp,&tr),std::make_pair(&c->tci,&tc),std::make_pair(&c->eq,&eq),std::make_pair(&c->map,&map)}){QP_CUDA(cudaMalloc(pair.first,pair.second->size()*sizeof(int)));QP_CUDA(cudaMemcpy(*pair.first,pair.second->data(),pair.second->size()*sizeof(int),cudaMemcpyHostToDevice));}
    for(auto pair:{std::make_pair(&c->val,&gv),std::make_pair(&c->tv,&tv),std::make_pair(&c->b,&b)}){QP_CUDA(cudaMalloc(pair.first,pair.second->size()*sizeof(double)));QP_CUDA(cudaMemcpy(*pair.first,pair.second->data(),pair.second->size()*sizeof(double),cudaMemcpyHostToDevice));}
    QP_CUDA(cudaMalloc(&c->rhs,e*sizeof(double)));QP_CUDA(cudaMalloc(&c->sol,e*sizeof(double)));QP_CUDA(cudaMalloc(&c->tmp,n*sizeof(double)));QP_CUDA(cudaMalloc(&c->rest,m*sizeof(double)));
    QP_DSS(cudssCreate(&c->handle));QP_DSS(cudssConfigCreate(&c->config));QP_DSS(cudssDataCreate(c->handle,&c->data));
    QP_DSS(cudssMatrixCreateCsr(&c->matrix,e,e,c->nnz,c->rp,nullptr,c->ci,c->val,CUDA_R_32I,CUDA_R_64F,CUDSS_MTYPE_SPD,CUDSS_MVIEW_UPPER,CUDSS_BASE_ZERO));
    QP_DSS(cudssMatrixCreateDn(&c->rin,e,1,e,c->rhs,CUDA_R_64F,CUDSS_LAYOUT_COL_MAJOR));QP_DSS(cudssMatrixCreateDn(&c->sout,e,1,e,c->sol,CUDA_R_64F,CUDSS_LAYOUT_COL_MAJOR));
    double ft=qp_now();QP_DSS(cudssExecute(c->handle,CUDSS_PHASE_ANALYSIS,c->config,c->data,c->matrix,c->sout,c->rin));
    QP_DSS(cudssExecute(c->handle,CUDSS_PHASE_FACTORIZATION,c->config,c->data,c->matrix,c->sout,c->rin));QP_CUDA(cudaDeviceSynchronize());c->factor_s=qp_now()-ft;
    int info=0;size_t written=0;QP_DSS(cudssDataGet(c->handle,c->data,CUDSS_DATA_INFO,&info,sizeof(info),&written));
    if(info){c->reason="gram_not_spd";return HPRQP_STATUS_NOT_IMPLEMENTED;}
    std::vector<double>test(e),sol(e);for(int i=0;i<e;i++)test[i]=std::sin(.173*(i+1))+.2*std::cos(.719*(i+1));
    QP_CUDA(cudaMemcpy(c->rhs,test.data(),e*sizeof(double),cudaMemcpyHostToDevice));QP_DSS(cudssExecute(c->handle,CUDSS_PHASE_SOLVE,c->config,c->data,c->matrix,c->sout,c->rin));
    QP_CUDA(cudaMemcpy(sol.data(),c->sol,e*sizeof(double),cudaMemcpyDeviceToHost));double err=0,an=0,xn=0,bn=0;
    for(int i=0;i<e;i++){if(!std::isfinite(sol[i])){c->reason="nonfinite_test_solve";return HPRQP_STATUS_INVALID_ARGUMENT;}long double z=0;double rowsum=0;for(int k=gr[i];k<gr[i+1];k++){z+=(long double)gv[k]*sol[gc[k]];rowsum+=std::fabs(gv[k]);}err=std::max(err,(double)std::fabs(z-test[i]));an=std::max(an,rowsum);xn=std::max(xn,std::fabs(sol[i]));bn=std::max(bn,std::fabs(test[i]));}
    c->backward=err/(an*xn+bn);c->relative=err/bn;
    if(!(c->backward<=1e-10)||(e<=5000&&!(c->relative<=1e-8))){c->reason="inverse_validation";return HPRQP_STATUS_INVALID_ARGUMENT;}
    QP_CUDA(cudaMemcpy(c->rhs,b.data(),e*sizeof(double),cudaMemcpyHostToDevice));QP_DSS(cudssExecute(c->handle,CUDSS_PHASE_SOLVE,c->config,c->data,c->matrix,c->sout,c->rin));
    QP_CUDA(cudaMemcpy(sol.data(),c->sol,e*sizeof(double),cudaMemcpyDeviceToHost));long double bquad=0;for(int i=0;i<e;i++)bquad+=(long double)b[i]*sol[i];
    if(!std::isfinite((double)bquad)||bquad < -1e-10){c->reason="rhs_metric_norm";return HPRQP_STATUS_INVALID_ARGUMENT;}
    double normb=std::sqrt((double)(std::max(bquad,0.L)+bound_rest/c->beta)),normc=0;
    HPRQPStatus bs=map_cublas_status(cublasDnrm2((cublasHandle_t)w->blas_handle,n,p->c.values,1,&normc));if(bs!=HPRQP_STATUS_OK)return bs;
    double sigma=(normb>1e-16&&normc>1e-16&&normb<1e16&&normc<1e16)?normb/normc:1.;
    c->setup_s=qp_now()-tick;c->ready=true;w->sigma=sigma;w->lambda_max_A=2*c->beta;w->noC=0;
    hprqp_diagnostic("QP_GRAM_READY equality_rows=%d other_rows=%d n=%d gram_nnz=%zu beta=%.17g sigma=%.17g setup_s=%.9g factor_s=%.9g backward=%.17g rhs_relative=%.17g\n",e,m-e,n,c->nnz,c->beta,sigma,c->setup_s,c->factor_s,c->backward,c->relative);
    if(qp_trace(c)){
        std::fprintf(stderr,"HPRQP_GRAM_TRACE {\"stage\":\"setup\",\"e\":%d,\"n\":%d,\"m\":%d,\"beta\":%.17g,\"sigma\":%.17g,\"eq\":[",e,n,m,c->beta,sigma);
        for(int i=0;i<e;i++)std::fprintf(stderr,"%s%d",i?",":"",eq[i]);std::fprintf(stderr,"],\"gram\":[");
        for(int i=0;i<e;i++)for(int j=0;j<e;j++){double v=0;for(int k=gr[i];k<gr[i+1];k++)if(gc[k]==j)v=gv[k];std::fprintf(stderr,"%s%.17g",i||j?",":"",v);}std::fprintf(stderr,"]");qp_vec("b",c->b,e);qp_vec("c",p->c.values,n);qp_host("al",al);qp_host("au",au);std::vector<double>denseA(m*n,0);for(int i=0;i<m;i++)for(int j=ar[i];j<ar[i+1];j++)denseA[i*n+ac[j]]=av[j];qp_host("A",denseA);std::fprintf(stderr,"}\n");
    }
    return HPRQP_STATUS_OK;
}
HPRQPStatus gpu_qp_metric_setup(HPRQPGPUWorkspace*w,const HPRQPGPUProblem*p,const HPRQPParams*params){
    if(params->gram_mode==HPRQP_GRAM_OFF)return HPRQP_STATUS_OK;
    if(p->num_row_cones||p->num_var_cones)return HPRQP_STATUS_OK;
    if(!w->m||!w->n||w->empty_A){hprqp_diagnostic("QP_GRAM_SKIP reason=no_equalities\n");return HPRQP_STATUS_OK;}
    double tick=qp_now();auto*c=new(std::nothrow) QPMetric;if(!c){hprqp_diagnostic("QP_GRAM_SKIP reason=host_allocation\n");return HPRQP_STATUS_OK;}
    w->qp_row_metric_ctx=c;HPRQPStatus status;
    try{status=qp_build(w,p,c);}catch(const std::bad_alloc&){c->reason="host_allocation";status=HPRQP_STATUS_ALLOCATION_FAILED;}
    if(status!=HPRQP_STATUS_OK){hprqp_diagnostic("QP_GRAM_SKIP reason=%s setup_s=%.9g status=%d\n",c->reason,qp_now()-tick,int(status));bool fatal=c->fatal;gpu_qp_metric_free(w);cudaGetLastError();if(fatal)return status;}
    return HPRQP_STATUS_OK;
}

#else
#include "gpu/qp_row_metric.h"
#include <cstdio>
int gpu_qp_metric_enabled(const HPRQPGPUWorkspace*){return 0;}
void gpu_qp_metric_free(HPRQPGPUWorkspace*){}
HPRQPStatus gpu_qp_metric_setup(HPRQPGPUWorkspace*,const HPRQPGPUProblem*p,const HPRQPParams*params){
    if(params->gram_mode!=HPRQP_GRAM_OFF && !p->num_row_cones)
        hprqp_diagnostic("GRAM_SKIP reason=cudss_not_built\n");
    return HPRQP_STATUS_OK;
}
HPRQPStatus gpu_qp_metric_finish_y(HPRQPGPUWorkspace*,double,double){return HPRQP_STATUS_NOT_IMPLEMENTED;}
HPRQPStatus gpu_qp_metric_quadratic(HPRQPGPUWorkspace*,double*){return HPRQP_STATUS_NOT_IMPLEMENTED;}
#endif
