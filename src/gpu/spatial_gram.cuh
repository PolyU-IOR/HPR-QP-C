#include "shared/diagnostics.h"
// S698: exact equality Gram access, independent of primal box/contact masks.
#include <cufft.h>
#include <cusolverDn.h>
#include <fstream>
#include <iomanip>
#include <stdexcept>
struct SpatialGram {
 int N=0,L=0,m=0,k=0; double setup_s=0,structure_defect=0;
 cufftHandle forward=0,backward=0;cusolverDnHandle_t dense=nullptr;
 double *real=nullptr,*r=nullptr,*di=nullptr,*tmp=nullptr,*t=nullptr,*u=nullptr,*weight=nullptr,*schur=nullptr,*z=nullptr,*factor_work=nullptr;
 cufftDoubleComplex *freq=nullptr;int *idx=nullptr,*info=nullptr;
};
#define SG_FFT(x) do{auto e=(x);if(e!=CUFFT_SUCCESS){hprqp_diagnostic("S698_FFT_ERROR line=%d status=%d\n",__LINE__,int(e));return HPRQP_STATUS_NOT_IMPLEMENTED;}}while(0)
#define SG_DENSE(x) do{auto e=(x);if(e!=CUSOLVER_STATUS_SUCCESS){hprqp_diagnostic("S698_DENSE_ERROR line=%d status=%d\n",__LINE__,int(e));return HPRQP_STATUS_NOT_IMPLEMENTED;}}while(0)
static void sg_free(SpatialGram*s){if(!s)return;cudaDeviceSynchronize();if(s->forward)cufftDestroy(s->forward);if(s->backward)cufftDestroy(s->backward);if(s->dense)cusolverDnDestroy(s->dense);for(auto p:{s->real,s->r,s->di,s->tmp,s->t,s->u,s->weight,s->schur,s->z,s->factor_work})cudaFree(p);cudaFree(s->freq);cudaFree(s->idx);cudaFree(s->info);delete s;}
__global__ static void sg_extend(const double*x,double*out,int N,int L){int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=L*L)return;int i=z/L,j=z%L;if(i==0||j==0||i==N+1||j==N+1){out[z]=0;return;}int a=i<=N?i-1:L-i-1,b=j<=N?j-1:L-j-1;out[z]=((i<=N)==(j<=N)?1.:-1.)*x[a*N+b];}
__global__ static void sg_spectrum(cufftDoubleComplex*x,int N,int L){int z=blockIdx.x*blockDim.x+threadIdx.x;int width=L/2+1;if(z>=L*width)return;int i=z/width,j=z%width;if(i==0||j==0||i==N+1||j==N+1){x[z].x=0;x[z].y=0;return;}double a=sin(3.14159265358979323846*i/L),b=sin(3.14159265358979323846*j/L),d=4*(a*a+b*b);x[z].x/=d;x[z].y/=d;}
__global__ static void sg_extract(const double*x,double*out,int N,int L){int z=blockIdx.x*blockDim.x+threadIdx.x;if(z<N*N)out[z]=x[(z/N+1)*L+z%N+1]/((double)L*L);}
__global__ static void sg_scale(double*x,const double*d,int m){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m)x[i]*=d[i];}
__global__ static void sg_divide(const double*x,const double*r,double*out,int m){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m)out[i]=x[i]/r[i];}
__global__ static void sg_gather(const double*x,const int*idx,const double*w,double*out,int k,int diagonal){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<k)out[j]=w[j]*x[idx[j]]+(j==diagonal?1.:0.);}
__global__ static void sg_scatter(double*out,const int*idx,const double*w,const double*z,int k){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<k)out[idx[j]]-=w[j]*z[j];}
__global__ static void sg_unit(double*out,const int*idx,const double*w,int j){if(threadIdx.x==0&&blockIdx.x==0)out[idx[j]]=w[j];}
static HPRQPStatus sg_poisson(SpatialGram*s,const double*in,double*out){
 sg_extend<<<(s->L*s->L+255)/256,256>>>(in,s->real,s->N,s->L);
 SG_FFT(cufftExecD2Z(s->forward,s->real,s->freq));
 sg_spectrum<<<(s->L*(s->L/2+1)+255)/256,256>>>(s->freq,s->N,s->L);
 SG_FFT(cufftExecZ2D(s->backward,s->freq,s->real));
 sg_extract<<<(s->m+255)/256,256>>>(s->real,out,s->N,s->L);return map_cuda_status(cudaGetLastError());
}
static HPRQPStatus sg_Hinv(SpatialGram*s,const double*in,double*out){RM_TRY(sg_poisson(s,in,s->tmp));sg_scale<<<(s->m+255)/256,256>>>(s->tmp,s->di,s->m);return sg_poisson(s,s->tmp,out);}
static HPRQPStatus sg_solve(SpatialGram*s,const double*rhs,double*out){
 sg_divide<<<(s->m+255)/256,256>>>(rhs,s->r,s->u,s->m);RM_TRY(sg_Hinv(s,s->u,s->t));
 sg_gather<<<(s->k+255)/256,256>>>(s->t,s->idx,s->weight,s->z,s->k,-1);
 SG_DENSE(cusolverDnDpotrs(s->dense,CUBLAS_FILL_MODE_LOWER,s->k,1,s->schur,s->k,s->z,s->k,s->info));
 // Form one corrected RHS before H^-1, avoiding subtraction of large solutions.
 sg_divide<<<(s->m+255)/256,256>>>(rhs,s->r,s->u,s->m);sg_scatter<<<(s->k+255)/256,256>>>(s->u,s->idx,s->weight,s->z,s->k);
 RM_TRY(sg_Hinv(s,s->u,out));sg_divide<<<(s->m+255)/256,256>>>(out,s->r,out,s->m);return map_cuda_status(cudaGetLastError());
}
static HPRQPStatus sg_setup(RowMetric*c,const std::vector<int>&ar,const std::vector<int>&ac,const std::vector<double>&av){
 double tick=now();auto*s=new SpatialGram;c->spatial=s;s->m=c->l;s->N=std::lround(std::sqrt((double)c->l));s->L=2*(s->N+1);int N=s->N,m=s->m,n=c->n;
 if(N<3||N*N!=m)return HPRQP_STATUS_NOT_IMPLEMENTED;
 std::vector<int>degree(n,0),state(m,-1),where(n,-1);std::vector<double>diag(m),r(m,1),d(m),weight(m,0);
 for(int k=0;k<ar[m];k++)degree[ac[k]]++;
 for(int i=0;i<m;i++)for(int k=ar[i];k<ar[i+1];k++)if(degree[ac[k]]>1&&av[k]>0){if(state[i]>=0)return HPRQP_STATUS_NOT_IMPLEMENTED;state[i]=ac[k];diag[i]=av[k]/4;}
 for(int i=0;i<m;i++){if(state[i]<0||where[state[i]]>=0||!(diag[i]>0))return HPRQP_STATUS_NOT_IMPLEMENTED;where[state[i]]=i;}
 for(int j=0;j<n;j++)if(degree[j]>1&&where[j]<0)return HPRQP_STATUS_NOT_IMPLEMENTED;
 for(int i=1;i<m;i++){int parent=i%N?i-1:i-N;double a=0;for(int k=ar[i];k<ar[i+1];k++)if(ac[k]==state[parent])a=av[k];r[i]=-a*r[parent]/diag[parent];if(!(r[i]>0)||!std::isfinite(r[i]))return HPRQP_STATUS_NOT_IMPLEMENTED;}
 double mean=0;for(auto a:r)mean+=std::log(a)/m;for(int i=0;i<m;i++){r[i]/=std::exp(mean);d[i]=diag[i]/r[i];}
 double err=0,scale=0;
 for(int i=0;i<m;i++){
  int count=0;for(int k=ar[i];k<ar[i+1];k++){
   int j=where[ac[k]];scale=std::max(scale,std::fabs(av[k]));
   if(j>=0){int delta=std::abs(j-i);bool neighbor=delta==N||(delta==1&&i/N==j/N);if(j!=i&&!neighbor)return HPRQP_STATUS_NOT_IMPLEMENTED;err=std::max(err,std::fabs(av[k]-r[i]*d[j]*(j==i?4.:-1.)));count++;}
   else {if(degree[ac[k]]!=1||av[k]>=0||(i/N!=0&&i/N!=N-1&&i%N!=0&&i%N!=N-1))return HPRQP_STATUS_NOT_IMPLEMENTED;weight[i]+=(av[k]/r[i])*(av[k]/r[i]);}
  }
  int expected=1+(i/N>0)+(i/N<N-1)+(i%N>0)+(i%N<N-1);if(count!=expected)return HPRQP_STATUS_NOT_IMPLEMENTED;
 }
 s->structure_defect=err/scale;if(s->structure_defect>1e-12)return HPRQP_STATUS_NOT_IMPLEMENTED;
 std::vector<int>idx;std::vector<double>wt;for(int i=0;i<m;i++){d[i]=1/(d[i]*d[i]);if(weight[i]>0){idx.push_back(i);wt.push_back(std::sqrt(weight[i]));}}
 s->k=idx.size();if(s->k!=4*N-4)return HPRQP_STATUS_NOT_IMPLEMENTED;
 for(auto pair:{std::make_pair(&s->r,&r),std::make_pair(&s->di,&d),std::make_pair(&s->weight,&wt)}){RM_CUDA(cudaMalloc(pair.first,pair.second->size()*sizeof(double)));RM_CUDA(cudaMemcpy(*pair.first,pair.second->data(),pair.second->size()*sizeof(double),cudaMemcpyHostToDevice));}
 RM_CUDA(cudaMalloc(&s->idx,s->k*sizeof(int)));RM_CUDA(cudaMemcpy(s->idx,idx.data(),s->k*sizeof(int),cudaMemcpyHostToDevice));
 for(auto p:{&s->tmp,&s->t,&s->u}){RM_CUDA(cudaMalloc(p,m*sizeof(double)));}
 RM_CUDA(cudaMalloc(&s->real,(size_t)s->L*s->L*sizeof(double)));RM_CUDA(cudaMalloc(&s->freq,(size_t)s->L*(s->L/2+1)*sizeof(cufftDoubleComplex)));
 RM_CUDA(cudaMalloc(&s->schur,(size_t)s->k*s->k*sizeof(double)));RM_CUDA(cudaMalloc(&s->z,s->k*sizeof(double)));RM_CUDA(cudaMalloc(&s->info,sizeof(int)));
 SG_FFT(cufftPlan2d(&s->forward,s->L,s->L,CUFFT_D2Z));SG_FFT(cufftPlan2d(&s->backward,s->L,s->L,CUFFT_Z2D));SG_DENSE(cusolverDnCreate(&s->dense));
 for(int j=0;j<s->k;j++){RM_CUDA(cudaMemset(s->u,0,m*sizeof(double)));sg_unit<<<1,1>>>(s->u,s->idx,s->weight,j);RM_TRY(sg_Hinv(s,s->u,s->t));sg_gather<<<(s->k+255)/256,256>>>(s->t,s->idx,s->weight,s->schur+(size_t)j*s->k,s->k,j);}
 int lwork=0;SG_DENSE(cusolverDnDpotrf_bufferSize(s->dense,CUBLAS_FILL_MODE_LOWER,s->k,s->schur,s->k,&lwork));RM_CUDA(cudaMalloc(&s->factor_work,lwork*sizeof(double)));SG_DENSE(cusolverDnDpotrf(s->dense,CUBLAS_FILL_MODE_LOWER,s->k,s->schur,s->k,s->factor_work,lwork,s->info));int info=-1;RM_CUDA(cudaMemcpy(&info,s->info,sizeof(int),cudaMemcpyDeviceToHost));if(info!=0){hprqp_diagnostic("S698_FACTOR_FAILURE info=%d\n",info);return HPRQP_STATUS_NOT_IMPLEMENTED;}
 s->setup_s=now()-tick;hprqp_diagnostic("S698_STRUCTURE_READY N=%d rank=%d defect=%.17g setup_s=%.9g\n",N,s->k,s->structure_defect,s->setup_s);return HPRQP_STATUS_OK;
}
static HPRQPStatus sg_validate_profile(RowMetric*c,const std::vector<int>&rp,const std::vector<int>&ci,const std::vector<double>&val){
 auto*s=c->spatial;int m=c->l;std::vector<double>b(m),x(m),direct(m),known(m);double an=0;for(int i=0;i<m;i++){double a=0;for(int j=rp[i];j<rp[i+1];j++)a+=std::fabs(val[j]);an=std::max(an,a);}
 double worstback=0,worstrhs=0,worstdiff=0,worstforward=0;bool passed=true;
 for(int k=0;k<4;k++){
  for(int i=0;i<m;i++){known[i]=std::sin(.173*(i+1))+.2*std::cos(.719*(i+1));b[i]=k==0?known[i]:k==1?std::sin(3.141592653589793*(i/s->N+1)/(s->N+1))*std::sin(3.141592653589793*(i%s->N+1)/(s->N+1)):i==0?1.:0.;}
  if(k==3)for(int i=0;i<m;i++){long double v=0;for(int j=rp[i];j<rp[i+1];j++)v+=(long double)val[j]*known[ci[j]];b[i]=(double)v;}
  RM_CUDA(cudaMemcpy(c->rhs,b.data(),m*sizeof(double),cudaMemcpyHostToDevice));RM_DSS(cudssExecute(c->handle,CUDSS_PHASE_SOLVE,c->config,c->data,c->matrix,c->sout,c->rin));RM_CUDA(cudaMemcpy(direct.data(),c->sol,m*sizeof(double),cudaMemcpyDeviceToHost));
  RM_TRY(sg_solve(s,c->rhs,c->sol));RM_CUDA(cudaMemcpy(x.data(),c->sol,m*sizeof(double),cudaMemcpyDeviceToHost));int info=-1;RM_CUDA(cudaMemcpy(&info,s->info,sizeof(int),cudaMemcpyDeviceToHost));if(info)return HPRQP_STATUS_NOT_IMPLEMENTED;
  double er=0,xn=0,bn=0,dn=0,diff=0,fw=0,kn=0;for(int i=0;i<m;i++){if(!std::isfinite(x[i])||!std::isfinite(direct[i]))return HPRQP_STATUS_INVALID_ARGUMENT;long double v=0;for(int j=rp[i];j<rp[i+1];j++)v+=(long double)val[j]*x[ci[j]];er=std::max(er,(double)std::fabs(v-b[i]));xn=std::max(xn,std::fabs(x[i]));bn=std::max(bn,std::fabs(b[i]));dn=std::max(dn,std::fabs(direct[i]));diff=std::max(diff,std::fabs(x[i]-direct[i]));if(k==3){fw=std::max(fw,std::fabs(x[i]-known[i]));kn=std::max(kn,std::fabs(known[i]));}}
  worstback=std::max(worstback,er/(an*xn+bn));worstrhs=std::max(worstrhs,er/bn);worstdiff=std::max(worstdiff,diff/(1+dn));if(k==3)worstforward=fw/(1+kn);
 }
 bool small=m<=5000;passed=worstback<=1e-12&&worstrhs<=(small?1e-8:1e-5)&&worstdiff<=(small?1e-7:1e-5)&&worstforward<=(small?1e-7:1e-5);
 std::vector<double>wall,device;std::vector<int>arms;
 // Reporting is optional and cannot decide numerical activation.
 const char*out=std::getenv("HPRQP_SPATIAL_GRAM_REPORT");if(out){std::ofstream f(out);f<<std::setprecision(17)<<"{\"passed\":"<<(passed?"true":"false")<<",\"N\":"<<s->N<<",\"rank\":"<<s->k<<",\"structure_defect\":"<<s->structure_defect<<",\"setup_s\":"<<s->setup_s<<",\"max_backward\":"<<worstback<<",\"max_rhs_relative\":"<<worstrhs<<",\"max_direct_relative\":"<<worstdiff<<",\"manufactured_forward\":"<<worstforward<<",\"rhs_checks\":4,\"batch_actions\":32,\"batches\":[";
 for(size_t i=0;i<wall.size();i++){if(i)f<<",";f<<"{\"spatial\":"<<(arms[i]?"true":"false")<<",\"wall_s\":"<<wall[i]<<",\"device_s\":"<<device[i]<<"}";}f<<"]}\n";f.close();if(!f)std::fprintf(stderr,"HPRQP_SPATIAL_GRAM_REPORT warning: cannot write diagnostic report\n");}
 hprqp_diagnostic("S698_ACCURACY passed=%d backward=%.9g rhs=%.9g direct=%.9g forward=%.9g\n",passed,worstback,worstrhs,worstdiff,worstforward);return passed?HPRQP_STATUS_OK:HPRQP_STATUS_NOT_IMPLEMENTED;
}
