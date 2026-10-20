#undef NDEBUG
#include "packed_spmv.cuh"
#include "shared/sparse.h"
#include <cmath>
#include <cassert>
#include <set>
__global__ void action(HPRQPGPUELL e,const double*x,double*y){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<e.stride)y[i]=packed_spmv_dot(e,x,i);}
static double verify(HPRQPMatrixCSR*a,bool expect){
 unsigned width=packed_spmv_width(a);assert(bool(width)==expect);if(!width)return 0;
 HPRQPGPUELL e{};assert(packed_spmv_pack(&e,a,width));std::vector<int32_t>cols(a->nrow*width);std::vector<double>val(a->nrow*width);
 assert(cudaMemcpy(cols.data(),e.columns,cols.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);assert(cudaMemcpy(val.data(),e.values,val.size()*8,cudaMemcpyDeviceToHost)==cudaSuccess);
 for(size_t i=0;i<a->nrow;i++)for(size_t k=0;k<width;k++){size_t j=a->row_ptr[i]+k,p=k*a->nrow+i;if(j<a->row_ptr[i+1]){assert(cols[p]==a->col_ind[j]&&val[p]==a->values[j]);}else assert(cols[p]==-1&&val[p]==0);}
 std::vector<double>x(a->ncol),y(a->nrow);for(size_t j=0;j<x.size();j++)x[j]=sin(.19*j)+cos(.011*j);double*dx=nullptr,*dy=nullptr;
 assert(cudaMalloc(&dx,x.size()*8)==cudaSuccess);assert(cudaMalloc(&dy,y.size()*8)==cudaSuccess);assert(cudaMemcpy(dx,x.data(),x.size()*8,cudaMemcpyHostToDevice)==cudaSuccess);
 action<<<(a->nrow+127)/128,128>>>(e,dx,dy);assert(cudaMemcpy(y.data(),dy,y.size()*8,cudaMemcpyDeviceToHost)==cudaSuccess);double err=0;
 for(size_t i=0;i<a->nrow;i++){long double ref=0;double same=0;for(size_t j=a->row_ptr[i];j<a->row_ptr[i+1];j++){ref+=(long double)a->values[j]*x[a->col_ind[j]];same=std::fma(a->values[j],x[a->col_ind[j]],same);}assert(y[i]==same);err=std::max(err,(double)(fabsl(ref-y[i])/(1+fabsl(ref))));}
 assert(err<1e-12);cudaFree(dx);cudaFree(dy);packed_spmv_free(&e);return err;
}
int main(){double worst=0;int count=0;
 for(int fixture=0;fixture<8;fixture++){
  size_t m=fixture==0?1:fixture==1?127:fixture==2?257:128,n=fixture==2?311:256;
  std::vector<size_t>rp(1,0),ci;std::vector<double>vv;
  for(size_t i=0;i<m;i++){int k=fixture==0?2:fixture==1?3:fixture==2?7:fixture==3?31:fixture==4?33:fixture==5?(i==0?32:1):fixture==6?0:(i%5==0?0:5);std::set<size_t>js;for(int t=0;t<k;t++)js.insert((i*11+t*7)%n);for(auto j:js){ci.push_back(j);vv.push_back(sin(1.7*i+.3*j));}rp.push_back(ci.size());}
  HPRQPMatrixCSR a{};a.nrow=m;a.ncol=n;a.nnz=ci.size();size_t dummy_col=0;double dummy_value=0; a.row_ptr=rp.data();a.col_ind=ci.empty()?&dummy_col:ci.data();a.values=vv.empty()?&dummy_value:vv.data();bool ok=fixture<4||fixture==7;worst=std::max(worst,verify(&a,ok));count++;
  printf("FIXTURE %d passed primary\n",fixture);fflush(stdout);HPRQPMatrixCSR at{};assert(matrix_csr_transpose(&at,&a)==HPRQP_STATUS_OK);bool okt=bool(packed_spmv_width(&at));worst=std::max(worst,verify(&at,okt));count++;matrix_csr_free(&at);
 }
 printf("PACKED_SPMV_TEST cases=%d max_relative_error=%.17g\n",count,worst);return 0;
}
