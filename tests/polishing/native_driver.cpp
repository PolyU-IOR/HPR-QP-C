/* Small native-model integration driver. No benchmark policy or external hook. */
#include "hprqp.h"
#include "../../src/shared/gram_policy.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <cmath>
template<class T> void read(std::ifstream& f,T*p,size_t n){if(n&&!f.read((char*)p,n*sizeof(T)))throw std::runtime_error("truncated fixture");}
std::vector<double> vec(std::ifstream&f,size_t n){std::vector<double> v(n);read(f,v.data(),n);return v;}
HPRQPMatrixCSR matrix(std::ifstream&f){uint64_t d[3];read(f,d,3);HPRQPMatrixCSR a={};if(matrix_csr_alloc(&a,d[0],d[1],d[2]))throw std::runtime_error("alloc");read(f,a.row_ptr,d[0]+1);read(f,a.col_ind,d[2]);read(f,a.values,d[2]);return a;}
int main(int argc,char**argv){try{
 bool inspect=argc==3 && std::string(argv[2])=="--inspect-gram";
 if(argc<6 && !inspect)return 2;
 std::ifstream f(argv[1],std::ios::binary);char magic[8];read(f,magic,8);if(memcmp(magic,"HPRNAT01",8))throw std::runtime_error("magic");
 auto Q=matrix(f),A=matrix(f);size_t n=Q.nrow,m=A.nrow;
 auto c=vec(f,n),l=vec(f,n),u=vec(f,n),al=vec(f,m),au=vec(f,m),rhs=vec(f,m);double constant;read(f,&constant,1);uint64_t nc;read(f,&nc,1);std::vector<size_t> offsets(nc?nc+1:0);read(f,offsets.data(),offsets.size());
 HPRQPModel model={};HPRQPSOCSpec cones={};cones.num_row_cones=nc;cones.row_offsets=offsets.data();cones.rhs=rhs.data();
 if(build_from_QAbc_soc(&Q,&A,c.data(),al.data(),au.data(),l.data(),u.data(),constant,&cones,&model))throw std::runtime_error("build");
 if(inspect){std::cout<<hprqp_gram_auto_eligible(&model)<<"\n";model_free(&model);matrix_csr_free(&Q);matrix_csr_free(&A);return 0;}
 HPRQPParams p;params_set_defaults(&p);p.gram_mode=(HPRQPGramMode)std::stoi(argv[3]);p.polishing=std::stoi(argv[4]);p.polish_directory=argv[2];p.polish_keep_files=1;
 p.polish_python=std::string(argv[5])=="-"?nullptr:argv[5];p.stoptol=1e-8;p.check_iter=100;p.time_limit=60;p.max_iter=1000000;p.verbose=0;p.print_frequency=0;p.spmv_mode_a=p.spmv_mode_q=HPRQP_SPMV_MODE_CUSPARSE;
 if(argc>6&&std::string(argv[6])!="-")p.polish_worker=argv[6];if(argc>7){p.max_iter=std::stoull(argv[7]);p.stoptol=1e-14;}
 // Optional complete-run contract: retain native accuracy and charge the
 // requested time allowance. The fixed-update parity interface above is unchanged.
 if(argc>8){p.time_limit=std::stod(argv[8]);p.stoptol=1e-8;}
 HPRQPResults r={};auto status=solve(&model,&p,&r);
 std::cout<<std::setprecision(17)<<"{\"status\":"<<status<<",\"error\":"<<(std::isfinite(r.residuals)?r.residuals:-1.)<<",\"objective\":"<<r.primal_obj<<",\"iterations\":"<<r.iter<<",\"gram\":"<<r.gram_active<<",\"attempts\":"<<r.polish_attempts<<",\"accepted\":"<<r.polish_accepted<<",\"total_time\":"<<r.total_time<<"}\n";
 if(r.x){std::ofstream o(std::string(argv[2])+"/solution.bin",std::ios::binary);o.write("HPRSOLS1",8);uint64_t sz[]={n,model.A.nrow,n,n};o.write((char*)sz,32);for(auto pair:{std::pair<double*,size_t>{r.x,n},{r.y,model.A.nrow},{r.z,n},{r.w,n}})if(pair.second)o.write((char*)pair.first,pair.second*8);}
 results_free(&r);model_free(&model);matrix_csr_free(&Q);matrix_csr_free(&A);
 return status==HPRQP_STATUS_OPTIMAL||status==HPRQP_STATUS_MAX_ITER?0:3;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}
