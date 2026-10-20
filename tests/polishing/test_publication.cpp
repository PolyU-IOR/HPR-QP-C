// Process cleanup and certificate rejection, independent of CUDA.
#undef NDEBUG
#include "polish/async_process.hpp"
#include "polish/cpu_publication.hpp"
#include <cassert>
#include <thread>
#include <functional>
using namespace hpr_polish;
static void rejected(const std::function<void()>& f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}assert(caught);}
int main(){
 char path[]="/tmp/hprqp-publication-test-XXXXXX";assert(mkdtemp(path));std::string ep=path;
 Process child;child.start({"/bin/sleep","10"},ep+"/timeout.log",.01);
 std::this_thread::sleep_for(std::chrono::milliseconds(20));assert(child.poll()==Process::TIMED_OUT);assert(!child.running());
 child.start({"/bin/sleep","10"},ep+"/cancel.log",10);child.cancel();assert(child.state==Process::CANCELLED);
 child.start({"/bin/false"},ep+"/crash.log",10);while(child.poll()==Process::RUNNING)std::this_thread::sleep_for(std::chrono::milliseconds(2));assert(child.exit_code==1);
 setenv("OPENBLAS_NUM_THREADS","7",1);
 child.start({"/bin/sh","-c","test \"$OPENBLAS_NUM_THREADS\" = 1 && test \"$OMP_NUM_THREADS\" = 1"},ep+"/threads.log",10);
 while(child.poll()==Process::RUNNING)std::this_thread::sleep_for(std::chrono::milliseconds(2));
 assert(child.exit_code==0);assert(std::string(getenv("OPENBLAS_NUM_THREADS"))=="7");
 const std::string token(32,'0');{std::ofstream f(ep+"/READY");f<<"HPRCPU01\n"<<token<<"\n";}
 assert(publication_matches(ep+"/READY",token));assert(!publication_matches(ep+"/READY",std::string(32,'1')));
 {std::ofstream f(ep+"/candidate.bin",std::ios::binary);f.write("HPRPOL01",8);uint64_t dims[]={1,0};f.write((char*)dims,16);double data[]={0,0,0};f.write((char*)data,24);}
 auto cert=[&](double error,double tolerance=1e-8){std::ofstream f(ep+"/certificate.bin",std::ios::binary);f.write("HPRCPU01",8);char id[16]={};f.write(id,16);uint64_t dims[]={1,0};f.write((char*)dims,16);auto hash=payload_digest(ep+"/candidate.bin");f.write((char*)hash.data(),32);double v[]={tolerance,error,0,0,0,0};f.write((char*)v,48);};
 cert(0);read_cpu_certificate(ep,token,1,0,1e-8);
 rejected([&](){read_cpu_certificate(ep,std::string(32,'1'),1,0,1e-8);});
 {std::fstream f(ep+"/candidate.bin",std::ios::in|std::ios::out|std::ios::binary);f.seekp(24);double x=1;f.write((char*)&x,8);}
 rejected([&](){read_cpu_certificate(ep,token,1,0,1e-8);});
 cert(1e-4);rejected([&](){read_cpu_certificate(ep,token,1,0,1e-8);});
 cert(NAN);rejected([&](){read_cpu_certificate(ep,token,1,0,1e-8);});
 cert(0,1e-6);rejected([&](){read_cpu_certificate(ep,token,1,0,1e-8);});
 std::filesystem::remove_all(ep);
}
