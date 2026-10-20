#include "polish.h"
#include "polish/async_process.hpp"
#include "polish/cpu_publication.hpp"
#include "polish/progress.hpp"
#include "polish/worker_path.hpp"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <iomanip>

#ifndef HPRQP_SOURCE_DIR
#define HPRQP_SOURCE_DIR "."
#endif
namespace {
using Clock=std::chrono::steady_clock;
using namespace hpr_polish;
std::string quote(const std::string& s) {
    std::string out="\"";
    for(unsigned char c:s) { if(c<32)throw std::runtime_error("control character in path"); if(c=='\\'||c=='\"')out+='\\';out+=char(c); }
    return out+'\"';
}
std::string token() {
    std::ifstream f("/dev/urandom",std::ios::binary); unsigned char b[16];
    if(!f.read(reinterpret_cast<char*>(b),16))throw std::runtime_error("entropy unavailable");
    const char *hex="0123456789abcdef";std::string s;
    for(auto a:b) {s+=hex[a>>4];s+=hex[a&15];}return s;
}
template<class T> void put(std::ofstream& f,const T *v,size_t n) {
    if(n)f.write(reinterpret_cast<const char*>(v),sizeof(T)*n);
}
void export_model(const HPRQPModel& m,const std::string& path) {
    static_assert(sizeof(size_t)==8,"polishing transport requires a 64-bit host");
    std::ofstream f(path,std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);f.write("HPRNAT01",8);
    for(auto a:{&m.Q,&m.A}) {uint64_t d[]={a->nrow,a->ncol,a->nnz};put(f,d,3);put(f,a->row_ptr,a->nrow+1);put(f,a->col_ind,a->nnz);put(f,a->values,a->nnz);}
    size_t n=m.Q.ncol,r=m.A.nrow;
    put(f,m.c,n);put(f,m.l,n);put(f,m.u,n);put(f,m.AL,r);put(f,m.AU,r);
    if(m.soc_rhs)put(f,m.soc_rhs,r);else{std::vector<double> z(r);put(f,z.data(),r);}
    put(f,&m.obj_constant,1);uint64_t cones=m.num_row_cones;put(f,&cones,1);
    if(cones)put(f,m.row_cone_offsets,cones+1);f.close();
    if(chmod(path.c_str(),0400))throw std::runtime_error("model file permissions");
}
struct Candidate {std::vector<double>x,w,z,y,s;};
Candidate read_candidate(const std::string& path,size_t n,size_t m) {
    std::ifstream f(path,std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);
    char magic[8];uint64_t dims[2];f.read(magic,8);f.read(reinterpret_cast<char*>(dims),16);
    if(memcmp(magic,"HPRPOL01",8)||dims[0]!=n||dims[1]!=m)throw std::runtime_error("candidate dimensions");
    Candidate c{std::vector<double>(n),std::vector<double>(n),std::vector<double>(n),std::vector<double>(m),std::vector<double>(m)};
    for(auto v:{&c.x,&c.w,&c.z,&c.y,&c.s}) {
        if(v->size())f.read(reinterpret_cast<char*>(v->data()),8*v->size());
        for(double a:*v)if(!std::isfinite(a))throw std::runtime_error("candidate not finite");
    }
    return c;
}
class Owner {
public:
    HPRQPParams params;const HPRQPModel *model;
    MilestoneProgress progress;Process child;
    std::string run,episode,id,python,worker;
    bool pending=false,accepted=false,disabled=false;
    size_t attempts=0;double child_spent=0;
    Owner(const HPRQPModel *m,const HPRQPParams& p):params(p),model(m) {
        python=p.polish_python?p.polish_python:"python3";
        worker=worker_path(p.polish_worker,
                          package_anchors(reinterpret_cast<const void*>(&hprqp_polish_create)),
                          HPRQP_SOURCE_DIR);
        if(p.polish_episode_seconds<0||p.polish_total_seconds<0||!std::isfinite(p.polish_episode_seconds)||!std::isfinite(p.polish_total_seconds))throw std::runtime_error("invalid worker budgets");
    }
    ~Owner() {
        child.cancel();
        if(!run.empty()&&!params.polish_keep_files){std::error_code ec;std::filesystem::remove_all(run,ec);}
    }
    void event(const char* what,size_t iter) {
        if(run.empty())return;
        std::ofstream f(run+"/events.jsonl",std::ios::app);
        f<<"{\"event\":"<<quote(what)<<",\"iteration\":"<<iter<<",\"attempt\":"<<attempts<<"}\n";
        if(params.verbose)std::fprintf(stderr,"POLISH %s iteration=%zu attempt=%zu\n",what,iter,attempts);
    }
    int poll(HPRQPSolverBackendContext *ctx,const HPRQPSolverBackendOps *ops,const HPRQPResiduals *base,
             size_t iter,double seconds,double *x,double*w,double*z,double*y,HPRQPResiduals*out) {
        if(disabled)return 0;
        auto tick=Clock::now();
        auto remaining=[&](){return params.time_limit-seconds-std::chrono::duration<double>(Clock::now()-tick).count();};
        if(remaining()<=0){child.cancel();pending=false;return 0;}
        if(pending) {
            auto state=child.poll();if(state==Process::RUNNING)return 0;
            pending=false;child_spent+=child.age();
            if(state!=Process::EXITED||child.exit_code||!publication_matches(episode+"/READY",id)){event("NO_VALID_PUBLICATION",iter);return 0;}
            const size_t n=model->Q.ncol,m=model->A.nrow;
            auto cert=read_cpu_certificate(episode,id,n,m,params.stoptol);
            auto c=read_candidate(episode+"/candidate.bin",n,m);
            if(remaining()<=0)return 0;
            // These are HOST result buffers. There is deliberately no GPU upload.
            std::copy(c.x.begin(),c.x.end(),x);std::copy(c.w.begin(),c.w.end(),w);
            std::copy(c.z.begin(),c.z.end(),z);std::copy(c.y.begin(),c.y.end(),y);
            *out={};out->is_updated=1;out->err_Rp_org_bar=cert.primal_feasibility;out->err_Rd_org_bar=cert.dual_feasibility;
            out->rel_gap_bar=cert.gap;out->primal_obj_bar=cert.primal_objective;out->dual_obj_bar=cert.dual_objective;
            out->KKTx_and_gap_org_bar=std::max({cert.primal_feasibility,cert.dual_feasibility,cert.gap});
            accepted=true;event("CPU_CERTIFICATE_ACCEPT",iter);return 1;
        }
        if((params.polish_max_attempts&&attempts>=params.polish_max_attempts)||
           (params.polish_total_seconds>0&&child_spent>=params.polish_total_seconds)||
           !progress.eligible(iter,params.check_iter,base->err_Rp_org_bar,base->err_Rd_org_bar,base->KKTx_and_gap_org_bar,params.stoptol))return 0;
        if(run.empty()) {
            std::string parent=params.polish_directory?params.polish_directory:"/tmp";
            std::string pattern=(std::filesystem::absolute(parent)/"hprqp-polish-XXXXXX").string();
            std::vector<char> buf(pattern.begin(),pattern.end());buf.push_back(0);
            char *d=mkdtemp(buf.data());if(!d)throw std::runtime_error("private worker directory");run=d;
            export_model(*model,run+"/model.bin");
        }
        progress.launched(base->err_Rp_org_bar,base->err_Rd_org_bar,base->KKTx_and_gap_org_bar,params.stoptol);
        ++attempts;id=token();episode=run+"/episode_"+std::to_string(attempts);
        if(!std::filesystem::create_directory(episode))throw std::runtime_error("episode collision");
        const size_t n=model->Q.ncol,m=model->A.nrow;
        std::vector<double> sx(n),sw(n),sz(n),sy(m);
        if(ops->copy_output_iterates(ctx,sx.data(),sw.data(),sz.data(),sy.data())!=HPRQP_STATUS_OK)throw std::runtime_error("snapshot export failed");
        std::ofstream f(episode+"/snapshot.bin",std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);
        f.write("HPRSOLS1",8);uint64_t dims[]={n,m,n,n};put(f,dims,4);
        put(f,sx.data(),n);put(f,sy.data(),m);put(f,sz.data(),n);put(f,sw.data(),n);f.close();
        if(chmod((episode+"/snapshot.bin").c_str(),0400))throw std::runtime_error("snapshot permissions");
        double budget=remaining();
        if(params.polish_episode_seconds>0)budget=std::min(budget,params.polish_episode_seconds);
        if(params.polish_total_seconds>0)budget=std::min(budget,params.polish_total_seconds-child_spent);
        if(!(budget>0)||!std::isfinite(budget))return 0;
        std::ofstream req(episode+"/request.json");req.exceptions(std::ios::badbit|std::ios::failbit);
        req<<std::setprecision(17)<<"{\"token\":"<<quote(id)<<",\"parent_pid\":"<<getpid()<<",\"model\":"<<quote(run+"/model.bin")
           <<",\"dimensions\":["<<n<<","<<m<<"],\"iteration\":"<<iter<<",\"tolerance\":"<<params.stoptol
           <<",\"seconds\":"<<budget<<",\"memory_bytes\":"<<params.polish_memory_bytes<<",\"config\":{\"backend\":\"auto\"}}\n";req.close();
        child.start({python,worker,episode+"/request.json"},episode+"/worker.log",budget);
        pending=true;event("LAUNCHED",iter);return 0;
    }
};
}
void *hprqp_polish_create(const HPRQPModel *m,const HPRQPParams *p) {
    if(!p->polishing||m->num_var_cones)return nullptr;
    try{return new Owner(m,*p);}catch(const std::exception& e){if(p->verbose)std::fprintf(stderr,"POLISH_DISABLED %s\n",e.what());return nullptr;}
}
int hprqp_polish_poll(void *handle,HPRQPSolverBackendContext *ctx,const HPRQPSolverBackendOps *ops,
    const HPRQPResiduals *base,size_t iter,double seconds,double*x,double*w,double*z,double*y,HPRQPResiduals*out) {
    auto o=static_cast<Owner*>(handle);if(!o)return 0;
    try{return o->poll(ctx,ops,base,iter,seconds,x,w,z,y,out);}
    catch(const std::exception& e){o->child.cancel();o->pending=false;o->disabled=true;o->event("DISABLED",iter);
        if(o->params.verbose)std::fprintf(stderr,"POLISH_DISABLED %s\n",e.what());return 0;}
}
void hprqp_polish_destroy(void *handle,HPRQPResults *r) {
    auto o=static_cast<Owner*>(handle);if(!o)return;
    if(r){r->polish_attempts=o->attempts;r->polish_accepted=o->accepted;}
    delete o;
}
