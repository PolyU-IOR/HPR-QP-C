#pragma once
// Linux/POSIX process ownership only. No GPU pointers and no numerical policy.
#include <spawn.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
extern char **environ;

namespace hpr_polish {
class Process {
    pid_t pid_ = -1;
    std::chrono::steady_clock::time_point started_;
    double budget_ = 0.;
public:
    enum State { IDLE, RUNNING, EXITED, TIMED_OUT, CANCELLED };
    State state = IDLE;
    int exit_code = -1;
    Process() = default;
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;
    ~Process() { cancel(); }
    bool running() const { return pid_ > 0; }
    double age() const { return state == IDLE ? 0. : std::chrono::duration<double>(std::chrono::steady_clock::now()-started_).count(); }
    void start(const std::vector<std::string>& args, const std::string& log, double seconds) {
        if(running() || args.empty() || !(seconds > 0.)) throw std::runtime_error("invalid worker start");
        posix_spawn_file_actions_t actions;
        posix_spawnattr_t attrs;
        int rc=posix_spawn_file_actions_init(&actions); if(rc)throw std::runtime_error("spawn actions");
        rc=posix_spawnattr_init(&attrs);
        if(rc){posix_spawn_file_actions_destroy(&actions);throw std::runtime_error("spawn attributes");}
        auto cleanup=[&](){posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attrs);};
        rc=posix_spawn_file_actions_addopen(&actions,STDOUT_FILENO,log.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
        if(!rc)rc=posix_spawn_file_actions_adddup2(&actions,STDOUT_FILENO,STDERR_FILENO);
        if(!rc)rc=posix_spawnattr_setflags(&attrs,POSIX_SPAWN_SETPGROUP);
        if(!rc)rc=posix_spawnattr_setpgroup(&attrs,0); // child owns its process group
        std::vector<char*> argv;for(const auto& a:args)argv.push_back(const_cast<char*>(a.c_str()));argv.push_back(nullptr);
        started_=std::chrono::steady_clock::now();budget_=seconds;
        if(!rc)rc=posix_spawnp(&pid_,argv[0],&actions,&attrs,argv.data(),environ);
        cleanup();
        if(rc){pid_=-1;state=IDLE;throw std::runtime_error("posix_spawn error "+std::to_string(rc));}
        state=RUNNING;exit_code=-1;
    }
    State poll() {
        if(!running())return state;
        int status=0;pid_t r=waitpid(pid_,&status,WNOHANG);
        if(r==pid_){
            // Child still belongs to our unrecycled process group at wait return.
            pid_=-1;state=EXITED;
            exit_code=WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
        } else if(r<0 && errno!=EINTR) {
            if(errno==ECHILD){pid_=-1;state=EXITED;exit_code=-1;}
            else throw std::runtime_error("worker waitpid");
        }
        // A completed candidate need not be fresh: delayed owner polling is
        // distinct from a still-running worker exceeding its resource budget.
        if(r==0&&age()>=budget_)stop(TIMED_OUT);
        return state;
    }
    void cancel(){if(running())stop(CANCELLED);}
private:
    void stop(State why){
        // SIGKILL is confined to this dedicated worker group; no grace wait on HPR.
        if(pid_<=0)return;
        kill(-pid_,SIGKILL);
        int status=0;while(waitpid(pid_,&status,0)<0 && errno==EINTR){}
        pid_=-1;state=why;exit_code=137;
    }
};

inline bool publication_matches(const std::string& path,const std::string& token){
    std::ifstream f(path,std::ios::binary);std::string a,b,extra;
    if(!std::getline(f,a)||!std::getline(f,b))return false;
    return a=="HPRCPU01"&&b==token&&f.peek()==EOF;
}
} // namespace hpr_polish
