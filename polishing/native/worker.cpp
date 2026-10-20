#include "common.hpp"
#include "polish/cpu_publication.hpp"
#include <fstream>
#include <iostream>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
namespace hn = hpr_native;
namespace {
std::string digest(const hn::fs::path &p) {
    auto bytes = hpr_polish::payload_digest(p.string());
    const char *hex = "0123456789abcdef";
    std::string out;
    for (auto b : bytes) {
        out += hex[b >> 4];
        out += hex[b & 15];
    }
    return out;
}
void limits(double seconds, uint64_t memory) {
    using hn::require;
    require(std::isfinite(seconds) && seconds > 0 &&
                (memory == 0 || memory >= 128 * uint64_t(1 << 20)),
            "resource budget");
    if (memory) {
        rlimit r{memory, memory};
        require(!setrlimit(RLIMIT_AS, &r), "memory limit");
    }
    rlim_t cpu = std::max(1., std::ceil(seconds));
    rlimit r{cpu, cpu + 1};
    require(!setrlimit(RLIMIT_CPU, &r), "CPU limit");
    itimerval timer{};
    timer.it_value.tv_sec = time_t(seconds);
    timer.it_value.tv_usec = suseconds_t((seconds - timer.it_value.tv_sec) * 1e6);
    if (!timer.it_value.tv_sec && !timer.it_value.tv_usec)
        timer.it_value.tv_usec = 1;
    require(!setitimer(ITIMER_REAL, &timer, nullptr), "wall limit");
}
int run(int argc, char **argv) {
    using namespace hn;
    double start = clock_s();
    require(argc == 2, "usage: hprqp_polish_worker request.json");
    fs::path request = fs::absolute(argv[1]), ep = request.parent_path();
    std::ifstream f(request);
    require(bool(f), "request missing");
    Json req = Json::parse(f);
    std::string token = req.at("token");
    require(token.size() == 32 && token.find_first_not_of("0123456789abcdef") == std::string::npos,
            "token");
    int parent = req.at("parent_pid");
    require(getppid() == parent, "owner disappeared before startup");
    require(prctl(PR_SET_PDEATHSIG, SIGKILL) == 0, "parent death signal");
    require(getppid() == parent, "owner disappeared during startup");
    double seconds = req.at("seconds");
    limits(seconds, req.at("memory_bytes").get<uint64_t>());
    for (const char *name :
         {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"})
        setenv(name, "1", 1);
    require(!fs::exists(ep / "READY"), "episode already published");
    fs::path model_path = req.at("model").get<std::string>(), snapshot = ep / "snapshot.bin";
    std::string model_hash = digest(model_path), snapshot_hash = digest(snapshot);
    for (auto pair : {std::make_pair("model_sha256", model_hash),
                      std::make_pair("snapshot_sha256", snapshot_hash)})
        if (req.contains(pair.first))
            require(req.at(pair.first) == pair.second, "request hash mismatch");
    Model m = read_model(model_path);
    require(req.at("dimensions") == Json::array({m.n, m.m}), "request dimensions");
    LVec u = read_snapshot(snapshot, m);
    Json options = req.value("config", Json::object());
    for (auto it = options.begin(); it != options.end(); ++it)
        require(it.key() == "backend" || it.key() == "dense_cone_limit" ||
                    it.key() == "direct_dimension" || it.key() == "assembly_nnz",
                "unsupported configuration");
    std::string backend = options.value("backend", std::string("generic"));
    require(backend == "auto" || backend == "generic" || backend == "banded", "backend");
    for (const char *key : {"dense_cone_limit", "direct_dimension", "assembly_nnz"})
        if (options.contains(key))
            require(options.at(key).is_number_integer() && options.at(key).get<int64_t>() > 0,
                    "configuration resource cap");
    double tolerance = req.at("tolerance");
    require(std::isfinite(tolerance) && tolerance > 0, "tolerance");
    Candidate c;
    bool available = false;
    Json result = episode(m, u, std::max(1e-9, seconds - (clock_s() - start)), c, available);
    result.update({{"token", token},
                   {"model_sha256", model_hash},
                   {"snapshot_sha256", snapshot_hash},
                   {"snapshot_iteration", req.at("iteration")}});
    if (available && clock_s() - start < seconds) {
        double checking = clock_s();
        Vec oldy = c.y, oldz = c.z;
        prepare_candidate(m, c);
        result["dual_cleanup_maxabs"] =
            std::max(infnorm(Vec(c.y - oldy)), infnorm(Vec(c.z - oldz)));
        write_candidate(ep / "candidate.bin", c, m);
        Candidate verified = read_candidate(ep / "candidate.bin", m);
        Json certificate = check(m, verified, tolerance);
        result["independent_cpu_check"] = certificate;
        result["cpu_check_s"] = clock_s() - checking;
        result["candidate_sha256"] = digest(ep / "candidate.bin");
        available = certificate.at("passed");
        result["independent_cpu_verified"] = available;
        result["status"] = available ? "CPU_CERTIFIED" : "CPU_CHECK_REJECT";
        if (available) {
            std::array<unsigned char, 120> packet{};
            std::memcpy(packet.data(), "HPRCPU01", 8);
            for (int i = 0; i < 16; ++i)
                packet[8 + i] = std::stoul(token.substr(2 * i, 2), nullptr, 16);
            uint64_t dims[2] = {uint64_t(m.n), uint64_t(m.m)};
            std::memcpy(packet.data() + 24, dims, 16);
            auto hash = hpr_polish::payload_digest((ep / "candidate.bin").string());
            std::memcpy(packet.data() + 40, hash.data(), 32);
            double vals[6] = {tolerance,
                              certificate.at("primal_feasibility"),
                              certificate.at("dual_feasibility"),
                              certificate.at("relative_gap"),
                              certificate.at("primal_objective"),
                              certificate.at("dual_objective")};
            std::memcpy(packet.data() + 72, vals, 48);
            atomic_write(ep / "certificate.bin", packet.data(), packet.size());
        }
    } else
        available = false;
    result["process_s"] = clock_s() - start;
    result["process_s_scope"] =
        "startup through CPU certificate, before result and READY publication";
    rusage usage{};
    require(!getrusage(RUSAGE_SELF, &usage), "resource usage");
    result["peak_rss_kib"] = usage.ru_maxrss;
    atomic_text(ep / "result.json", result.dump(2) + "\n");
    if (available && clock_s() - start < seconds)
        atomic_text(ep / "READY", "HPRCPU01\n" + token + "\n");
    return 0;
}
} // namespace
int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "native polishing: " << e.what() << '\n';
        return 2;
    }
}
