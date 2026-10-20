#include "polish/worker_path.hpp"
#include <fstream>
#include <iostream>
#include <unistd.h>

int main() {
    namespace fs=std::filesystem;
    char pattern[]="/tmp/hprqp-worker-path-XXXXXX";
    auto dir=mkdtemp(pattern);
    if (!dir) return 1;
    fs::path root=dir, source=root/"checkout", prefix=root/"moved";
    auto put=[](const fs::path& p) {fs::create_directories(p.parent_path());std::ofstream(p)<<"# test\n";};
    auto src=source/"polishing/worker.py", installed=prefix/"share/hprqp/polishing/worker.py";
    put(src);put(installed);
    int failures=0;
    auto require=[&](bool value) {if (!value) ++failures;};
    using hpr_polish::worker_path;
    require(worker_path(nullptr,{prefix/"bin/solve_qp"},source)==installed.string());
    require(worker_path(nullptr,{prefix/"lib/libhprqp.so"},source)==installed.string());
    require(worker_path(nullptr,{prefix/"lib/x86_64-linux-gnu/libhprqp.so"},source)==installed.string());
    require(worker_path(src.c_str(),{prefix/"bin/solve_qp"},source)==src.string());
    fs::remove(installed);
    require(worker_path(nullptr,{root/"build/solve_qp"},source)==src.string());
    try {worker_path("/not/a/worker",{},source);++failures;} catch(const std::runtime_error&) {}
    try {worker_path(nullptr,{},fs::path("."));++failures;} catch(const std::runtime_error&) {}
    fs::remove_all(root);
    if (failures) std::cerr<<failures<<" worker lookup failures\n";
    return failures?1:0;
}
