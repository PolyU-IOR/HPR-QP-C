#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <stdexcept>
#include <dlfcn.h>
#include <unistd.h>

namespace hpr_polish {
// Search relative to the loaded package, never for executable code in the CWD.
inline std::string worker_path(const char *override_path,
                              const std::vector<std::filesystem::path>& anchors,
                              const std::filesystem::path& source,
                              bool python_reference = false) {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;
    if (override_path) {
        candidates.emplace_back(override_path);
    } else {
        for (const auto& anchor : anchors) {
            auto dir = anchor.parent_path();
            candidates.push_back(dir / (python_reference ? "polishing/worker.py" : "hprqp_polish_worker"));
            // bin/, lib/, lib64/, and lib/<architecture>/ installations.
            for (int i=0; i<3 && !dir.empty(); ++i, dir=dir.parent_path())
                candidates.push_back(dir / (python_reference ? "share/hprqp/polishing/worker.py" :
                                                               "libexec/hprqp/hprqp_polish_worker"));
        }
        if (source.is_absolute())
            candidates.push_back(source / (python_reference ? "polishing/worker.py" :
                                                              "build-native-polish/hprqp_polish_worker"));
    }
    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (fs::is_regular_file(candidate, ec) && (python_reference || access(candidate.c_str(), X_OK)==0))
            return fs::absolute(candidate).string();
    }
    throw std::runtime_error("CPU worker unavailable; build/install hprqp_polish_worker");
}

inline std::vector<std::filesystem::path> package_anchors(const void *symbol) {
    std::vector<std::filesystem::path> paths;
    Dl_info info{};
    if (dladdr(symbol, &info) && info.dli_fname)
        paths.push_back(std::filesystem::absolute(info.dli_fname));
    std::error_code ec;
    auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) paths.push_back(exe);
    return paths;
}
}
