#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>
namespace hpr_native {
using Vec = Eigen::VectorXd;
using LVec = Eigen::Matrix<long double, Eigen::Dynamic, 1>;
using Mat = Eigen::MatrixXd;
using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using CSR = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
using Triplet = Eigen::Triplet<double>;
using Json = nlohmann::json;
namespace fs = std::filesystem;
inline double clock_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
inline void require(bool ok, const char *msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
inline double infnorm(const Vec &x) { return x.size() ? x.cwiseAbs().maxCoeff() : 0.; }
struct Model {
    int n = 0, m = 0, l = 0;
    CSR Q, A;
    Vec c, lo, hi, rl, ru, rhs;
    double constant = 0;
    std::vector<std::pair<int, int>> cones;
    void validate() const;
};
struct Candidate {
    Vec x, w, z, y, s;
};
struct ConeDerivative {
    int a, b, kind;
    Vec q;
    double tr;
}; // zero=0, identity=1, apex=2, middle=3
struct Derivative {
    Vec diag;
    std::vector<ConeDerivative> cones;
    int m;
    Vec action(const Vec &) const;
    Sparse matrix() const;
};
struct Evaluation {
    LVec F, p, s, g;
    Vec E;
    Derivative S;
    double norm = 0;
};
Model read_model(const fs::path &);
LVec read_snapshot(const fs::path &, const Model &);
void write_candidate(const fs::path &, const Candidate &, const Model &);
Candidate read_candidate(const fs::path &, const Model &);
void atomic_write(const fs::path &, const void *, size_t);
inline void atomic_text(const fs::path &p, const std::string &s) {
    atomic_write(p, s.data(), s.size());
}
Evaluation evaluate(const Model &, const LVec &);
Candidate candidate(const Model &, const LVec &, const Evaluation &);
void prepare_candidate(const Model &, Candidate &);
Json check(const Model &, const Candidate &, double tolerance = 1e-8);
struct System {
    const Model &m;
    Evaluation initial;
    explicit System(const Model &v, const LVec &u) : m(v), initial(evaluate(v, u)) {}
    Vec action(const Vec &) const;
    Sparse matrix() const;
};
struct Access {
    Json info;
    virtual ~Access() = default;
    virtual Vec solve(const Vec &, Json &) = 0;
};
std::unique_ptr<Access> sparse_access(const System &);
std::unique_ptr<Access> band_access(const System &);
std::unique_ptr<Access> spatial_access(const System &, const LVec &, const std::vector<int> &);
std::vector<int> spatial_indices(const Model &);
Json episode(const Model &, const LVec &, double seconds, Candidate &, bool &);
} // namespace hpr_native
