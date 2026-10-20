#include "linalg.hpp"
#include <numeric>
#include <queue>
#include <slu_ddefs.h>
extern "C" {
void dgbtrf_(int *, int *, int *, int *, double *, int *, int *, int *);
void dgbtrs_(char *, int *, int *, int *, int *, double *, int *, int *, double *, int *, int *);
}
namespace hpr_native {
namespace {
class SparseAccess final : public Access {
    const System &s;
    Sparse J;
    SVD decomp;
    int rank = 0;
    SuperMatrix L{}, U{};
    std::vector<int> pr, pc;
    bool factored = false;

  public:
    explicit SparseAccess(const System &sys) : s(sys), J(sys.matrix()) {
        double tick = clock_s();
        int n = J.rows();
        if (n <= 512) {
            decomp = svd(Mat(J));
            double cutoff =
                n * std::numeric_limits<double>::epsilon() * (decomp.s.size() ? decomp.s[0] : 0);
            while (rank < decomp.s.size() && decomp.s[rank] > cutoff)
                ++rank;
            info = {{"access", "dense_svd"}, {"rank", rank}};
        } else {
            SuperMatrix a, b;
            Vec dummy = Vec::Zero(n);
            pr.resize(n);
            pc.resize(n);
            dCreate_CompCol_Matrix(&a, n, n, J.nonZeros(), J.valuePtr(), J.innerIndexPtr(),
                                   J.outerIndexPtr(), SLU_NC, SLU_D, SLU_GE);
            dCreate_Dense_Matrix(&b, n, 1, dummy.data(), n, SLU_DN, SLU_D, SLU_GE);
            superlu_options_t opts;
            set_default_options(&opts);
            opts.ColPerm = COLAMD;
            SuperLUStat_t stat;
            StatInit(&stat);
            int code = 0;
            dgssv(&opts, &a, pc.data(), pr.data(), &L, &U, &b, &stat, &code);
            StatFree(&stat);
            Destroy_SuperMatrix_Store(&a);
            Destroy_SuperMatrix_Store(&b);
            if (code) {
                if (L.Store)
                    Destroy_SuperNode_Matrix(&L);
                if (U.Store)
                    Destroy_CompCol_Matrix(&U);
                L.Store = U.Store = nullptr;
                throw std::runtime_error("SPARSE_FACTOR_REJECT:" + std::to_string(code));
            }
            factored = true;
            info = {{"access", "sparse_lu"}, {"rank", nullptr}};
        }
        info.update({{"dimension", n}, {"factorizations", 1}, {"setup_s", clock_s() - tick}});
    }
    ~SparseAccess() {
        if (factored) {
            Destroy_SuperNode_Matrix(&L);
            Destroy_CompCol_Matrix(&U);
        }
    }
    Vec solve(const Vec &b, Json &rec) override {
        double t = clock_s();
        Vec d;
        if (!factored)
            d = decomp.Vt.topRows(rank).transpose() *
                ((decomp.U.leftCols(rank).transpose() * b).cwiseQuotient(decomp.s.head(rank)));
        else {
            d = b;
            SuperMatrix rhs;
            dCreate_Dense_Matrix(&rhs, b.size(), 1, d.data(), b.size(), SLU_DN, SLU_D, SLU_GE);
            SuperLUStat_t stat;
            StatInit(&stat);
            int code = 0;
            dgstrs(NOTRANS, &L, &U, pc.data(), pr.data(), &rhs, &stat, &code);
            StatFree(&stat);
            Destroy_SuperMatrix_Store(&rhs);
            require(!code, "SPARSE_SOLVE_REJECT");
        }
        rec = {{"defect", (s.action(d) - b).norm() / std::max(b.norm(), 1e-300)},
               {"solve_s", clock_s() - t}};
        return d;
    }
};
class BandAccess final : public Access {
    const System &s;
    std::vector<int> gauges, perm, piv;
    int n, lo = 0, hi = 0, ld = 0;
    Mat ab;

  public:
    explicit BandAccess(const System &sys) : s(sys) {
        double tick = clock_s();
        Sparse J = sys.matrix();
        n = J.rows();
        const Model &m = s.m;
        Vec ident = Vec::Zero(m.m);
        for (int i = 0; i < m.l; ++i)
            ident[i] = s.initial.S.diag[i] == 1;
        for (auto &d : s.initial.S.cones)
            if (d.kind == 1)
                ident.segment(d.a, d.b - d.a).setOnes();
        Vec q = m.Q.diagonal(), forbidden = Vec::Zero(m.n);
        bool diag = true;
        for (int i = 0; i < m.n; ++i)
            for (CSR::InnerIterator it(m.Q, i); it; ++it)
                if (it.row() != it.col() && it.value())
                    diag = false;
        for (int i = 0; i < m.m; ++i)
            if (!ident[i])
                for (CSR::InnerIterator it(m.A, i); it; ++it)
                    forbidden[it.col()] += std::abs(it.value());
        if (diag)
            for (int i = 0; i < m.n; ++i)
                if (s.initial.E[i] && q[i] == 0 && m.c[i] == 0 && forbidden[i] == 0)
                    gauges.push_back(i);
        double coldef = 0, rowdef = 0;
        if (!gauges.empty()) {
            CSR jr = J;
            Sparse ac = m.A;
            for (int g : gauges) {
                for (Sparse::InnerIterator it(J, g); it; ++it)
                    coldef = std::max(coldef, std::abs(it.value()));
                Eigen::SparseVector<double> row = jr.row(g).transpose();
                for (Sparse::InnerIterator ai(ac, g); ai; ++ai)
                    if (ai.value())
                        row += ai.value() *
                               Eigen::SparseVector<double>(jr.row(m.n + ai.row()).transpose());
                for (Eigen::SparseVector<double>::InnerIterator it(row); it; ++it)
                    rowdef = std::max(rowdef, std::abs(it.value()));
            }
            require(std::max(coldef, rowdef) <= 1e-11, "BAND_GAUGE_IDENTITY_REJECT");
            std::vector<bool> is(n, false);
            for (int g : gauges)
                is[g] = true;
            std::vector<Triplet> t;
            for (int j = 0; j < J.outerSize(); ++j)
                for (Sparse::InnerIterator it(J, j); it; ++it)
                    if (!is[it.row()] && it.value())
                        t.emplace_back(it.row(), it.col(), it.value());
            for (int g : gauges)
                t.emplace_back(g, g, 1);
            J.setZero();
            J.setFromTriplets(t.begin(), t.end());
        }
        std::vector<std::vector<int>> adj(n);
        for (int j = 0; j < J.outerSize(); ++j)
            for (Sparse::InnerIterator it(J, j); it; ++it) {
                adj[it.row()].push_back(it.col());
                adj[it.col()].push_back(it.row());
            }
        for (auto &v : adj) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return adj[a].size() < adj[b].size(); });
        std::vector<bool> seen(n, false);
        std::queue<int> queue;
        for (int root : order)
            if (!seen[root]) {
                seen[root] = true;
                queue.push(root);
                while (!queue.empty()) {
                    int i = queue.front();
                    queue.pop();
                    perm.push_back(i);
                    std::vector<int> next;
                    for (int j : adj[i])
                        if (!seen[j]) {
                            seen[j] = true;
                            next.push_back(j);
                        }
                    std::stable_sort(next.begin(), next.end(),
                                     [&](int a, int b) { return adj[a].size() < adj[b].size(); });
                    for (int j : next)
                        queue.push(j);
                }
            }
        std::reverse(perm.begin(), perm.end());
        std::vector<int> inv(n);
        for (int i = 0; i < n; ++i)
            inv[perm[i]] = i;
        for (int j = 0; j < J.outerSize(); ++j)
            for (Sparse::InnerIterator it(J, j); it; ++it) {
                int d = inv[it.row()] - inv[it.col()];
                lo = std::max(lo, d);
                hi = std::max(hi, -d);
            }
        ld = 2 * lo + hi + 1;
        require(std::max(lo, hi) <= 128 && uint64_t(ld) * n * 8 <= (uint64_t(1) << 30),
                "BAND_ACCESS_RESOURCE");
        ab = Mat::Zero(ld, n);
        for (int j = 0; j < J.outerSize(); ++j)
            for (Sparse::InnerIterator it(J, j); it; ++it)
                ab(lo + hi + inv[it.row()] - inv[it.col()], inv[it.col()]) = it.value();
        piv.resize(n);
        int code = 0;
        dgbtrf_(&n, &n, &lo, &hi, ab.data(), &ld, piv.data(), &code);
        if (code)
            throw std::runtime_error("BAND_FACTOR_REJECT:" + std::to_string(code));
        info = {{"access", "common_kkt_band"},
                {"factorizations", 1},
                {"lower_bandwidth", lo},
                {"upper_bandwidth", hi},
                {"gauges", gauges.size()},
                {"gauge_identity", {{"zero_columns", coldef}, {"dependent_rows", rowdef}}},
                {"setup_s", clock_s() - tick}};
    }
    Vec solve(const Vec &b, Json &rec) override {
        double tick = clock_s();
        Vec r = b, d(n);
        for (int g : gauges)
            r[g] = 0;
        for (int i = 0; i < n; ++i)
            d[i] = r[perm[i]];
        int one = 1, code = 0;
        char trans = 'N';
        dgbtrs_(&trans, &n, &lo, &hi, &one, ab.data(), &ld, piv.data(), d.data(), &n, &code);
        require(!code, "BAND_SOLVE_REJECT");
        for (int i = 0; i < n; ++i)
            r[perm[i]] = d[i];
        rec = {{"defect", (s.action(r) - b).norm() / std::max(b.norm(), 1e-300)},
               {"solve_s", clock_s() - tick}};
        return r;
    }
};
} // namespace
std::unique_ptr<Access> sparse_access(const System &s) { return std::make_unique<SparseAccess>(s); }
std::unique_ptr<Access> band_access(const System &s) {
    try {
        return std::make_unique<BandAccess>(s);
    } catch (const std::runtime_error &e) {
        if (std::string(e.what()).find("BAND_FACTOR_REJECT") != 0 || s.m.n + s.m.m > 512)
            throw;
        auto a = sparse_access(s);
        a->info["band_decline"] = e.what();
        return a;
    }
}
} // namespace hpr_native
