#include "linalg.hpp"
#define POCKETFFT_NO_MULTITHREADING
#include <map>
#include <numeric>
#include <pocketfft/pocketfft_hdronly.h>
namespace hpr_native {
std::vector<int> spatial_indices(const Model &m) {
    int l = m.l, g = int(std::llround(std::sqrt(l)));
    if (!l || g * g != l)
        return {};
    for (int i = 0; i < l; ++i)
        if (m.rl[i] != m.ru[i] || !std::isfinite(m.rl[i]))
            return {};
    std::vector<int> degree(m.n, 0);
    for (int i = 0; i < l; ++i)
        for (CSR::InnerIterator it(m.A, i); it; ++it)
            ++degree[it.col()];
    std::vector<int> I(l, -1), colseen(m.n, 0);
    int cols = 0, count = 0;
    for (int d : degree)
        cols += (d > 1);
    if (cols != l)
        return {};
    for (int i = 0; i < l; ++i)
        for (CSR::InnerIterator it(m.A, i); it; ++it)
            if (degree[it.col()] > 1 && it.value() > 0) {
                if (I[i] >= 0 || colseen[it.col()])
                    return {};
                I[i] = it.col();
                colseen[it.col()] = 1;
                ++count;
            }
    if (count != l)
        return {};
    std::vector<int> inv(m.n, -1);
    for (int i = 0; i < l; ++i)
        inv[I[i]] = i;
    for (int i = 0; i < l; ++i) {
        if (m.Q.coeff(I[i], I[i]) <= 0)
            return {};
        int seen = 0;
        for (CSR::InnerIterator it(m.A, i); it; ++it)
            if (inv[it.col()] >= 0) {
                int j = inv[it.col()];
                double want = j == i ? 4.
                              : ((j == i - 1 && i % g) || (j == i + 1 && i % g < g - 1) ||
                                 j == i - g || j == i + g)
                                  ? -1.
                                  : 0.;
                if (it.value() != want)
                    return {};
                seen += (want != 0);
            }
        int needed = 1 + (i % g > 0) + (i % g < g - 1) + (i >= g) + (i + g < l);
        if (seen != needed)
            return {};
    }
    return I;
}
namespace {
class SpatialAccess final : public Access {
    const System &s;
    const Model &m;
    int g, l, nb, nc, size;
    std::vector<int> I, B;
    Vec qs, qb, denom;
    Sparse L, C, R, S, P, H, Hi, Z, pre;
    Mat GU;
    bool prepared = false;
    Json ga, pm;
    struct Lift {
        Vec fixed, ds0, dl0, y1, rhs;
    };
    Vec poisson(const Vec &b) const {
        require(b.size() == l, "Poisson dimension");
        Vec v = b;
        pocketfft::shape_t shape{size_t(g), size_t(g)}, axes{0, 1};
        pocketfft::stride_t stride{ptrdiff_t(8 * g), 8};
        double scale = 1. / (2 * (g + 1));
        pocketfft::dst(shape, stride, stride, axes, 1, v.data(), v.data(), scale, false, 1);
        v.array() /= denom.array();
        pocketfft::dst(shape, stride, stride, axes, 1, v.data(), v.data(), scale, false, 1);
        return v;
    }
    Vec action(const Vec &z) const {
        Vec b = z.head(nb), yy = z.tail(nc), yc = H * yy, ds = poisson(L * b),
            dl = poisson(Vec(qs.cwiseProduct(ds) + C.transpose() * yc));
        Vec out(size);
        out.head(nb) = qb.cwiseProduct(b) - R.transpose() * yc + L.transpose() * dl;
        out.tail(nc) = -H * (R * b - C * ds) - S * yy;
        return out;
    }
    Lift rhs_lift(const Vec &b) const {
        Lift o;
        o.fixed = Vec::Zero(m.n);
        for (int i = 0; i < m.n; ++i)
            if (!s.initial.E[i])
                o.fixed[i] = b[i];
        Vec af = m.A * o.fixed, rg = b.tail(m.m) - af + s.initial.S.action(af);
        o.ds0 = poisson(rg.head(l));
        Vec rfI(l), rfB(nb);
        for (int i = 0; i < l; ++i)
            rfI[i] = b[I[i]];
        for (int i = 0; i < nb; ++i)
            rfB[i] = b[B[i]];
        o.dl0 = poisson(Vec(qs.cwiseProduct(o.ds0) - rfI));
        Vec f = rfB + L.transpose() * o.dl0, cd = C * o.ds0, h = rg.tail(nc) - cd + S * cd;
        o.y1 = P * h;
        o.rhs.resize(size);
        o.rhs.head(nb) = f + R.transpose() * o.y1 - L.transpose() * poisson(C.transpose() * o.y1);
        o.rhs.tail(nc) = -Hi * h;
        return o;
    }
    Vec lift(const Lift &o, const Vec &z) const {
        Vec b = z.head(nb), yc = H * z.tail(nc) + o.y1, ds = poisson(L * b),
            dl = poisson(Vec(qs.cwiseProduct(ds) + C.transpose() * yc));
        Vec d(m.n + m.m);
        d.head(m.n) = o.fixed;
        for (int i = 0; i < nb; ++i)
            d[B[i]] = b[i];
        for (int i = 0; i < l; ++i)
            d[I[i]] = o.ds0[i] - ds[i];
        d.segment(m.n, l) = o.dl0 - dl;
        d.tail(nc) = yc;
        return d;
    }
    void gauges() {
        Sparse hr = H * R;
        std::vector<Triplet> t;
        for (int i = 0; i < nb; ++i)
            if (qb[i])
                t.emplace_back(i, i, std::sqrt(qb[i]));
        for (int j = 0; j < L.outerSize(); ++j)
            for (Sparse::InnerIterator it(L, j); it; ++it)
                t.emplace_back(nb + it.row(), it.col(), it.value());
        for (int j = 0; j < hr.outerSize(); ++j)
            for (Sparse::InnerIterator it(hr, j); it; ++it)
                if (it.value())
                    t.emplace_back(nb + l + it.row(), it.col(), it.value());
        Sparse gc(nb + l + nc, nb);
        gc.setFromTriplets(t.begin(), t.end());
        Sparse ac = gc.cwiseAbs();
        auto groups = components(Sparse(ac.transpose() * ac));
        std::vector<Triplet> zt;
        int count = 0, largest = 0;
        for (const auto &ix : groups) {
            int k = ix.size();
            largest = std::max(largest, k);
            require(k <= 32, "SPATIAL_GAUGE_COMPONENT_RESOURCE");
            std::map<int, int> rows;
            for (int j : ix)
                for (Sparse::InnerIterator it(gc, j); it; ++it)
                    rows.emplace(it.row(), 0);
            int row = 0;
            for (auto &v : rows)
                v.second = row++;
            Mat a = Mat::Zero(rows.size(), k);
            for (int j = 0; j < k; ++j)
                for (Sparse::InnerIterator it(gc, ix[j]); it; ++it)
                    a(rows.at(it.row()), j) += it.value();
            Mat basis;
            if (!a.rows())
                basis = Mat::Identity(k, k);
            else {
                auto d = svd(a, true);
                double cutoff = 32 * std::max(a.rows(), a.cols()) *
                                std::numeric_limits<double>::epsilon() * d.s[0];
                int rank = 0;
                while (rank < d.s.size() && d.s[rank] > cutoff)
                    ++rank;
                basis = d.Vt.bottomRows(k - rank).transpose();
            }
            for (int col = 0; col < basis.cols(); ++col) {
                for (int j = 0; j < k; ++j)
                    if (basis(j, col))
                        zt.emplace_back(ix[j], count, basis(j, col));
                ++count;
            }
        }
        Z.resize(size, count);
        Z.setFromTriplets(zt.begin(), zt.end());
        double defect = 0;
        for (int j = 0; j < count; ++j)
            defect = std::max(defect, action(Vec(Z.col(j))).norm());
        require(defect <= 1e-11, "SPATIAL_GAUGE_IDENTITY_REJECT");
        ga = {{"count", count}, {"action_defect", defect}, {"largest_component", largest}};
        if (count) {
            require(uint64_t(8) * (m.n + m.m) * count <= uint64_t(8) * 1024 * 1024 * 1024,
                    "NULLSPACE_ACCESS_RESOURCE");
            Lift o = rhs_lift(Vec::Zero(m.n + m.m));
            Mat G(m.n + m.m, count);
            for (int j = 0; j < count; ++j)
                G.col(j) = lift(o, Vec(Z.col(j)));
            auto d = svd(G);
            require(d.s.minCoeff() > d.s.maxCoeff() * std::max(G.rows(), G.cols()) *
                                         std::numeric_limits<double>::epsilon(),
                    "FULL_NULLSPACE_RANK_REJECT");
            GU = std::move(d.U);
        }
    }
    void preconditioner() {
        Sparse T = .25 * L, W = R - C * T, Q = diagonal(qb) + T.transpose() * diagonal(qs) * T,
               K = -W.transpose() * H;
        std::vector<Triplet> t;
        for (int j = 0; j < Q.outerSize(); ++j)
            for (Sparse::InnerIterator it(Q, j); it; ++it)
                if (it.value())
                    t.emplace_back(it.row(), it.col(), it.value());
        for (int j = 0; j < K.outerSize(); ++j)
            for (Sparse::InnerIterator it(K, j); it; ++it)
                if (it.value()) {
                    t.emplace_back(it.row(), nb + it.col(), it.value());
                    t.emplace_back(nb + it.col(), it.row(), it.value());
                }
        for (int j = 0; j < S.outerSize(); ++j)
            for (Sparse::InnerIterator it(S, j); it; ++it)
                if (it.value())
                    t.emplace_back(nb + it.row(), nb + it.col(), -it.value());
        Sparse approx(size, size);
        approx.setFromTriplets(t.begin(), t.end());
        auto groups = components(approx);
        t.clear();
        int floored = 0, largest = 0;
        for (const auto &ix : groups) {
            int k = ix.size();
            largest = std::max(k, largest);
            require(k <= 32, "SPATIAL_PRECONDITIONER_COMPONENT_RESOURCE");
            Mat a(k, k);
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < k; ++j)
                    a(i, j) = approx.coeff(ix[i], ix[j]);
            auto [v, U] = eigen_symmetric(Mat(.5 * (a + a.transpose())));
            double cut = std::max(1e-12, 1e-3 * infnorm(v));
            for (int i = 0; i < k; ++i) {
                floored += std::abs(v[i]) < cut;
                v[i] = 1 / std::max(std::abs(v[i]), cut);
            }
            Mat p = U * v.asDiagonal() * U.transpose();
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < k; ++j)
                    t.emplace_back(ix[i], ix[j], p(i, j));
        }
        pre.resize(size, size);
        pre.setFromTriplets(t.begin(), t.end());
        pm = {{"blocks", groups.size()},
              {"largest_block", largest},
              {"floored_eigenvalues", floored},
              {"nnz", pre.nonZeros()}};
        prepared = true;
        info["preconditioners"] = 1;
    }

  public:
    SpatialAccess(const System &sys, const LVec &u, const std::vector<int> &indices)
        : s(sys), m(s.m), g(int(std::llround(std::sqrt(m.l)))), l(m.l), I(indices) {
        double tick = clock_s();
        require(!I.empty(), "SPATIAL_TENSOR_UNAVAILABLE");
        std::vector<bool> state(m.n, false);
        for (int i : I) {
            require(s.initial.E[i] == 1, "SPATIAL_STATE_PIVOT_UNAVAILABLE");
            state[i] = true;
        }
        for (int i = 0; i < m.n; ++i)
            if (s.initial.E[i] && !state[i])
                B.push_back(i);
        nb = B.size();
        nc = m.m - l;
        size = nb + nc;
        Vec q = m.Q.diagonal();
        qs.resize(l);
        qb.resize(nb);
        for (int i = 0; i < l; ++i)
            qs[i] = q[I[i]];
        for (int i = 0; i < nb; ++i)
            qb[i] = q[B[i]];
        std::vector<int> lr(l), cr(nc);
        std::iota(lr.begin(), lr.end(), 0);
        std::iota(cr.begin(), cr.end(), l);
        L = select(m.A, lr, B);
        C = select(m.A, cr, I);
        R = select(m.A, cr, B);
        Sparse sf = s.initial.S.matrix();
        S = sf.bottomRightCorner(nc, nc);
        std::vector<Triplet> pt, ht, hit;
        LVec raw = m.A.cast<long double>() * u.head(m.n) - m.rhs.cast<long double>() - u.tail(m.m);
        for (auto [a, b] : m.cones) {
            require(b - a == 3, "SPATIAL_CONE_UNSUPPORTED");
            long double tv = raw[a], rv = raw.segment(a + 1, 2).norm();
            require(!(tv == 0 && rv == 0), "SPATIAL_APEX_DERIVATIVE_UNSUPPORTED");
            Mat p = Mat::Zero(3, 3), h = p, hi = p;
            if (rv <= tv)
                p.setIdentity();
            else if (rv <= -tv) {
                h.setIdentity();
                hi.setIdentity();
            } else {
                Eigen::Vector3d ep, em, et;
                double q1 = double(raw[a + 1] / rv), q2 = double(raw[a + 2] / rv);
                ep << 1, q1, q2;
                em << 1, -q1, -q2;
                et << 0, -q2, q1;
                ep /= std::sqrt(2.);
                em /= std::sqrt(2.);
                double gamma = double((rv - tv) / (2 * rv));
                require(gamma > 0, "SPATIAL_MIDDLE_EIGENVALUE");
                p = ep * ep.transpose();
                h = em * em.transpose() + std::sqrt(gamma) * et * et.transpose();
                hi = em * em.transpose() + et * et.transpose() / std::sqrt(gamma);
            }
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    if (p(i, j))
                        pt.emplace_back(a - l + i, a - l + j, p(i, j));
                    if (h(i, j))
                        ht.emplace_back(a - l + i, a - l + j, h(i, j));
                    if (hi(i, j))
                        hit.emplace_back(a - l + i, a - l + j, hi(i, j));
                }
        }
        P.resize(nc, nc);
        H.resize(nc, nc);
        Hi.resize(nc, nc);
        P.setFromTriplets(pt.begin(), pt.end());
        H.setFromTriplets(ht.begin(), ht.end());
        Hi.setFromTriplets(hit.begin(), hit.end());
        Vec lam(g);
        for (int i = 0; i < g; ++i) {
            double v = std::sin(std::acos(-1.) * (i + 1) / (2 * (g + 1)));
            lam[i] = 4 * v * v;
        }
        denom.resize(l);
        for (int i = 0; i < g; ++i)
            for (int j = 0; j < g; ++j)
                denom[i * g + j] = lam[i] + lam[j];
        gauges();
        info = {{"access", "spatial"}, {"boundary_order", size}, {"gauges", ga},
                {"factorizations", 0}, {"preconditioners", 0},   {"setup_s", clock_s() - tick}};
    }
    Vec solve(const Vec &b, Json &rec) override {
        double tick = clock_s();
        Lift o = rhs_lift(b);
        double compatibility = (Z.transpose() * o.rhs).norm() / std::max(o.rhs.norm(), 1e-300);
        require(compatibility <= 1e-10, "SPATIAL_RANGE_REJECT");
        if (!prepared)
            preconditioner();
        auto a = [&](const Vec &v) -> Vec { return action(v) + Z * (Z.transpose() * v); };
        auto pmv = [&](const Vec &v) -> Vec {
            Vec z = Z * (Z.transpose() * v), r = pre * (v - z);
            return r - Z * (Z.transpose() * r) + z;
        };
        Vec rhs = o.rhs - Z * (Z.transpose() * o.rhs);
        int its, flag;
        Vec z = minres(a, pmv, rhs, its, flag), d = lift(o, z);
        if (GU.cols())
            d -= GU * (GU.transpose() * d);
        rec = {{"defect", (s.action(d) - b).norm() / std::max(b.norm(), 1e-300)},
               {"relative_compatibility", compatibility},
               {"iterations", its},
               {"minres_info", flag},
               {"boundary_defect", (action(z) - o.rhs).norm() / std::max(o.rhs.norm(), 1e-300)},
               {"solve_s", clock_s() - tick},
               {"preconditioner", pm}};
        return d;
    }
};
} // namespace
std::unique_ptr<Access> spatial_access(const System &s, const LVec &u, const std::vector<int> &I) {
    return std::make_unique<SpatialAccess>(s, u, I);
}
} // namespace hpr_native
