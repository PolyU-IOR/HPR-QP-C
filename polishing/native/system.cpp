#include "common.hpp"
namespace hpr_native {
Vec Derivative::action(const Vec &d) const {
    Vec o = Vec::Zero(m);
    o.head(diag.size()) = diag.cwiseProduct(d.head(diag.size()));
    for (const auto &v : cones) {
        int a = v.a, b = v.b;
        if (v.kind == 1)
            o.segment(a, b - a) = d.segment(a, b - a);
        else if (v.kind == 2)
            o.segment(a, b - a) = .5 * d.segment(a, b - a);
        else if (v.kind == 3) {
            Vec dv = d.segment(a + 1, b - a - 1);
            double qv = v.q.dot(dv);
            o[a] = .5 * (d[a] + qv);
            o.segment(a + 1, b - a - 1) = .5 * (v.q * d[a] + (1 + v.tr) * dv - v.tr * v.q * qv);
        }
    }
    return o;
}
Sparse Derivative::matrix() const {
    std::vector<Triplet> t;
    for (int i = 0; i < diag.size(); ++i)
        if (diag[i])
            t.emplace_back(i, i, diag[i]);
    for (const auto &v : cones) {
        require(v.b - v.a <= 64, "DENSE_CONE_ACCESS_RESOURCE");
        int d = v.b - v.a, a = v.a;
        if (v.kind == 1 || v.kind == 2) {
            for (int i = 0; i < d; ++i)
                t.emplace_back(a + i, a + i, v.kind == 1 ? 1. : .5);
        } else if (v.kind == 3) {
            t.emplace_back(a, a, .5);
            for (int i = 1; i < d; ++i) {
                t.emplace_back(a, a + i, .5 * v.q[i - 1]);
                t.emplace_back(a + i, a, .5 * v.q[i - 1]);
                for (int j = 1; j < d; ++j) {
                    double x = .5 * ((i == j ? 1 + v.tr : 0) - v.tr * v.q[i - 1] * v.q[j - 1]);
                    if (x)
                        t.emplace_back(a + i, a + j, x);
                }
            }
        }
    }
    Sparse s(m, m);
    s.setFromTriplets(t.begin(), t.end());
    return s;
}
Evaluation evaluate(const Model &m, const LVec &u) {
    require(u.size() == m.n + m.m && u.allFinite(), "private iterate");
    Evaluation e;
    LVec x = u.head(m.n), y = u.tail(m.m);
    e.g = m.Q.cast<long double>() * x + m.c.cast<long double>() -
          m.A.transpose().cast<long double>() * y;
    LVec raw = x - e.g;
    e.p = raw;
    e.E.resize(m.n);
    for (int i = 0; i < m.n; ++i) {
        e.p[i] = std::clamp(raw[i], (long double)m.lo[i], (long double)m.hi[i]);
        e.E[i] = raw[i] > m.lo[i] && raw[i] < m.hi[i];
    }
    LVec ax = m.A.cast<long double>() * x - m.rhs.cast<long double>(), v = ax - y;
    e.s = v;
    e.S.m = m.m;
    e.S.diag.resize(m.l);
    for (int i = 0; i < m.l; ++i) {
        e.s[i] = std::clamp(v[i], (long double)m.rl[i], (long double)m.ru[i]);
        e.S.diag[i] = v[i] > m.rl[i] && v[i] < m.ru[i];
    }
    for (auto [a, b] : m.cones) {
        long double t = v[a];
        LVec z = v.segment(a + 1, b - a - 1);
        double r = double(z.norm());
        ConeDerivative d{a, b, 0, Vec(), 0};
        if (r == 0 && t == 0) {
            e.s.segment(a, b - a).setZero();
            d.kind = 2;
        } else if (r <= t)
            d.kind = 1;
        else if (r <= -t)
            e.s.segment(a, b - a).setZero();
        else {
            LVec q = z / r;
            long double tr = t / r;
            d.kind = 3;
            d.q = q.cast<double>();
            d.tr = double(tr);
            e.s[a] = .5L * (t + r);
            e.s.segment(a + 1, b - a - 1) = .5L * (1 + tr) * z;
        }
        e.S.cones.push_back(std::move(d));
    }
    e.F.resize(m.n + m.m);
    e.F.head(m.n) = x - e.p;
    e.F.tail(m.m) = ax - e.s;
    require(e.F.allFinite(), "nonfinite residual");
    e.norm = double(e.F.norm());
    return e;
}
Candidate candidate(const Model &m, const LVec &u, const Evaluation &e) {
    Candidate c;
    c.x = e.p.cast<double>();
    c.w = c.x;
    c.z = (e.g - e.F.head(m.n)).cast<double>();
    c.y = (u.tail(m.m) - e.F.tail(m.m)).cast<double>();
    c.s = e.s.cast<double>();
    return c;
}
Vec System::action(const Vec &d) const {
    Vec dx = d.head(m.n), dy = d.tail(m.m), ad = m.A * dx;
    Vec out(m.n + m.m);
    out.head(m.n) = (Vec::Ones(m.n) - initial.E).cwiseProduct(dx) +
                    initial.E.cwiseProduct(Vec(m.Q * dx - m.A.transpose() * dy));
    out.tail(m.m) = ad - initial.S.action(ad) + initial.S.action(dy);
    return out;
}
Sparse System::matrix() const {
    uint64_t estimate = uint64_t(m.Q.nonZeros()) + 2 * uint64_t(m.A.nonZeros()) + m.n + m.m;
    for (auto [a, b] : m.cones)
        estimate +=
            uint64_t(b - a) * (uint64_t(m.A.outerIndexPtr()[b] - m.A.outerIndexPtr()[a]) + b - a);
    require(uint64_t(m.n) + m.m <= 3000000 && estimate <= 40000000, "ASSEMBLY_ACCESS_RESOURCE");
    Sparse s = initial.S.matrix(), I(m.m, m.m);
    I.setIdentity();
    Sparse bottom = (I - s) * m.A;
    std::vector<Triplet> t;
    t.reserve(estimate);
    for (int i = 0; i < m.n; ++i) {
        if (!initial.E[i])
            t.emplace_back(i, i, 1);
        else
            for (CSR::InnerIterator it(m.Q, i); it; ++it)
                t.emplace_back(i, it.col(), it.value());
    }
    for (int i = 0; i < m.m; ++i)
        for (CSR::InnerIterator it(m.A, i); it; ++it)
            if (initial.E[it.col()])
                t.emplace_back(it.col(), m.n + i, -it.value());
    for (int j = 0; j < bottom.outerSize(); ++j)
        for (Sparse::InnerIterator it(bottom, j); it; ++it)
            if (it.value())
                t.emplace_back(m.n + it.row(), it.col(), it.value());
    for (int j = 0; j < s.outerSize(); ++j)
        for (Sparse::InnerIterator it(s, j); it; ++it)
            if (it.value())
                t.emplace_back(m.n + it.row(), m.n + it.col(), it.value());
    Sparse J(m.n + m.m, m.n + m.m);
    J.setFromTriplets(t.begin(), t.end());
    require(J.nonZeros() <= 40000000, "ASSEMBLY_ACCESS_RESOURCE");
    return J;
}
} // namespace hpr_native
