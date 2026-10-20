#include "common.hpp"
#include <limits>
namespace hpr_native {
namespace {
// Deliberately independent from the correction projector, derivative and slack.
Vec project_soc(const Vec &v) {
    double t = v[0], r = v.tail(v.size() - 1).norm();
    require(std::isfinite(r), "checker cone overflow");
    if (r <= t)
        return v;
    if (r <= -t)
        return Vec::Zero(v.size());
    Vec o(v.size());
    o[0] = .5 * t + .5 * r;
    o.tail(v.size() - 1) = (.5 + .5 * t / r) * v.tail(v.size() - 1);
    return o;
}
Vec interval(const Vec &v, const Vec &lo, const Vec &hi) { return v.cwiseMax(lo).cwiseMin(hi); }
Vec dual_interval(Vec v, const Vec &lo, const Vec &hi) {
    for (int i = 0; i < v.size(); ++i) {
        if (lo[i] == -INFINITY)
            v[i] = std::min(v[i], 0.);
        if (hi[i] == INFINITY)
            v[i] = std::max(v[i], 0.);
    }
    return v;
}
double dot(const Vec &a, const Vec &b) {
    long double sum = 0;
    for (int i = 0; i < a.size(); ++i)
        sum += (long double)a[i] * b[i];
    return double(sum);
}
double support(const Vec &v, const Vec &lo, const Vec &hi) {
    long double pos = 0, neg = 0;
    for (int i = 0; i < v.size(); ++i) {
        if (v[i] > 0) {
            if (lo[i] == -INFINITY)
                return -INFINITY;
            pos += (long double)v[i] * lo[i];
        } else if (v[i] < 0) {
            if (hi[i] == INFINITY)
                return -INFINITY;
            neg += (long double)v[i] * hi[i];
        }
    }
    return double(pos) + double(neg);
}
} // namespace
void prepare_candidate(const Model &m, Candidate &c) {
    c.z = dual_interval(c.z, m.lo, m.hi);
    c.y.head(m.l) = dual_interval(c.y.head(m.l), m.rl.head(m.l), m.ru.head(m.l));
    for (auto [a, b] : m.cones) {
        Vec v = project_soc(c.y.segment(a, b - a));
        long double sum = 0;
        for (int j = 1; j < v.size(); ++j)
            sum += (long double)v[j] * v[j];
        long double r = std::sqrt(sum);
        if (r > 0) {
            long double margin = 8 * std::numeric_limits<double>::epsilon() * (v.size() - 1) * r;
            double upper = std::nextafter(double(r + margin), INFINITY);
            v[0] = std::max(v[0], upper);
        }
        c.y.segment(a, b - a) = v;
    }
}
Json check(const Model &m, const Candidate &c, double tol) {
    require(std::isfinite(tol) && tol > 0, "checker tolerance");
    require(c.x.size() == m.n && c.w.size() == m.n && c.z.size() == m.n && c.y.size() == m.m &&
                c.s.size() == m.m,
            "checker dimensions");
    for (const Vec *v : {&c.x, &c.w, &c.z, &c.y, &c.s})
        require(v->allFinite(), "checker candidate finite");
    Vec qx = m.Q * c.x, qw = m.Q * c.w, ax = m.A * c.x, aty = m.A.transpose() * c.y;
    require(qx.allFinite() && qw.allFinite() && ax.allFinite() && aty.allFinite(),
            "checker product overflow");
    double bn = infnorm(m.rhs);
    for (int i = 0; i < m.m; ++i) {
        if (std::isfinite(m.rl[i]))
            bn = std::max(bn, std::abs(m.rl[i]));
        if (std::isfinite(m.ru[i]))
            bn = std::max(bn, std::abs(m.ru[i]));
    }
    Vec shifted = ax - m.rhs, p = interval(shifted, m.rl, m.ru);
    for (auto [a, b] : m.cones)
        p.segment(a, b - a) = project_soc(shifted.segment(a, b - a));
    double row = infnorm(Vec(p - shifted)) / (1 + std::max(bn, infnorm(ax))),
           box = infnorm(Vec(c.x - interval(c.x, m.lo, m.hi))), primal_error = std::max(row, box);
    double station = infnorm(Vec(qw + m.c - aty - c.z)) /
                     (1 + std::max({infnorm(m.c), infnorm(qw), infnorm(aty)}));
    Vec zp = dual_interval(c.z, m.lo, m.hi), yp = c.y;
    yp.head(m.l) = dual_interval(c.y.head(m.l), m.rl.head(m.l), m.ru.head(m.l));
    bool cone = true;
    for (auto [a, b] : m.cones) {
        yp.segment(a, b - a) = project_soc(c.y.segment(a, b - a));
        cone = cone && (c.y[a] >= c.y.segment(a + 1, b - a - 1).norm());
    }
    double domain = std::max(infnorm(Vec(c.z - zp)) / (1 + infnorm(c.z)),
                             infnorm(Vec(c.y - yp)) / (1 + infnorm(c.y)));
    double zs = support(c.z, m.lo, m.hi),
           ys = support(c.y.head(m.l), m.rl.head(m.l), m.ru.head(m.l));
    bool finite = cone && std::isfinite(zs + ys);
    double primal = .5 * dot(c.x, qx) + dot(m.c, c.x) + m.constant,
           dual = finite ? -.5 * dot(c.w, qw) + zs + ys + dot(c.y, m.rhs) + m.constant : -INFINITY;
    double de = finite ? std::max(station, domain) : INFINITY;
    double gap = std::isfinite(primal) && std::isfinite(dual)
                     ? std::abs(primal - dual) / (1 + std::max(std::abs(primal), std::abs(dual)))
                     : INFINITY;
    double error = std::max({primal_error, de, gap});
    bool pass =
        std::isfinite(error) && std::isfinite(primal) && std::isfinite(dual) && error <= tol;
    return {{"passed", pass},
            {"tolerance", tol},
            {"primal_feasibility", primal_error},
            {"dual_feasibility", de},
            {"relative_gap", gap},
            {"error", error},
            {"primal_objective", primal},
            {"dual_objective", dual},
            {"row_feasibility", row},
            {"variable_bounds", box},
            {"stationarity", station},
            {"dual_domain", domain},
            {"finite_dual_support", finite},
            {"checker", "independent_unscaled_native_cpu_cpp_v1"}};
}
} // namespace hpr_native
