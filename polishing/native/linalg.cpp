#include "linalg.hpp"
#include <numeric>
extern "C" {
void dgesdd_(char *, int *, int *, double *, int *, double *, double *, int *, double *, int *,
             double *, int *, int *, int *);
void dsyev_(char *, char *, int *, double *, int *, double *, double *, int *, int *);
}
namespace hpr_native {
SVD svd(const Mat &input, bool full) {
    int m = input.rows(), n = input.cols(), k = std::min(m, n), lda = std::max(1, m),
        ldu = std::max(1, m), ldvt = std::max(1, full ? n : k), info = 0, lwork = -1;
    SVD out;
    out.s.resize(k);
    out.U = Mat::Identity(m, full ? m : k);
    out.Vt = Mat::Identity(full ? n : k, n);
    if (!k)
        return out;
    Mat a = input;
    Vec work(1);
    std::vector<int> iw(8 * k);
    char job = full ? 'A' : 'S';
    dgesdd_(&job, &m, &n, a.data(), &lda, out.s.data(), out.U.data(), &ldu, out.Vt.data(), &ldvt,
            work.data(), &lwork, iw.data(), &info);
    require(info == 0 && std::isfinite(work[0]) && work[0] < INT32_MAX, "SVD workspace");
    lwork = std::max(1, int(work[0]));
    work.resize(lwork);
    dgesdd_(&job, &m, &n, a.data(), &lda, out.s.data(), out.U.data(), &ldu, out.Vt.data(), &ldvt,
            work.data(), &lwork, iw.data(), &info);
    require(info == 0, "SVD failed");
    return out;
}
std::pair<Vec, Mat> eigen_symmetric(const Mat &input) {
    Mat a = input;
    int n = a.rows(), ld = std::max(1, n), info = 0, lwork = -1;
    Vec s(n), work(1);
    if (!n)
        return {s, a};
    char job = 'V', uplo = 'L';
    dsyev_(&job, &uplo, &n, a.data(), &ld, s.data(), work.data(), &lwork, &info);
    require(info == 0, "eigen workspace");
    lwork = std::max(1, int(work[0]));
    work.resize(lwork);
    dsyev_(&job, &uplo, &n, a.data(), &ld, s.data(), work.data(), &lwork, &info);
    require(info == 0, "eigen failed");
    return {s, a};
}
Sparse diagonal(const Vec &v) {
    Sparse a(v.size(), v.size());
    a.reserve(v.size());
    for (int i = 0; i < v.size(); ++i)
        if (v[i])
            a.insert(i, i) = v[i];
    a.makeCompressed();
    return a;
}
Sparse select(const CSR &a, const std::vector<int> &rows, const std::vector<int> &cols) {
    std::vector<int> map(a.cols(), -1);
    for (int j = 0; j < int(cols.size()); ++j)
        map[cols[j]] = j;
    std::vector<Triplet> t;
    for (int i = 0; i < int(rows.size()); ++i)
        for (CSR::InnerIterator it(a, rows[i]); it; ++it)
            if (map[it.col()] >= 0)
                t.emplace_back(i, map[it.col()], it.value());
    Sparse b(rows.size(), cols.size());
    b.setFromTriplets(t.begin(), t.end());
    return b;
}
std::vector<std::vector<int>> components(const Sparse &a) {
    require(a.rows() == a.cols(), "graph square");
    std::vector<int> parent(a.rows());
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](int i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    for (int j = 0; j < a.outerSize(); ++j)
        for (Sparse::InnerIterator it(a, j); it; ++it) {
            int u = root(it.row()), v = root(it.col());
            if (u != v)
                parent[u] = v;
        }
    std::vector<std::vector<int>> out;
    std::vector<int> index(a.rows(), -1);
    for (int i = 0; i < a.rows(); ++i) {
        int r = root(i);
        if (index[r] < 0) {
            index[r] = out.size();
            out.emplace_back();
        }
        out[index[r]].push_back(i);
    }
    return out;
}
// Adapted from SciPy 1.16.2 scipy/sparse/linalg/_isolve/minres.py.
// Copyright SciPy Developers; BSD-3-Clause, see third_party/SCIPY_LICENSE.txt.
// Paige--Saunders MINRES recurrence, matching the fixed SciPy 1.16 stopping
// tests used by the reference worker. No nonlinear stopping or refresh here.
Vec minres(const std::function<Vec(const Vec &)> &A, const std::function<Vec(const Vec &)> &M,
           const Vec &b, int &itn, int &flag) {
    const double eps = std::numeric_limits<double>::epsilon(), rtol = 1e-12;
    Vec x = Vec::Zero(b.size()), r1 = b, y = M(r1);
    double b1 = r1.dot(y);
    require(b1 >= 0, "indefinite preconditioner");
    itn = 0;
    flag = 0;
    if (b1 == 0 || b.norm() == 0)
        return x;
    b1 = std::sqrt(b1);
    double oldb = 0, beta = b1, dbar = 0, epsln = 0, phibar = b1, tnorm2 = 0, gmax = 0,
           gmin = std::numeric_limits<double>::max(), cs = -1, sn = 0;
    Vec w = Vec::Zero(b.size()), w2 = w, r2 = r1;
    int stop = 0;
    while (itn < 200) {
        ++itn;
        Vec v = y / beta;
        y = A(v);
        if (itn >= 2)
            y -= (beta / oldb) * r1;
        double alpha = v.dot(y);
        y -= (alpha / beta) * r2;
        r1 = r2;
        r2 = y;
        y = M(r2);
        oldb = beta;
        beta = r2.dot(y);
        require(beta >= 0, "non-symmetric preconditioned system");
        beta = std::sqrt(beta);
        tnorm2 += alpha * alpha + oldb * oldb + beta * beta;
        if (itn == 1 && beta / b1 <= 10 * eps)
            stop = -1;
        double oldeps = epsln, delta = cs * dbar + sn * alpha, gbar = sn * dbar - cs * alpha;
        epsln = sn * beta;
        dbar = -cs * beta;
        double root = std::hypot(gbar, dbar), gamma = std::max(std::hypot(gbar, beta), eps);
        cs = gbar / gamma;
        sn = beta / gamma;
        double phi = cs * phibar;
        phibar = sn * phibar;
        Vec w1 = w2;
        w2 = w;
        w = (v - oldeps * w1 - delta * w2) / gamma;
        x += phi * w;
        gmax = std::max(gmax, gamma);
        gmin = std::min(gmin, gamma);
        double anorm = std::sqrt(tnorm2), ynorm = x.norm(), epsx = anorm * ynorm * eps,
               test1 = ynorm && anorm ? phibar / (anorm * ynorm) : INFINITY,
               test2 = anorm ? root / anorm : INFINITY, acond = gmax / gmin;
        if (!stop) {
            if (1 + test2 <= 1)
                stop = 2;
            if (1 + test1 <= 1)
                stop = 1;
            if (itn >= 200)
                stop = 6;
            if (acond >= .1 / eps)
                stop = 4;
            if (epsx >= b1)
                stop = 3;
            if (test2 <= rtol)
                stop = 2;
            if (test1 <= rtol)
                stop = 1;
        }
        if (stop)
            break;
    }
    flag = stop == 6 ? 200 : 0;
    return x;
}
} // namespace hpr_native
