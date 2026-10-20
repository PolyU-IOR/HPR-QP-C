#include "common.hpp"
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <unistd.h>
namespace hpr_native {
static_assert(sizeof(double) == 8 && sizeof(uint64_t) == 8, "64-bit binary transport");
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "little-endian binary transport");
namespace {
struct Reader {
    std::ifstream f;
    uint64_t remaining;
    explicit Reader(const fs::path &p) : f(p, std::ios::binary), remaining(fs::file_size(p)) {
        require(bool(f), "open input");
    }
    void bytes(void *p, size_t n) {
        require(n <= remaining, "truncated input");
        f.read(static_cast<char *>(p), n);
        require(bool(f), "read input");
        remaining -= n;
    }
    uint64_t u64() {
        uint64_t x;
        bytes(&x, 8);
        return x;
    }
    double real() {
        double x;
        bytes(&x, 8);
        return x;
    }
    void magic(const char *s) {
        char b[8];
        bytes(b, 8);
        require(std::memcmp(b, s, 8) == 0, "input magic");
    }
    Vec vec(int n) {
        require(n >= 0 && uint64_t(n) <= remaining / 8, "vector size");
        Vec v(n);
        bytes(v.data(), size_t(n) * 8);
        return v;
    }
    CSR sparse() {
        uint64_t r = u64(), c = u64(), nnz = u64();
        require(r < INT32_MAX && c < INT32_MAX && nnz < INT32_MAX, "sparse dimensions");
        require(r + 1 + 2 * nnz <= remaining / 8, "sparse size");
        std::vector<uint64_t> rp(r + 1), ci(nnz);
        std::vector<double> val(nnz);
        bytes(rp.data(), 8 * (r + 1));
        bytes(ci.data(), 8 * nnz);
        bytes(val.data(), 8 * nnz);
        require(rp.front() == 0 && rp.back() == nnz, "row offsets");
        std::vector<Triplet> t;
        t.reserve(nnz);
        for (int i = 0; i < int(r); ++i) {
            require(rp[i] <= rp[i + 1] && rp[i + 1] <= nnz, "row offsets");
            for (uint64_t k = rp[i]; k < rp[i + 1]; ++k) {
                require(ci[k] < c && std::isfinite(val[k]), "sparse entry");
                t.emplace_back(i, int(ci[k]), val[k]);
            }
        }
        CSR out(r, c);
        out.setFromTriplets(t.begin(), t.end());
        out.makeCompressed();
        return out;
    }
    void end() { require(remaining == 0, "trailing bytes"); }
};
} // namespace
void Model::validate() const {
    require(Q.rows() == n && Q.cols() == n, "Q dimensions");
    require(c.allFinite() && rhs.allFinite() && std::isfinite(constant), "finite model");
    auto bounds = [](const Vec &a, const Vec &b) {
        for (int i = 0; i < a.size(); ++i)
            require(!std::isnan(a[i]) && !std::isnan(b[i]) && a[i] <= b[i] && a[i] != INFINITY &&
                        b[i] != -INFINITY,
                    "bounds");
    };
    bounds(lo, hi);
    bounds(rl, ru);
    CSR diff = Q - CSR(Q.transpose());
    double d = 0, q = 0;
    for (int k = 0; k < diff.nonZeros(); ++k)
        d = std::max(d, std::abs(diff.valuePtr()[k]));
    for (int k = 0; k < Q.nonZeros(); ++k)
        q = std::max(q, std::abs(Q.valuePtr()[k]));
    require(d <= 1e-13 * std::max(1., q), "Q symmetry");
    int last = l;
    require(l >= 0 && l <= m, "cone suffix");
    for (auto [a, b] : cones) {
        require(a == last && b - a >= 2 && b <= m, "cone suffix");
        for (int j = a; j < b; ++j)
            require(rl[j] == -INFINITY && ru[j] == INFINITY, "cone bounds");
        last = b;
    }
    require(last == m, "cone suffix end");
}
Model read_model(const fs::path &p) {
    Reader r(p);
    r.magic("HPRNAT01");
    Model m;
    m.Q = r.sparse();
    m.A = r.sparse();
    m.n = m.A.cols();
    m.m = m.A.rows();
    m.c = r.vec(m.n);
    m.lo = r.vec(m.n);
    m.hi = r.vec(m.n);
    m.rl = r.vec(m.m);
    m.ru = r.vec(m.m);
    m.rhs = r.vec(m.m);
    m.constant = r.real();
    uint64_t nc = r.u64();
    require(nc <= uint64_t(m.m / 2), "cone count");
    m.l = m.m;
    if (nc) {
        uint64_t a = r.u64();
        require(a <= uint64_t(m.m), "cone start");
        m.l = a;
        for (uint64_t i = 0; i < nc; ++i) {
            uint64_t b = r.u64();
            require(b <= uint64_t(m.m), "cone end");
            m.cones.emplace_back(a, b);
            a = b;
        }
    }
    r.end();
    m.validate();
    return m;
}
LVec read_snapshot(const fs::path &p, const Model &m) {
    Reader r(p);
    r.magic("HPRSOLS1");
    require(r.u64() == uint64_t(m.n) && r.u64() == uint64_t(m.m) && r.u64() == uint64_t(m.n) &&
                r.u64() == uint64_t(m.n),
            "snapshot dimensions");
    Vec x = r.vec(m.n), y = r.vec(m.m), z = r.vec(m.n), w = r.vec(m.n);
    r.end();
    require(x.allFinite() && y.allFinite() && z.allFinite() && w.allFinite(), "finite snapshot");
    LVec u(m.n + m.m);
    u.head(m.n) = x.cast<long double>();
    u.tail(m.m) = y.cast<long double>();
    return u;
}
void atomic_write(const fs::path &p, const void *data, size_t n) {
    std::string name = p.string() + ".tmp-" + std::to_string(getpid());
    int fd = open(name.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    require(fd >= 0, "temporary output");
    try {
        const char *b = static_cast<const char *>(data);
        while (n) {
            ssize_t k = write(fd, b, n);
            if (k < 0 && errno == EINTR)
                continue;
            require(k > 0, "write output");
            b += k;
            n -= size_t(k);
        }
        require(fsync(fd) == 0, "sync output");
        int rc = close(fd);
        fd = -1;
        require(rc == 0, "close output");
        require(rename(name.c_str(), p.c_str()) == 0, "publish output");
    } catch (...) {
        if (fd >= 0)
            close(fd);
        unlink(name.c_str());
        throw;
    }
}
void write_candidate(const fs::path &p, const Candidate &c, const Model &m) {
    std::vector<char> b(24 + 8 * (size_t(3) * m.n + size_t(2) * m.m));
    std::memcpy(b.data(), "HPRPOL01", 8);
    uint64_t dims[2] = {uint64_t(m.n), uint64_t(m.m)};
    std::memcpy(b.data() + 8, dims, 16);
    size_t off = 24;
    for (const Vec *v : {&c.x, &c.w, &c.z, &c.y, &c.s}) {
        std::memcpy(b.data() + off, v->data(), v->size() * 8);
        off += v->size() * 8;
    }
    atomic_write(p, b.data(), b.size());
}
Candidate read_candidate(const fs::path &p, const Model &m) {
    Reader r(p);
    r.magic("HPRPOL01");
    require(r.u64() == uint64_t(m.n) && r.u64() == uint64_t(m.m), "candidate dimensions");
    Candidate c{r.vec(m.n), r.vec(m.n), r.vec(m.n), r.vec(m.m), r.vec(m.m)};
    r.end();
    return c;
}
} // namespace hpr_native
