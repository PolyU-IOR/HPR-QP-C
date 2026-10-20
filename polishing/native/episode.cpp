#include "common.hpp"
namespace hpr_native {
Json episode(const Model &m, const LVec &initial, double seconds, Candidate &out, bool &certified) {
    double start = clock_s();
    bool diag = true;
    for (int i = 0; i < m.n; ++i)
        for (CSR::InnerIterator it(m.Q, i); it; ++it)
            if ((it.col() != i && it.value() != 0) || (it.col() == i && it.value() < 0))
                diag = false;
    std::string family = "generic", name = "sparse";
    std::vector<int> I;
    bool cone3 = !m.cones.empty();
    for (auto [a, b] : m.cones)
        cone3 = cone3 && b - a == 3;
    if (diag && cone3) {
        I = spatial_indices(m);
        family = I.empty() ? "banded_native" : "spatial";
        name = I.empty() ? "band" : "spatial";
    } else if (diag && m.cones.empty() && m.n + m.m <= 400000 &&
               m.Q.nonZeros() + m.A.nonZeros() <= 1500000 && m.rhs.isZero(0) && m.rl == m.ru &&
               m.rl.allFinite())
        family = "equality_face";
    Json result;
    certified = false;
    try {
        double tick = clock_s();
        LVec u = initial;
        System sys(m, u);
        Evaluation e = sys.initial;
        std::unique_ptr<Access> access;
        if (name == "spatial") {
            try {
                access = spatial_access(sys, u, I);
            } catch (const std::runtime_error &ex) {
                std::string why = ex.what();
                if (why == "SPATIAL_STATE_PIVOT_UNAVAILABLE" ||
                    why == "SPATIAL_APEX_DERIVATIVE_UNSUPPORTED" ||
                    why == "SPATIAL_TENSOR_UNAVAILABLE" || why == "NULLSPACE_ACCESS_RESOURCE") {
                    access = sparse_access(sys);
                    access->info["spatial_decline"] = why;
                } else
                    throw;
            }
        } else
            access = name == "band" ? band_access(sys) : sparse_access(sys);
        Json trace = Json::array();
        std::string status = "CORRECTION_CAP";
        for (int k = 0; k < 8; ++k) {
            if (clock_s() - start >= seconds) {
                status = "TIME_ENVELOPE";
                break;
            }
            Json linear;
            Vec d = access->solve(Vec(-e.F.cast<double>()), linear);
            if (!d.allFinite()) {
                status = "NONFINITE_REJECT";
                break;
            }
            LVec trial = u + d.cast<long double>();
            Evaluation next = evaluate(m, trial);
            Candidate c = candidate(m, trial, next);
            prepare_candidate(m, c);
            Json cert = check(m, c);
            trace.push_back({{"iteration", k + 1},
                             {"base_norm", e.norm},
                             {"norm", next.norm},
                             {"linear", linear},
                             {"certificate", cert},
                             {"elapsed_s", clock_s() - tick}});
            if (cert.at("passed").get<bool>()) {
                out = std::move(c);
                certified = true;
                status = "CPU_CERTIFIED";
                break;
            }
            if (linear.at("defect").get<double>() > 1e-8) {
                status = "LINEAR_ACCURACY_REJECT";
                break;
            }
            if (next.norm > 100 * std::max(sys.initial.norm, 1e-300)) {
                status = "GROWTH_REJECT";
                break;
            }
            u = std::move(trial);
            e = std::move(next);
        }
        result = {{"status", status},
                  {"certified", certified},
                  {"trace", trace},
                  {"access", access->info},
                  {"routine_s", clock_s() - tick},
                  {"private_mantissa_bits", std::numeric_limits<long double>::digits - 1}};
    } catch (const std::exception &e) {
        result = {{"status", "ACCESS_OR_NUMERICAL_REJECT"},
                  {"certified", false},
                  {"reason", e.what()},
                  {"rejection_type", "native_exception"},
                  {"trace", Json::array()}};
        certified = false;
    }
    result.update({{"correction", "S734_common_projected_KKT"},
                   {"detected_structure", family},
                   {"requested_access", name},
                   {"adapter_s", clock_s() - start},
                   {"iterative_ssn", false},
                   {"relinearizations", 0},
                   {"derivative_refreshes", 0},
                   {"active_set_updates", 0},
                   {"partial_feedback", false},
                   {"implementation", "native_cpp"}});
    return result;
}
} // namespace hpr_native
