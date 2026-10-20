// Test-only boundary for independent equation and certificate comparisons.
#include "common.hpp"
#include <iostream>

using namespace hpr_native;
Json array(const Vec &v) {
    Json a = Json::array();
    for (double x : v)
        a.push_back(x);
    return a;
}
int main(int argc, char **argv) {
    try {
        require(argc >= 4, "native_inspect mode model input [access]");
        Model m = read_model(argv[2]);
        if (std::string(argv[1]) == "check") {
            std::cout << check(m, read_candidate(argv[3], m)).dump() << '\n';
        } else {
            LVec u = read_snapshot(argv[3], m);
            System s(m, u);
            Mat J(s.matrix());
            Json rows = Json::array();
            for (int i = 0; i < J.rows(); ++i)
                rows.push_back(array(J.row(i).transpose()));
            Vec direction = Vec::LinSpaced(u.size(), -.7, .9);
            Json r = {{"F", array(s.initial.F.cast<double>())},
                      {"J", rows},
                      {"Jd", array(s.action(direction))}};
            if (argc > 4) {
                std::unique_ptr<Access> a;
                std::string name = argv[4];
                if (name == "spatial")
                    a = spatial_access(s, u, spatial_indices(m));
                else if (name == "band")
                    a = band_access(s);
                else
                    a = sparse_access(s);
                // Compatible right-hand side: evaluate access independently
                // of nonlinear readiness and nullspace representative choices.
                Vec b = s.action(direction);
                Json linear;
                Vec d = a->solve(b, linear);
                r.update({{"solution", array(d)}, {"linear", linear}, {"access", a->info}});
            }
            std::cout << r.dump() << '\n';
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
