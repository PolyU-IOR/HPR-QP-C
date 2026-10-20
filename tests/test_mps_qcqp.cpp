#include "hprqp.h"
#include "shared/mps_reader.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static HPRQPStatus load(const std::string &text, HPRQPModel *model) {
    char name[] = "/tmp/hprqp-qcqp-XXXXXX";
    int fd = mkstemp(name);
    require(fd >= 0, "temporary file");
    FILE *f = fdopen(fd, "w");
    require(f != nullptr, "fdopen");
    require(fwrite(text.data(), 1, text.size(), f) == text.size(), "fixture write");
    fclose(f);
    HPRQPStatus status = build_from_mps(name, model);
    unlink(name);
    return status;
}

static std::string fixture(const std::string &sense, const std::string &linear,
                           const std::string &rhs, const std::string &qc,
                           const std::string &extra = "") {
    return "NAME QC\nROWS\n N OBJ\n " + sense + " QC\nCOLUMNS\n X OBJ -1 QC " +
        linear + "\n Y OBJ 0\nRHS\n RHS1 OBJ -3 QC " + rhs +
        "\nBOUNDS\n UP BND X 4\n FX BND Y 0\n" + extra +
        "QCMATRIX QC\n" + qc + "ENDATA\n";
}

static void rejected(const std::string &text) {
    HPRQPModel m{};
    auto status = load(text, &m);
    model_free(&m);
    require(status != HPRQP_STATUS_OK, "unsupported/malformed input was accepted");
}

static double entry(const HPRQPMatrixCSR &a, size_t r, size_t c) {
    for (size_t k = a.row_ptr[r]; k < a.row_ptr[r+1]; ++k)
        if (a.col_ind[k] == c) return a.values[k];
    return 0.0;
}

static void check_cone(const HPRQPModel &m) {
    require(m.num_row_cones == 1 && m.A.nrow == 3, "missing quadratic row");
    require(m.Q.ncol == 2 && m.Q.nnz == 0, "unexpected objective/variables");
    require(m.c[0] == -1 && m.obj_constant == 3, "objective/offset lost");
    require(m.l[0] == 0 && m.u[0] == 4, "variable bounds lost");
    require(entry(m.A,0,0) == -1 && entry(m.A,1,0) == -1, "linear QC coefficient");
    require(std::abs(entry(m.A,2,0)-std::sqrt(2.0)) < 1e-14, "QCMATRIX factor-of-two");
    require(m.soc_rhs[0] == -5 && m.soc_rhs[1] == -3, "QC RHS sign");
    for (double x : {-1., 0., 1., 2., 3., 4.}) {
        double t=5-x, v=3-x, tail=std::sqrt(2.0)*x;
        bool soc = t >= std::hypot(v, tail)-1e-13;
        bool qc = .5*x*x+x <= 4;
        require(soc == qc, "SOC/original quadratic feasibility mismatch");
    }
}

int main(int argc, char **argv) {
    try {
        HPRQPModel m{};
        for (const auto &text : {
                fixture("L","1","4"," X X .5\n"),
                fixture("G","-1","-4"," X X -.5\n"),
                fixture("L","1","4"," X X .25\n X X 2.5D-1\n")}) {
            require(load(text, &m) == HPRQP_STATUS_OK, "valid QCQP rejected");
            check_cone(m);
            model_free(&m);
        }
        // Duplicate COO coefficients must not move subsequent rows.
        require(load("NAME DUP\nROWS\n N OBJ\n E LIN\n L QC\nCOLUMNS\n X OBJ -1 LIN 1\n"
                     " X LIN 2 QC 1\n Y QC 2\nRHS\n R LIN 3 QC 4\n"
                     "QCMATRIX QC\n Y Y .5\nENDATA\n",&m) == HPRQP_STATUS_OK, "mixed QC");
        require(m.row_cone_offsets[0] == 1 && entry(m.A,0,0) == 3 &&
                entry(m.A,1,0) == -1 && entry(m.A,1,1) == -2 &&
                m.AL[0] == 3 && m.AU[0] == 3, "duplicate row aggregation");
        model_free(&m);
        // Zero quadratic part remains an ordinary row, even with equality.
        require(load(fixture("E","1","2"," X X 0\n"),&m) == HPRQP_STATUS_OK, "zero QC");
        require(m.num_row_cones == 0 && m.A.nrow == 1 && m.AU[0] == 2, "zero QC row lost");
        model_free(&m);
        for (auto qc : {" X Y 1\n", " Z Z 1\n", " X X nan\n", " X X inf\n",
                        " X X bad\n", " X X 1e999\n", " X X\n",
                        " X X -.5\n", " X X .5\nQCMATRIX QC\n X X .5\n"})
            rejected(fixture("L","1","4",qc));
        rejected(fixture("E","1","4"," X X .5\n"));
        rejected(fixture("G","1","4"," X X .5\n"));
        rejected(fixture("L","1","4"," X X .5\n","RANGES\n R QC 2\n"));
        rejected(fixture("L","oops","4"," X X .5\n"));
        rejected(fixture("L","1","oops"," X X .5\n"));
        rejected(fixture("L","1","4"," X X .5\n","INDICATORS\n IF QC X 1\n"));
        auto text = fixture("L","1","4"," X X .5\n");
        auto unknown=text; unknown.replace(unknown.find("QCMATRIX QC"),11,"QCMATRIX ZZ");
        rejected(unknown);
        rejected(text.substr(0,text.find("ENDATA")));
        auto mixed=text; mixed.replace(mixed.find("UP BND X 4"),10,"BV BND X");
        rejected(mixed);
        rejected("NAME MAX\nOBJSENSE MAX\nROWS\n N O\nCOLUMNS\n X O 1\nENDATA\n");
        const std::string objective="NAME FULL\nROWS\n N O\nCOLUMNS\n X O 0\n Y O 0\n";
        require(load(objective+"QMATRIX\n X X 2\n X Y 1\n Y X 1\n Y Y 4\nENDATA\n",&m)
                == HPRQP_STATUS_OK, "full objective rejected");
        require(entry(m.Q,0,1)==1 && entry(m.Q,1,0)==1, "QMATRIX doubled");
        model_free(&m);
        rejected(objective+"QMATRIX\n X Y 1\nENDATA\n");
        if (argc == 2 && strcmp(argv[1],"--gpu")==0) {
            require(load(fixture("L","1","4"," X X .5\n"),&m) == HPRQP_STATUS_OK, "live load");
            HPRQPParams p; params_set_defaults(&p);
            p.use_gpu=1; p.stoptol=1e-8; p.time_limit=30;
            HPRQPResults r{};
            auto status=solve(&m,&p,&r);
            require(status==HPRQP_STATUS_OPTIMAL && r.residuals<=1e-8, "live native termination");
            require(std::abs(r.x[0]-2)<1e-6 && std::abs(r.primal_obj-1)<1e-6 &&
                    .5*r.x[0]*r.x[0]+r.x[0]<=4+1e-6, "live raw-QCQP solution");
            results_free(&r); model_free(&m);
        }
        puts("test_mps_qcqp passed");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr,"test_mps_qcqp FAILED: %s\n",e.what());
        return 1;
    }
}
