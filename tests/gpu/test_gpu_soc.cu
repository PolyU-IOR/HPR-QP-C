#include "hprqp.h"
#include "shared/build.h"
#include "cuda_common.h"
#include "soc_common.cuh"
#include <vector>
#include <algorithm>
#include <limits>

#define REQUIRE(c) do { if (!(c)) { fprintf(stderr, "SOC check failed at %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)
#define OK(c) REQUIRE((c) == HPRQP_STATUS_OK)
using Vec = std::vector<double>;

static HPRQPMatrixCSR matrix(const Vec& a, size_t m, size_t n)
{
    HPRQPMatrixCSR out = {};
    size_t nnz = 0, k = 0;
    for (double v : a) if (v != 0.0) ++nnz;
    OK(matrix_csr_alloc(&out, m, n, nnz));
    for (size_t i = 0; i < m; ++i) {
        out.row_ptr[i] = k;
        for (size_t j = 0; j < n; ++j) if (a[i*n+j] != 0.0) {
            out.col_ind[k] = j; out.values[k++] = a[i*n+j];
        }
    }
    out.row_ptr[m] = k;
    return out;
}

static Vec product(const HPRQPMatrixCSR& a, const Vec& x)
{
    Vec v(a.nrow, 0.0);
    for (size_t i = 0; i < a.nrow; ++i) {
        long double s = 0;
        for (size_t j = a.row_ptr[i]; j < a.row_ptr[i+1]; ++j) s += (long double)a.values[j] * x[a.col_ind[j]];
        v[i] = (double)s;
    }
    return v;
}

static Vec projection(Vec x, size_t first = 0, size_t end = 0)
{
    if (!end) end = x.size();
    long double norm = 0.0L, t = x[first];
    for (size_t j = first+1; j < end; ++j) norm = hypotl(norm, x[j]);
    if (norm <= t) return x;
    if (norm <= -t) {
        for (size_t j = first; j < end; ++j) x[j] = 0;
    } else {
        x[first] = (double)((norm+t)/2);
        for (size_t j = first+1; j < end; ++j) x[j] = (double)((1+t/norm)*x[j]/2);
    }
    return x;
}

static double infnorm(const Vec& v)
{
    double n = 0;
    for (double x : v) { REQUIRE(std::isfinite(x)); n = std::max(n, fabs(x)); }
    return n;
}
static void near(const Vec& a, const Vec& b, double tol)
{
    REQUIRE(a.size() == b.size());
    double scale = 1.0 + infnorm(b);
    for (size_t i = 0; i < a.size(); ++i) REQUIRE(fabs(a[i]-b[i])/scale <= tol);
}
static double dot(const Vec& x, const Vec& y)
{
    long double s = 0;
    for (size_t j = 0; j < x.size(); ++j) s += (long double)x[j]*y[j];
    return (double)s;
}

static void projection_edges()
{
    std::vector<Vec> cases = {{0}, {-2}, {3,1,1}, {-3,1,1}, {0,3,4},
        {5,3,4}, {-5,3,4}, {0,3e200,4e200}, {0,3e-200,4e-200}};
    for (size_t n : {33, 257, 4097}) {
        Vec v(n, 0.25); v[0] = -0.5; cases.push_back(v);
    }
    for (const Vec& v : cases) {
        HPRQPGPUVector in = {}, out = {}; size_t *offsets = NULL;
        size_t host_offsets[] = {0, v.size()};
        OK(gpu_vector_alloc(&in, v.size())); OK(gpu_vector_alloc(&out, v.size()));
        OK(gpu_vector_copy_from_host(&in, v.data(), v.size()));
        REQUIRE(cudaMalloc((void **)&offsets, 2*sizeof(size_t)) == cudaSuccess);
        REQUIRE(cudaMemcpy(offsets, host_offsets, 2*sizeof(size_t), cudaMemcpyHostToDevice) == cudaSuccess);
        OK(gpu_soc_project(out.values, in.values, NULL, NULL, 0, offsets, 1));
        Vec got(v.size()); OK(gpu_vector_copy_to_host(&out, got.data(), got.size()));
        Vec expected = projection(v);
        double scale = std::max(infnorm(v), std::numeric_limits<double>::min());
        for (size_t j = 0; j < v.size(); ++j) {
            if (!(fabs(got[j]-expected[j])/scale < 2e-14))
                fprintf(stderr,"projection n=%zu j=%zu input=%.17g got=%.17g expected=%.17g scaled_error=%.17g\n",v.size(),j,v[j],got[j],expected[j],fabs(got[j]-expected[j])/scale);
            REQUIRE(fabs(got[j]-expected[j])/scale < 2e-14);
        }
        // Moreau orthogonality, membership and in-place behavior on moderate data.
        if (scale > 1e-100 && scale < 1e100) {
            Vec dual(v.size());
            for (size_t j = 0; j < v.size(); ++j) dual[j] = got[j]-v[j];
            near(projection(dual), dual, 2e-14);
            REQUIRE(fabs(dot(got,dual)) < 2e-12*(1+dot(v,v)));
        }
        OK(gpu_soc_project(in.values, in.values, NULL, NULL, 0, offsets, 1));
        OK(gpu_vector_copy_to_host(&in, got.data(), got.size()));
        for (size_t j = 0; j < v.size(); ++j) {
            if (!(fabs(got[j]-expected[j])/scale < 2e-14))
                fprintf(stderr,"projection n=%zu j=%zu input=%.17g got=%.17g expected=%.17g scaled_error=%.17g\n",v.size(),j,v[j],got[j],expected[j],fabs(got[j]-expected[j])/scale);
            REQUIRE(fabs(got[j]-expected[j])/scale < 2e-14);
        }
        cudaFree(offsets); gpu_vector_free(&out); gpu_vector_free(&in);
    }
    puts("SOC projection: 12 edge/large/extreme-magnitude cases passed");
}

static HPRQPModel mixed_model(int q_kind)
{
    const size_t n = 8, m = 7;
    Vec a(m*n), q(n*n, 0.0), c(n), lower(n,-INFINITY), upper(n,INFINITY);
    Vec al(m,-INFINITY), au(m,INFINITY), rhs(m,0.0);
    Vec x = {0.25,-0.5,5,3,4,2,0,0};
    Vec y = {0.2,0.3,2,-2,0,0,0}, z = {0.4,0,1,-0.6,-0.8,0,0,0};
    for (size_t i = 0; i < m; ++i) for (size_t j = 0; j < n; ++j)
        a[i*n+j] = (i == 3) ? 0.0 : ((int)((i*3+j*5+1)%11)-5)/7.0;
    for (size_t j = 0; j < n; ++j) {
        if (q_kind) q[j*n+j] = 1.0 + j/8.0;
        if (q_kind == 2 && j+1 < n) {
            q[j*n+j] += 0.25; q[(j+1)*n+j+1] += 0.25;
            q[j*n+j+1] = q[(j+1)*n+j] = -0.25;
        }
    }
    HPRQPMatrixCSR Q = matrix(q,n,n), A = matrix(a,m,n), AT = {};
    OK(matrix_csr_transpose(&AT,&A));
    Vec ax=product(A,x), aty=product(AT,y), qx=product(Q,x);
    for (size_t j=0;j<n;++j) c[j]=aty[j]+z[j]-qx[j];
    lower[0]=x[0]; upper[0]=1.0; lower[1]=-1; upper[1]=1;
    al[0]=au[0]=ax[0]; al[1]=ax[1];
    Vec slack={ax[0],ax[1],1,1,3,0,0};
    for (size_t i=2;i<m;++i) rhs[i]=ax[i]-slack[i];
    size_t vars[]={2,5,8}, rows[]={2,4,7};
    HPRQPSOCSpec cones={2,vars,2,rows,rhs.data()};
    HPRQPModel model={};
    OK(build_from_QAbc_soc(&Q,&A,c.data(),al.data(),au.data(),lower.data(),upper.data(),0,&cones,&model));
    REQUIRE(model.A.nrow==m && model.A.row_ptr[3]==model.A.row_ptr[4]);
    HPRQPModel copy={}; OK(model_copy(&copy,&model));
    REQUIRE(copy.var_cone_offsets != model.var_cone_offsets && copy.soc_rhs != model.soc_rhs);
    REQUIRE(copy.num_var_cones==2 && copy.num_row_cones==2 && copy.soc_rhs[3]==-1);
    model_free(&model); matrix_csr_free(&Q); matrix_csr_free(&A); matrix_csr_free(&AT);
    return copy;
}

static void certify(const HPRQPModel& p, const HPRQPResults& r, double tol)
{
    Vec x(r.x,r.x+p.Q.ncol), y(p.A.nrow), z(r.z,r.z+p.Q.ncol);
    for(size_t i=0;i<y.size();++i)y[i]=r.y[i];
    Vec qx=product(p.Q,x), ax=product(p.A,x), aty=product(p.AT,y), rd(x.size());
    size_t linear_n=p.num_var_cones?p.var_cone_offsets[0]:x.size();
    size_t linear_m=p.num_row_cones?p.row_cone_offsets[0]:y.size();
    double primal=0, dual=0, support=0;
    for (size_t j=0;j<x.size();++j) rd[j]=qx[j]+p.c[j]-aty[j]-z[j];
    for (size_t j=0;j<linear_n;++j) {
        primal=std::max(primal,std::max(p.l[j]-x[j],x[j]-p.u[j]));
        if (z[j]>0 && std::isfinite(p.l[j])) support+=z[j]*p.l[j];
        else if (z[j]<0 && std::isfinite(p.u[j])) support+=z[j]*p.u[j];
        else dual=std::max(dual,fabs(z[j]));
    }
    for (size_t i=0;i<linear_m;++i) {
        primal=std::max(primal,std::max(p.AL[i]-ax[i],ax[i]-p.AU[i]));
        if (y[i]>0 && std::isfinite(p.AL[i])) support+=y[i]*p.AL[i];
        else if (y[i]<0 && std::isfinite(p.AU[i])) support+=y[i]*p.AU[i];
        else dual=std::max(dual,fabs(y[i]));
    }
    for (size_t k=0;k<p.num_var_cones;++k) {
        Vec px=projection(x,p.var_cone_offsets[k],p.var_cone_offsets[k+1]);
        Vec pz=projection(z,p.var_cone_offsets[k],p.var_cone_offsets[k+1]);
        for (size_t j=p.var_cone_offsets[k];j<p.var_cone_offsets[k+1];++j) {
            primal=std::max(primal,fabs(px[j]-x[j])); dual=std::max(dual,fabs(pz[j]-z[j]));
        }
    }
    Vec slack=ax;
    if (p.num_row_cones) for (size_t i=linear_m;i<y.size();++i) { slack[i]-=p.soc_rhs[i]; support+=y[i]*p.soc_rhs[i]; }
    for (size_t k=0;k<p.num_row_cones;++k) {
        Vec ps=projection(slack,p.row_cone_offsets[k],p.row_cone_offsets[k+1]);
        Vec py=projection(y,p.row_cone_offsets[k],p.row_cone_offsets[k+1]);
        for (size_t i=p.row_cone_offsets[k];i<p.row_cone_offsets[k+1];++i) {
            primal=std::max(primal,fabs(ps[i]-slack[i])); dual=std::max(dual,fabs(py[i]-y[i]));
        }
    }
    Vec c(p.c,p.c+x.size());
    double pobj=0.5*dot(x,qx)+dot(c,x)+p.obj_constant;
    double dobj=-0.5*dot(x,qx)+support+p.obj_constant;
    double gap=fabs(pobj-dobj)/(1+std::max(fabs(pobj),fabs(dobj)));
    double stationarity=infnorm(rd)/(1+std::max(infnorm(c),std::max(infnorm(qx),infnorm(aty))));
    printf("  independent original: primal=%.3g dual_cone=%.3g stationarity=%.3g gap=%.3g\n",primal,dual,stationarity,gap);
    REQUIRE(primal<tol && dual<tol && stationarity<tol && gap<tol);
}

static void mixed_solves()
{
    for (int q=0;q<3;++q) for (int scaling=0;scaling<=1;++scaling) for (int custom=0;custom<=1;++custom) {
        HPRQPModel p=mixed_model(q);
        HPRQPParams params; params_set_defaults(&params);
        params.verbose=0; params.print_frequency=0; params.stoptol=1e-9;
        params.use_scaling=scaling; params.max_iter=100000; params.time_limit=30;
        params.spmv_mode_a=params.spmv_mode_q=custom?HPRQP_SPMV_MODE_CUSTOMIZED:HPRQP_SPMV_MODE_AUTO;
        HPRQPResults results={}; HPRQPStatus status=solve(&p,&params,&results);
        printf("mixed Q=%d scaling=%d custom=%d status=%d iter=%zu score=%.3g\n",q,scaling,custom,(int)status,results.iter,results.residuals);
        REQUIRE(status==HPRQP_STATUS_OPTIMAL); certify(p,results,2e-6);
        if (q) near(Vec(results.x,results.x+8),{0.25,-0.5,5,3,4,2,0,0},2e-5);
        params.use_gpu=0; HPRQPResults rejected={};
        REQUIRE(solve(&p,&params,&rejected)==HPRQP_STATUS_NOT_IMPLEMENTED);
        results_free(&results); model_free(&p);
    }
}

static void invalid_metadata()
{
    HPRQPModel p=mixed_model(1), out={};
    size_t invalid[]={2,2,8}, valid[]={2,5,8}, bad_end[]={2,5,9};
    HPRQPSOCSpec spec={2,invalid,p.num_row_cones,p.row_cone_offsets,p.soc_rhs};
    REQUIRE(build_from_QAbc_soc(&p.Q,&p.A,p.c,p.AL,p.AU,p.l,p.u,0,&spec,&out)==HPRQP_STATUS_INVALID_ARGUMENT);
    spec.var_offsets=bad_end;
    REQUIRE(build_from_QAbc_soc(&p.Q,&p.A,p.c,p.AL,p.AU,p.l,p.u,0,&spec,&out)==HPRQP_STATUS_INVALID_ARGUMENT);
    spec.var_offsets=valid; spec.rhs=NULL;
    REQUIRE(build_from_QAbc_soc(&p.Q,&p.A,p.c,p.AL,p.AU,p.l,p.u,0,&spec,&out)==HPRQP_STATUS_INVALID_ARGUMENT);
    spec.rhs=p.soc_rhs; p.l[2]=0;
    REQUIRE(build_from_QAbc_soc(&p.Q,&p.A,p.c,p.AL,p.AU,p.l,p.u,0,&spec,&out)==HPRQP_STATUS_INVALID_ARGUMENT);
    p.l[2]=-INFINITY; p.soc_rhs[0]=1;
    REQUIRE(build_from_QAbc_soc(&p.Q,&p.A,p.c,p.AL,p.AU,p.l,p.u,0,&spec,&out)==HPRQP_STATUS_INVALID_ARGUMENT);
    model_free(&out); model_free(&p);
    puts("SOC invalid metadata and ownership checks passed");
}

static void simple_solves()
{
    // Empty A, variable cone: Euclidean projection of (0,3,4).
    for (int scaling=0;scaling<2;++scaling) {
        HPRQPMatrixCSR Q=matrix({1,0,0,0,1,0,0,0,1},3,3), A=matrix({},0,3);
        double c[]={0,-3,-4}, l[]={-INFINITY,-INFINITY,-INFINITY}, u[]={INFINITY,INFINITY,INFINITY};
        size_t offsets[]={0,3}; HPRQPSOCSpec spec={1,offsets,0,NULL,NULL}; HPRQPModel p={};
        OK(build_from_QAbc_soc(&Q,&A,c,NULL,NULL,l,u,0,&spec,&p));
        HPRQPParams params; params_set_defaults(&params); params.verbose=0;params.print_frequency=0;
        params.use_scaling=scaling;params.stoptol=1e-9;params.max_iter=30000;
        HPRQPResults r={};REQUIRE(solve(&p,&params,&r)==HPRQP_STATUS_OPTIMAL);
        near(Vec(r.x,r.x+3),{2.5,1.5,2},2e-7);certify(p,r,2e-6);
        results_free(&r);model_free(&p);matrix_csr_free(&Q);matrix_csr_free(&A);
    }
    // Only an affine cone: min t, (t-2,3,4) in Q^3, optimum7.
    for (int scaling=0;scaling<2;++scaling) {
        HPRQPMatrixCSR Q=matrix({0},1,1), A=matrix({1,0,0},3,1);
        double c[]={1}, l[]={-INFINITY}, u[]={INFINITY}, al[]={-INFINITY,-INFINITY,-INFINITY},au[]={INFINITY,INFINITY,INFINITY};
        double rhs[]={2,-3,-4};size_t offsets[]={0,3};HPRQPSOCSpec spec={0,NULL,1,offsets,rhs};HPRQPModel p={};
        OK(build_from_QAbc_soc(&Q,&A,c,al,au,l,u,0,&spec,&p));REQUIRE(p.A.nrow==3);
        HPRQPParams params;params_set_defaults(&params);params.verbose=0;params.print_frequency=0;
        params.use_scaling=scaling;params.stoptol=1e-9;params.max_iter=100000;
        HPRQPResults r={}; REQUIRE(solve(&p,&params,&r)==HPRQP_STATUS_OPTIMAL);
        near(Vec(r.x,r.x+1),{7},2e-7);certify(p,r,2e-6);
        results_free(&r);model_free(&p);matrix_csr_free(&Q);matrix_csr_free(&A);
    }
    // Constant affine cone: preserve its feasibility even though A has no nonzeros.
    for (int feasible=0;feasible<2;++feasible) {
        HPRQPMatrixCSR Q=matrix({1},1,1), A=matrix({0,0},2,1);
        double c[]={-1},l[]={-INFINITY},u[]={INFINITY},al[]={-INFINITY,-INFINITY},au[]={INFINITY,INFINITY};
        double rhs[]={feasible?-3.0:1.0,0};size_t offsets[]={0,2};HPRQPSOCSpec spec={0,NULL,1,offsets,rhs};HPRQPModel p={};
        OK(build_from_QAbc_soc(&Q,&A,c,al,au,l,u,0,&spec,&p));REQUIRE(p.A.nrow==2);
        HPRQPParams params;params_set_defaults(&params);params.verbose=0;params.print_frequency=0;
        params.max_iter=1000;params.stoptol=1e-8;HPRQPResults r={};HPRQPStatus status=solve(&p,&params,&r);
        if (feasible) {REQUIRE(status==HPRQP_STATUS_OPTIMAL);near(Vec(r.x,r.x+1),{1},1e-7);certify(p,r,2e-6);}
        else {REQUIRE(status==HPRQP_STATUS_MAX_ITER);REQUIRE(r.residuals>=0.49);}
        results_free(&r);model_free(&p);matrix_csr_free(&Q);matrix_csr_free(&A);
    }
    puts("SOC six variable-only/affine-only/constant-row solves passed");
}

static void single_step()
{
    for (int q : {0, 2}) {
    HPRQPModel p=mixed_model(q);HPRQPParams params;params_set_defaults(&params);
    params.use_scaling=0;params.verbose=0;
    HPRQPGPUProblem gp={};HPRQPGPUWorkspace w={};HPRQPScaling sc={};
    OK(gpu_problem_upload(&gp,&p));OK(gpu_workspace_allocate(&w,&gp,&params));
    OK(gpu_problem_scale(&gp,&w,&params,&sc));OK(gpu_workspace_prepare_spmv(&w,&gp,0));OK(gpu_workspace_setup(&w,&gp,&params));
    REQUIRE((q == 0 || !w.Q_is_diag) && !w.noC);
    if (q == 0) {
        Vec poison(8, 123.0), zeros(8, 0.0), got(8);
        HPRQPGPUVector *products[]={&w.Qw,&w.Qw_bar,&w.Qx,&w.dQw,&w.QATdy};
        for(auto v:products) OK(gpu_vector_copy_from_host(v,poison.data(),8));
        OK(gpu_spmv_apply_q_w(&w,&gp));OK(gpu_spmv_apply_q_w_bar(&w,&gp));
        OK(gpu_spmv_apply_q_x_bar(&w,&gp));OK(gpu_spmv_apply_q_dw(&w,&gp));
        OK(gpu_spmv_apply_q_atdy(&w,&gp));
        for(auto v:products) {OK(gpu_vector_copy_to_host(v,got.data(),8));near(got,zeros,0);}
    }
    Vec x(8),oldw(8),y(7);
    for(size_t j=0;j<x.size();++j){x[j]=0.1*(j+1);oldw[j]=-0.03*(j+2);}
    for(size_t i=0;i<y.size();++i)y[i]=0.04*(i+1);
    Vec aty=product(p.AT,y),qw=product(p.Q,oldw),raw(8),xb(8),xh(8),wb1(8),temp(8),expectedz(8);
    const double sigma=w.sigma, f2=1/(1+sigma*w.lambda_max_Q),tau=sigma*w.lambda_max_A,a=1.0/6,b=1-a;
    for(size_t j=0;j<8;++j)raw[j]=x[j]+sigma*(-qw[j]+aty[j]-p.c[j]);
    xb=raw;for(size_t j=0;j<2;++j)xb[j]=std::min(std::max(raw[j],p.l[j]),p.u[j]);
    xb=projection(xb,2,5);xb=projection(xb,5,8);
    for(size_t j=0;j<8;++j){xh[j]=2*xb[j]-x[j];wb1[j]=(1-f2)*oldw[j]+f2*xh[j];expectedz[j]=(xb[j]-raw[j])/sigma;}
    Vec qwb=product(p.Q,wb1);
    for(size_t j=0;j<8;++j)temp[j]=xh[j]+sigma*(qw[j]-qwb[j]);
    Vec ax=product(p.A,temp),s(7),yb(7),yn(7);
    for(size_t i=0;i<7;++i)s[i]=ax[i]-p.soc_rhs[i]-tau*y[i];
    Vec rowraw=s;for(size_t i=0;i<2;++i)s[i]=std::min(std::max(s[i],p.AL[i]),p.AU[i]);
    s=projection(s,2,4);s=projection(s,4,7);
    for(size_t i=0;i<7;++i){yb[i]=(s[i]-rowraw[i])/tau;yn[i]=a*y[i]+b*(2*yb[i]-y[i]);}
    Vec atyb=product(p.AT,yb),wb(8),xn(8),wn(8);
    for(size_t j=0;j<8;++j){wb[j]=wb1[j]+sigma*f2*(atyb[j]-aty[j]);wn[j]=a*oldw[j]+b*(2*wb[j]-oldw[j]);xn[j]=a*x[j]+b*xh[j];}
    for(int full=0;full<2;++full)for(int custom=0;custom<2;++custom){
        w.to_check=full;w.spmv_mode_a=w.spmv_mode_q=custom?HPRQP_SPMV_MODE_CUSTOMIZED:HPRQP_SPMV_MODE_CUSPARSE;
        OK(gpu_vector_copy_from_host(&w.x,x.data(),8));OK(gpu_vector_copy_from_host(&w.last_x,x.data(),8));
        OK(gpu_vector_copy_from_host(&w.w,oldw.data(),8));OK(gpu_vector_copy_from_host(&w.last_w,oldw.data(),8));
        OK(gpu_vector_copy_from_host(&w.y,y.data(),7));OK(gpu_vector_copy_from_host(&w.last_y,y.data(),7));
        OK(gpu_vector_copy_from_host(&w.ATy,aty.data(),8));OK(gpu_vector_copy_from_host(&w.last_ATy,aty.data(),8));
        OK(gpu_iteration_step(&w,&gp,4));
        auto check=[&](HPRQPGPUVector& v,const Vec& e){Vec g(e.size());OK(gpu_vector_copy_to_host(&v,g.data(),g.size()));near(g,e,3e-13);};
        check(w.x_bar,xb);check(w.z_bar,expectedz);check(w.w_bar,wb);check(w.y_bar,yb);check(w.s,s);
        check(w.x,xn);check(w.y,yn);check(w.w,wn);
        HPRQPResiduals first={},second={};OK(gpu_compute_residuals_value(&w,&gp,&first,5));
        check(w.s,s);OK(gpu_compute_residuals_value(&w,&gp,&second,5));check(w.s,s);
        REQUIRE(first.KKTx_and_gap_org_bar==second.KKTx_and_gap_org_bar);
    }
    gpu_workspace_free(&w);gpu_problem_free(&gp);scaling_free(&sc);model_free(&p);
    }
    puts("SOC single-step host parity (zero/sparse Q), zero-Q scratch overwrite, both SpMV/check modes, and residual slack preservation passed");
}

static void scaling_contract()
{
    HPRQPModel p=mixed_model(2);HPRQPParams params;params_set_defaults(&params);params.verbose=0;
    HPRQPGPUProblem gp={};HPRQPGPUWorkspace w={};HPRQPScaling sc={};
    OK(gpu_problem_upload(&gp,&p));OK(gpu_workspace_allocate(&w,&gp,&params));
    OK(gpu_problem_scale(&gp,&w,&params,&sc));
    Vec rn(p.A.nrow),cn(p.Q.ncol),rhs(p.A.nrow),cv(p.Q.ncol);
    OK(gpu_vector_copy_to_host(&w.row_norm,rn.data(),rn.size()));
    OK(gpu_vector_copy_to_host(&w.col_norm,cn.data(),cn.size()));
    OK(gpu_vector_copy_to_host(&gp.soc_rhs,rhs.data(),rhs.size()));
    OK(gpu_vector_copy_to_host(&gp.c,cv.data(),cv.size()));
    for(size_t k=0;k<p.num_row_cones;++k)for(size_t i=p.row_cone_offsets[k];i<p.row_cone_offsets[k+1];++i)REQUIRE(rn[i]==rn[p.row_cone_offsets[k]]);
    for(size_t k=0;k<p.num_var_cones;++k)for(size_t j=p.var_cone_offsets[k];j<p.var_cone_offsets[k+1];++j)REQUIRE(cn[j]==cn[p.var_cone_offsets[k]]);
    for(size_t i=0;i<rn.size();++i){REQUIRE(rn[i]>0);REQUIRE(fabs(rhs[i]*rn[i]-p.soc_rhs[i])<2e-13*(1+fabs(p.soc_rhs[i])));}
    for(size_t j=0;j<cn.size();++j){REQUIRE(cn[j]>0);REQUIRE(fabs(cv[j]*cn[j]-p.c[j])<2e-13*(1+fabs(p.c[j])));}
    auto check_matrix=[&](const HPRQPMatrixCSR& before,const HPRQPGPUMatrixCSR& after,const Vec& rows,const Vec& cols){
        Vec values(before.nnz);REQUIRE(cudaMemcpy(values.data(),after.values,values.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
        for(size_t i=0;i<before.nrow;++i)for(size_t k=before.row_ptr[i];k<before.row_ptr[i+1];++k)
            REQUIRE(fabs(values[k]*rows[i]*cols[before.col_ind[k]]-before.values[k])<2e-13*(1+fabs(before.values[k])));
    };
    check_matrix(p.A,gp.A,rn,cn);check_matrix(p.AT,gp.AT,cn,rn);check_matrix(p.Q,gp.Q,cn,cn);
    REQUIRE(w.scale_norm_b_org>=infnorm(Vec(p.soc_rhs,p.soc_rhs+p.A.nrow)));
    gpu_workspace_free(&w);gpu_problem_free(&gp);scaling_free(&sc);model_free(&p);
    puts("SOC scaling: cone uniformity and A/AT/Q/c/offset round trips passed");
}

static void zero_cone_api()
{
    HPRQPMatrixCSR Q=matrix({1,0,0,2},2,2),A=matrix({1,1},1,2);
    double c[]={1,2},l[]={0,0},u[]={10,10},al[]={1},au[]={INFINITY};
    HPRQPModel old={};OK(build_from_QAbc(&Q,&A,c,al,au,l,u,0,&old));
    for(int null_spec=0;null_spec<2;++null_spec)for(int scaling=0;scaling<2;++scaling){
        HPRQPModel extended={};HPRQPSOCSpec spec={};
        OK(build_from_QAbc_soc(&Q,&A,c,al,au,l,u,0,null_spec?NULL:&spec,&extended));
        REQUIRE(extended.num_row_cones==0 && extended.num_var_cones==0);
        HPRQPParams params;params_set_defaults(&params);params.verbose=0;params.print_frequency=0;
        params.spmv_mode_a=params.spmv_mode_q=HPRQP_SPMV_MODE_CUSPARSE;params.use_scaling=scaling;
        HPRQPResults a={},b={};REQUIRE(solve(&old,&params,&a)==HPRQP_STATUS_OPTIMAL);
        REQUIRE(solve(&extended,&params,&b)==HPRQP_STATUS_OPTIMAL);
        REQUIRE(a.iter==b.iter && a.residuals==b.residuals && a.primal_obj==b.primal_obj);
        REQUIRE(memcmp(a.x,b.x,2*sizeof(double))==0 && memcmp(a.y,b.y,sizeof(double))==0);
        REQUIRE(memcmp(a.z,b.z,2*sizeof(double))==0 && memcmp(a.w,b.w,2*sizeof(double))==0);
        results_free(&a);results_free(&b);model_free(&extended);
    }
    model_free(&old);matrix_csr_free(&Q);matrix_csr_free(&A);
    puts("Zero-cone API: bitwise results match the existing QP builder (4 pairs)");
}

int main()
{
    REQUIRE(gpu_backend_available());
    projection_edges(); invalid_metadata(); single_step(); scaling_contract(); zero_cone_api(); simple_solves(); mixed_solves();
    puts("SOC GPU tests passed");
    return 0;
}
