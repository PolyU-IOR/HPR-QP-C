"""Independent certificate for the unscaled native convex QP/row-SOC model.

Does not use SSN residuals, its projector/Jacobian, paired slack, or GPU state.
Dual objective uses actual interval support endpoints and SOC dual feasibility.
"""
import numpy as np


def infnorm(v):
    return float(np.max(np.abs(v), initial=0.))


def dot(a, b):
    return float(np.sum(np.asarray(a, dtype=np.longdouble)*b, dtype=np.longdouble))


def soc_project(v):
    t = float(v[0]); r = float(np.linalg.norm(v[1:]))
    if not np.isfinite(r):
        raise ValueError('cone norm overflow')
    if r <= t:
        return v.copy()
    if r <= -t:
        return np.zeros_like(v)
    out = np.empty_like(v); out[0] = .5*t+.5*r
    out[1:] = (.5+.5*t/r)*v[1:]
    return out


def row_projection(model, value):
    out = np.clip(value, model.row_lower, model.row_upper)
    for a,b in model.cones:
        out[a:b] = soc_project(value[a:b])
    return out


def interval_dual_projection(v, lower, upper):
    out = v.copy()
    # inf_{u in [lower,upper]} v'u is finite iff the coefficient has
    # the right sign at every infinite endpoint.
    out[np.isneginf(lower)] = np.minimum(out[np.isneginf(lower)], 0.)
    out[np.isposinf(upper)] = np.maximum(out[np.isposinf(upper)], 0.)
    return out


def prepare_candidate(model, proposed):
    """Explicit private dual-domain cleanup, followed by full independent check.

    This is candidate preparation, not a hidden modification by check(). No
    primal point is changed and no candidate is returned to HPR before check.
    """
    out = {k:np.array(v, dtype=float, copy=True) for k,v in proposed.items()}
    out['z'] = interval_dual_projection(out['z'], model.lower, model.upper)
    l = model.linear_m
    out['y'][:l] = interval_dual_projection(out['y'][:l], model.row_lower[:l], model.row_upper[:l])
    for a,b in model.cones:
        v = soc_project(out['y'][a:b])
        # Guard the serialized dual cone margin across BLAS/CPU reductions.
        # This changes the private candidate, never the acceptance tolerance.
        tail=np.asarray(v[1:],dtype=np.longdouble)
        r=np.sqrt(np.sum(tail*tail,dtype=np.longdouble))
        if r>0:
            margin=8*np.finfo(float).eps*len(tail)*r
            upper=np.nextafter(float(r+margin),np.inf)
            if v[0]<upper:v[0]=upper
        out['y'][a:b] = v
    return out


def interval_infimum(v, lower, upper):
    positive, negative = v > 0, v < 0
    if np.isneginf(lower[positive]).any() or np.isposinf(upper[negative]).any():
        return -np.inf
    return dot(v[positive], lower[positive])+dot(v[negative], upper[negative])


def check(model, candidate, tolerance=1e-8):
    """Three controlling categories: primal feasibility, dual feasibility, gap.

    Q must be symmetric PSD as supplied by the native convex-model builder.
    Numeric tolerance does not turn this into an exact-arithmetic proof.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance')
    for key,size in [('x',model.n),('w',model.n),('z',model.n),('y',model.m),('s',model.m)]:
        v = np.asarray(candidate[key])
        if v.shape != (size,) or not np.isfinite(v).all():
            raise ValueError('invalid '+key)
    x,w,z,y = (candidate[k] for k in ['x','w','z','y'])
    # All products are recomputed from the complete immutable model.
    qx, qw, ax, aty = model.Q@x, model.Q@w, model.A@x, model.A.T@y
    if not all(np.isfinite(v).all() for v in [qx,qw,ax,aty]):
        raise ValueError('matrix product overflow')
    bnorm = max(infnorm(model.rhs), infnorm(model.row_lower[np.isfinite(model.row_lower)]),
                infnorm(model.row_upper[np.isfinite(model.row_upper)]))
    row_error = infnorm(row_projection(model,ax-model.rhs)-(ax-model.rhs))/(1+max(bnorm,infnorm(ax)))
    bound_error = infnorm(x-np.clip(x,model.lower,model.upper))
    primal_error = max(row_error,bound_error)
    stationarity = infnorm(qw+model.c-aty-z)/(1+max(infnorm(model.c),infnorm(qw),infnorm(aty)))
    zprojected = interval_dual_projection(z,model.lower,model.upper)
    l = model.linear_m
    yprojected = y.copy()
    yprojected[:l] = interval_dual_projection(y[:l],model.row_lower[:l],model.row_upper[:l])
    cone_feasible = True
    for a,b in model.cones:
        yprojected[a:b] = soc_project(y[a:b])
        cone_feasible &= bool(y[a] >= np.linalg.norm(y[a+1:b]))
    domain_error = max(infnorm(z-zprojected)/(1+infnorm(z)),
                       infnorm(y-yprojected)/(1+infnorm(y)))
    z_support = interval_infimum(z,model.lower,model.upper)
    y_support = interval_infimum(y[:l],model.row_lower[:l],model.row_upper[:l])
    finite_dual_support = bool(cone_feasible and np.isfinite(z_support+y_support))
    # SOC infimum is zero for a dual-cone member; otherwise it is -infinity.
    primal = .5*dot(x,qx)+dot(model.c,x)+model.constant
    dual = (-.5*dot(w,qw)+z_support+y_support+dot(y,model.rhs)+model.constant
            if finite_dual_support else -np.inf)
    dual_error = max(stationarity,domain_error) if finite_dual_support else np.inf
    gap = abs(primal-dual)/(1+max(abs(primal),abs(dual))) if np.isfinite([primal,dual]).all() else np.inf
    maximum = max(primal_error,dual_error,gap)
    passed = bool(np.isfinite([maximum,primal,dual]).all() and maximum <= tolerance)
    return dict(passed=passed,tolerance=float(tolerance),primal_feasibility=primal_error,
                dual_feasibility=dual_error,relative_gap=gap,error=maximum,
                primal_objective=primal,dual_objective=dual,row_feasibility=row_error,
                variable_bounds=bound_error,stationarity=stationarity,dual_domain=domain_error,
                finite_dual_support=finite_dual_support,checker='independent_unscaled_native_cpu_v1')
