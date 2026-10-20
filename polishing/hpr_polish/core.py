"""General sparse-Q, interval/row-SOC projected KKT polishing.

No GPU state is accessed. Internal certificates are preliminary; the independent
CPU support-objective checker is authoritative before owner publication.
"""
from dataclasses import dataclass
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la


def maxabs(v):
    return float(np.max(np.abs(v), initial=0.))


@dataclass
class Model:
    Q: sp.csr_matrix
    A: sp.csr_matrix
    c: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    row_lower: np.ndarray
    row_upper: np.ndarray
    rhs: np.ndarray
    cones: tuple = ()
    constant: float = 0.

    def __post_init__(self):
        self.Q = sp.csr_matrix(self.Q, dtype=float, copy=True)
        self.A = sp.csr_matrix(self.A, dtype=float, copy=True)
        self.n = self.A.shape[1]
        self.m = self.A.shape[0]
        if self.Q.shape != (self.n, self.n):
            raise ValueError("Q shape")
        for name, size in [("c", self.n), ("lower", self.n), ("upper", self.n),
                           ("row_lower", self.m), ("row_upper", self.m), ("rhs", self.m)]:
            v = np.array(getattr(self, name), dtype=float, copy=True)
            if v.shape != (size,) or np.isnan(v).any():
                raise ValueError(name)
            setattr(self, name, v)
        if not all(np.isfinite(v).all() for v in [self.Q.data, self.A.data, self.c, self.rhs]):
            raise ValueError("nonfinite model")
        if not np.isfinite(self.constant):
            raise ValueError("constant")
        if np.any(self.lower > self.upper) or np.any(self.row_lower > self.row_upper):
            raise ValueError("inconsistent bounds")
        if np.isposinf(self.lower).any() or np.isneginf(self.upper).any():
            raise ValueError("empty variable domain")
        if maxabs((self.Q-self.Q.T).data) > 1e-13*max(1., maxabs(self.Q.data)):
            raise ValueError("Q must be symmetric; convexity is a caller precondition")
        self.cones = tuple((int(a), int(b)) for a, b in self.cones)
        self.linear_m = self.cones[0][0] if self.cones else self.m
        if not 0 <= self.linear_m <= self.m:
            raise ValueError("cone start")
        last = self.linear_m
        for a, b in self.cones:
            if a != last or b-a < 2 or b > self.m:
                raise ValueError("cones must form a contiguous suffix")
            if not (np.isneginf(self.row_lower[a:b]).all() and np.isposinf(self.row_upper[a:b]).all()):
                raise ValueError("cone rows cannot also have interval bounds")
            last = b
        if last != self.m:
            raise ValueError("cone suffix")
        if np.isposinf(self.row_lower[:self.linear_m]).any() or np.isneginf(self.row_upper[:self.linear_m]).any():
            raise ValueError("empty row domain")


class RowDerivative:
    """Self-adjoint projector derivative; large SOC blocks stay implicit."""
    def __init__(self, diagonal, blocks, size):
        self.diagonal, self.blocks, self.size = diagonal, blocks, size

    def action(self, d):
        out = np.zeros_like(d, dtype=float)
        out[:len(self.diagonal)] = self.diagonal*d[:len(self.diagonal)]
        for a, b, kind, q, tr in self.blocks:
            if kind == "identity":
                out[a:b] = d[a:b]
            elif kind == "apex":
                out[a:b] = .5*d[a:b]
            elif kind == "middle":
                dv = d[a+1:b]
                qv = float(q@dv)
                out[a] = .5*(d[a]+qv)
                out[a+1:b] = .5*(q*d[a]+(1+tr)*dv-tr*q*qv)
        return out

    def sparse(self, maximum_cone=64):
        parts = [sp.diags(self.diagonal, format="csr")]
        for a, b, kind, q, tr in self.blocks:
            d = b-a
            if d > maximum_cone:
                return None
            if kind in ("identity", "apex"):
                parts.append(sp.eye(d, format="csr")*(1 if kind == "identity" else .5))
            elif kind == "zero":
                parts.append(sp.csr_matrix((d, d)))
            else:
                v = np.empty((d, d))
                v[0, 0] = .5
                v[0, 1:] = v[1:, 0] = .5*q
                v[1:, 1:] = .5*((1+tr)*np.eye(d-1)-tr*np.outer(q, q))
                parts.append(sp.csr_matrix(v))
        return sp.block_diag(parts, format="csr")


def project_rows(model, v):
    p = np.clip(v, model.row_lower, model.row_upper)
    l = model.linear_m
    diagonal = ((v[:l] > model.row_lower[:l]) & (v[:l] < model.row_upper[:l])).astype(float)
    blocks = []
    for a, b in model.cones:
        t, z = v[a], v[a+1:b]
        r = float(la.norm(z))
        q = None
        tr = 0.
        if r == 0 and t == 0:
            p[a:b] = 0.
            kind = "apex"
        elif r <= t:
            p[a:b] = v[a:b]
            kind = "identity"
        elif r <= -t:
            p[a:b] = 0.
            kind = "zero"
        else:
            q, tr = z/r, t/r
            p[a] = .5*(t+r)
            p[a+1:b] = .5*(1+tr)*z
            kind = "middle"
        blocks.append((a, b, kind, q, tr))
    return p, RowDerivative(diagonal, blocks, model.m)


def evaluate(model, u, alpha=1., beta=1.):
    if alpha <= 0 or beta <= 0 or not np.isfinite([alpha, beta]).all():
        raise ValueError("residual scales")
    if u.shape != (model.n+model.m,) or not np.isfinite(u).all():
        raise ValueError("private iterate")
    x, y = u[:model.n], u[model.n:]
    g = model.Q@x+model.c-model.A.T@y
    raw = x-alpha*g
    p = np.clip(raw, model.lower, model.upper)
    ax = model.A@x-model.rhs
    s, S = project_rows(model, ax-beta*y)
    F = np.r_[x-p, ax-s]
    E = ((raw > model.lower) & (raw < model.upper)).astype(float)
    if not np.isfinite(F).all():
        raise FloatingPointError("nonfinite projected KKT residual")
    return dict(F=F, p=p, s=s, E=E, S=S, g=g, norm=float(la.norm(F)))


def jacobian(model, e, alpha=1., beta=1.):
    n, m, E, S = model.n, model.m, e["E"], e["S"]

    def mv(d):
        dx, dy = d[:n], d[n:]
        adx = model.A@dx
        return np.r_[(1-E)*dx+alpha*E*(model.Q@dx-model.A.T@dy),
                     adx-S.action(adx)+beta*S.action(dy)]

    def rmv(d):
        vx, vy = d[:n], d[n:]
        ev, sy = E*vx, S.action(vy)
        return np.r_[(1-E)*vx+alpha*(model.Q.T@ev)+model.A.T@(vy-sy),
                     -alpha*(model.A@ev)+beta*sy]

    return sla.LinearOperator((n+m, n+m), matvec=mv, rmatvec=rmv, dtype=float)


def sparse_jacobian(model, e, config):
    estimate = model.Q.nnz+2*model.A.nnz+model.n+model.m
    for a, b in model.cones:
        estimate += (b-a)*(model.A[a:b].nnz+b-a)
    if model.n+model.m > config.direct_dimension or estimate > config.assembly_nnz:
        return None
    S = e["S"].sparse(config.dense_cone_limit)
    if S is None:
        return None
    E = sp.diags(e["E"], format="csr")
    J = sp.bmat([[sp.eye(model.n)-E+config.alpha*E@model.Q, -config.alpha*E@model.A.T],
                 [(sp.eye(model.m)-S)@model.A, config.beta*S]], format="csc")
    if J.nnz > config.assembly_nnz:
        return None
    return J


def candidate(model, u, e, alpha=1., beta=1.):
    # Projection-normal reconstruction, not a partial HPR state handback.
    rx, ry = e["F"][:model.n], e["F"][model.n:]
    return dict(x=e["p"].copy(), w=e["p"].copy(),
                z=e["g"]-rx/alpha, y=u[model.n:]-ry/beta, s=e["s"].copy())


def certificate(model, c):
    """Conservative CPU native precheck. Production GPU acceptance is external."""
    if not all(np.isfinite(v).all() for v in c.values()):
        return dict(error=float("inf"))
    x, w, y, z, s = (c[k] for k in ("x", "w", "y", "z", "s"))
    qx, qw, ax, aty = model.Q@x, model.Q@w, model.A@x, model.A.T@y
    bounds = np.r_[model.row_lower[np.isfinite(model.row_lower)],
                   model.row_upper[np.isfinite(model.row_upper)], model.rhs]
    rp = maxabs(project_rows(model, ax-model.rhs)[0]-(ax-model.rhs))/(1+max(maxabs(bounds), maxabs(ax)))
    rd = maxabs(qx+model.c-aty-z)/(1+max(maxabs(model.c), maxabs(qx), maxabs(aty)))
    primal = float(.5*x@qx+model.c@x+model.constant)
    dual = float(-.5*w@qw+z@x+y@(s+model.rhs)+model.constant)
    gap = abs(primal-dual)/(1+max(abs(primal), abs(dual)))
    box = maxabs(x-np.clip(x, model.lower, model.upper))
    normal = maxabs(x-np.clip(x-z, model.lower, model.upper))/(1+maxabs(x))
    row_normal = maxabs(s-project_rows(model, s-y)[0])/(1+maxabs(s))
    qw_match = maxabs(qw-qx)/(1+maxabs(qx))
    vals = [rp, rd, gap, box, normal, row_normal, qw_match]
    return dict(error=max(vals) if np.isfinite(vals+[primal, dual]).all() else float("inf"),
                rp=rp, rd=rd, gap=gap, box=box, normal=normal,
                row_normal=row_normal, primal=primal, dual=dual)


@dataclass
class Config:
    tolerance: float = 1e-8
    seconds: float = 20.
    alpha: float = 1.
    beta: float = 1.
    direct_dimension: int = 20000
    assembly_nnz: int = 2_000_000
    dense_cone_limit: int = 64
    backend: str = "generic"

    def __post_init__(self):
        positive = [self.tolerance, self.seconds, self.alpha, self.beta]
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("positive finite solver parameters required")
        for name in ["direct_dimension", "assembly_nnz", "dense_cone_limit"]:
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(name)
        if self.backend not in ("generic", "banded"):
            raise ValueError("backend")
