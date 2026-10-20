"""Structural dispatch for private corrections; cpu_check controls acceptance.

No filename or benchmark name is used for selection.
"""
from pathlib import Path
import time
import numpy as np
import scipy.sparse as sp
from .transport import Reader




def full_snapshot(path, model):
    with Reader(path) as f:
        f.magic(b'HPRSOLS1')
        assert tuple(f.array(4, '<u8')) == (model.n, model.m, model.n, model.n)
        x, y, z, w = f.array(model.n), f.array(model.m), f.array(model.n), f.array(model.n)
        f.end()
    assert all(np.isfinite(v).all() for v in (x,y,z,w))
    return x,y,z,w


def spatial_module():
    from .numerics import spatial
    return spatial


def as_legacy(model, native):
    l = int(model.cones[0][0]) if model.cones else model.m
    return native.Model(model.Q.diagonal(),model.A,model.c,model.lower,model.upper,
        model.row_lower[:l],model.row_upper[:l],model.rhs,l,
        np.array([l]+[b for a,b in model.cones], dtype=int),model.constant,'native')


def selected(model):
    """Cheap model gates before a private numerical episode.

    Spatial pivot checks the exact tensor matrix, not a square row count alone.
    The banded episode additionally verifies actual Jacobian bandwidth/gauges.
    """
    delta=model.Q-sp.diags(model.Q.diagonal());delta.eliminate_zeros()
    if delta.nnz or np.any(model.Q.diagonal()<0): return 'generic',None,None
    if not model.cones:
        eq=np.array_equal(model.row_lower,model.row_upper) and np.isfinite(model.row_lower).all()
        if eq and not np.any(model.rhs) and model.n+model.m<=400000 and model.Q.nnz+model.A.nnz<=1500000:
            return 'equality_face',None,None
        return 'generic',None,None
    if any(b-a!=3 for a,b in model.cones): return 'generic',None,None
    ex=spatial_module()
    from .numerics import native
    old=as_legacy(model,native);l=old.linear_m;g=int(round(np.sqrt(l)))
    if l and g*g==l and np.array_equal(old.al,old.au) and np.isfinite(old.al).all():
        try:
            I=ex.state_indices(old)
            from .numerics.poisson import tensor
            diff=old.A[:l,I]-tensor(g);diff.eliminate_zeros()
            if not diff.nnz and np.all(old.q[I]>0): return 'spatial',old,(ex,g)
        except (AssertionError,ValueError,IndexError): pass
    # The band adapter below explicitly binds its own resource-guarded native.
    return 'banded_native',old,None
