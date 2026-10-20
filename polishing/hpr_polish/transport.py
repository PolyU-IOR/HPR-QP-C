"""Versioned little-endian native model/snapshot transport; no pickle loading."""
import hashlib
import os
import struct
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from .core import Model


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    with open(tmp, 'xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


class Reader:
    def __init__(self, path):
        self.f = open(path, 'rb'); self.size = os.fstat(self.f.fileno()).st_size

    def array(self, count, dtype='<f8'):
        dtype = np.dtype(dtype)
        if count < 0 or count*dtype.itemsize > self.size-self.f.tell():
            raise ValueError('truncated/invalid array length')
        out = np.fromfile(self.f, dtype=dtype, count=count)
        if out.size != count:
            raise ValueError('truncated array')
        return out

    def magic(self, expected):
        if self.f.read(8) != expected:
            raise ValueError('magic')

    def matrix(self):
        nr, nc, nz = map(int, self.array(3, '<u8'))
        if max(nr, nc, nz) > 2**31-1:
            raise ValueError('index range')
        p, j = self.array(nr+1, '<u8'), self.array(nz, '<u8')
        v = self.array(nz)
        if p[0] != 0 or p[-1] != nz or np.any(p[1:] < p[:-1]) or np.any(j >= nc):
            raise ValueError('invalid CSR')
        a = sp.csr_matrix((v, j.astype(np.int64), p.astype(np.int64)), shape=(nr, nc))
        a.sum_duplicates(); a.sort_indices()
        return a

    def end(self):
        if self.f.read(1):
            raise ValueError('trailing bytes')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.f.close()


def write_model(path, model):
    with open(path, 'xb') as f:
        f.write(b'HPRNAT01')
        for a in [model.Q, model.A]:
            f.write(struct.pack('<QQQ', *a.shape, a.nnz))
            for v, typ in [(a.indptr, '<u8'), (a.indices, '<u8'), (a.data, '<f8')]:
                f.write(np.asarray(v, dtype=typ).tobytes())
        for v in [model.c, model.lower, model.upper, model.row_lower, model.row_upper, model.rhs]:
            f.write(np.asarray(v, dtype='<f8').tobytes())
        f.write(struct.pack('<dQ', model.constant, len(model.cones)))
        if model.cones:
            f.write(np.asarray([model.cones[0][0]]+[b for a,b in model.cones], dtype='<u8').tobytes())


def read_model(path):
    with Reader(path) as f:
        f.magic(b'HPRNAT01'); Q, A = f.matrix(), f.matrix(); m, n = A.shape
        c, lo, hi = (f.array(n) for _ in range(3))
        rl, ru, rhs = (f.array(m) for _ in range(3))
        constant = float(f.array(1)[0]); count = int(f.array(1, '<u8')[0])
        if count > m//2:
            raise ValueError('cone count')
        offsets = f.array(count+1, '<u8') if count else []
        f.end()
    return Model(Q, A, c, lo, hi, rl, ru, rhs, tuple(zip(offsets[:-1], offsets[1:])), constant)


def write_snapshot(path, model, u):
    with open(path, 'xb') as f:
        f.write(b'HPRSOLS1'); f.write(struct.pack('<QQQQ', model.n, model.m, model.n, model.n))
        for v in [u[:model.n], u[model.n:], np.zeros(model.n), u[:model.n]]:
            f.write(np.asarray(v, dtype='<f8').tobytes())


def read_snapshot(path, model):
    with Reader(path) as f:
        f.magic(b'HPRSOLS1')
        if tuple(f.array(4, '<u8')) != (model.n, model.m, model.n, model.n):
            raise ValueError('snapshot dimensions')
        x, y, z, w = f.array(model.n), f.array(model.m), f.array(model.n), f.array(model.n)
        f.end()
    if not all(np.isfinite(a).all() for a in [x,y,z,w]):
        raise ValueError('nonfinite snapshot')
    return np.r_[x,y]


def write_candidate(path, model, candidate):
    data = b'HPRPOL01'+struct.pack('<QQ', model.n, model.m)
    data += b''.join(np.asarray(candidate[k], dtype='<f8').tobytes() for k in ['x','w','z','y','s'])
    atomic_bytes(path, data)


def read_candidate(path, model):
    with Reader(path) as f:
        f.magic(b'HPRPOL01')
        if tuple(f.array(2, '<u8')) != (model.n, model.m):
            raise ValueError('candidate dimensions')
        out = {k:f.array(n) for k,n in [('x',model.n),('w',model.n),('z',model.n),('y',model.m),('s',model.m)]}
        f.end()
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError('nonfinite candidate')
    return out
