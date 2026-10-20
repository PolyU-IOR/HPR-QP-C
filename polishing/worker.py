#!/usr/bin/env python3
"""One isolated bounded episode, followed by an independent CPU certificate."""
import ctypes
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time


def main():
    start = time.monotonic()
    request_path = Path(sys.argv[1]).resolve()
    req = json.loads(request_path.read_text())
    ep = request_path.parent
    token = req['token']
    if len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
        raise ValueError('token')
    if os.getppid() != req['parent_pid']:
        raise RuntimeError('owner disappeared before startup')
    # Linux parent-death signal closes the orphan gap before Python numerical work.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'prctl')
    if os.getppid() != req['parent_pid']:
        raise RuntimeError('owner disappeared during startup')
    seconds = float(req['seconds']); memory = int(req['memory_bytes'])
    if not (math.isfinite(seconds) and seconds > 0 and (memory == 0 or memory >= 128*2**20)):
        raise ValueError('resource budget')
    if memory:
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    cpu_cap = max(1, math.ceil(seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_cap, cpu_cap+1))
    signal.setitimer(signal.ITIMER_REAL, max(1e-6, seconds-(time.monotonic()-start)))
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    from hpr_polish.core import Config
    from hpr_polish.transport import read_model, read_snapshot, write_candidate, read_candidate, digest, atomic_bytes
    from hpr_polish.cpu_check import prepare_candidate, check
    model_path, snapshot_path = Path(req['model']), ep/'snapshot.bin'
    model_hash, snapshot_hash = digest(model_path), digest(snapshot_path)
    for name, value in [('model_sha256', model_hash), ('snapshot_sha256', snapshot_hash)]:
        if name in req and req[name] != value:
            raise ValueError(name+' mismatch')
    model = read_model(model_path)
    for arr in [model.Q.data,model.Q.indices,model.Q.indptr,model.A.data,model.A.indices,model.A.indptr,
                model.c,model.lower,model.upper,model.row_lower,model.row_upper,model.rhs]:
        arr.setflags(write=False)
    if [model.n, model.m] != req['dimensions']:
        raise ValueError('request dimensions')
    u = read_snapshot(snapshot_path, model)
    options = req.get('config', {})
    if set(options)-{'backend', 'dense_cone_limit', 'direct_dimension', 'assembly_nnz'}:
        raise ValueError('unsupported configuration')
    auto = options.get('backend') == 'auto'
    if auto: options = dict(options, backend='generic')
    cfg = Config(**options, tolerance=float(req['tolerance']), seconds=max(1e-9, seconds-(time.monotonic()-start)))
    from hpr_polish.common_correction.episode import polish_common
    result = polish_common(model, u, cfg)
    candidate = result.pop('candidate')
    result.update(token=token, model_sha256=model_hash, snapshot_sha256=snapshot_hash,
                  snapshot_iteration=req['iteration'], process_s=time.monotonic()-start,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if candidate is not None and time.monotonic()-start < seconds:
        checking = time.monotonic()
        raw_candidate = candidate
        candidate = prepare_candidate(model,candidate)
        result['dual_cleanup_maxabs'] = max(float(abs(candidate[k]-raw_candidate[k]).max(initial=0.)) for k in ['y','z'])
        write_candidate(ep/'candidate.bin', model, candidate)
        # Check the serialized payload afresh, with no SSN cache or paired-gap reliance.
        verified = read_candidate(ep/'candidate.bin',model)
        certificate = check(model,verified,cfg.tolerance)
        result['independent_cpu_check'] = certificate
        result['cpu_check_s'] = time.monotonic()-checking
        result['candidate_sha256'] = digest(ep/'candidate.bin')
        result['independent_cpu_verified'] = certificate['passed']
        result['status'] = 'CPU_CERTIFIED' if certificate['passed'] else 'CPU_CHECK_REJECT'
        if certificate['passed']:
            packet = struct.pack('<8s16sQQ32s6d',b'HPRCPU01',bytes.fromhex(token),model.n,model.m,
                                 bytes.fromhex(result['candidate_sha256']),cfg.tolerance,
                                 certificate['primal_feasibility'],certificate['dual_feasibility'],
                                 certificate['relative_gap'],certificate['primal_objective'],certificate['dual_objective'])
            atomic_bytes(ep/'certificate.bin',packet)
        else:
            candidate = None
    else:
        candidate = None
    def finite_json(v):
        if type(v).__module__.startswith('numpy'): return finite_json(v.tolist())
        if isinstance(v, tuple): return [finite_json(x) for x in v]
        if isinstance(v, dict): return {k:finite_json(x) for k,x in v.items()}
        if isinstance(v, list): return [finite_json(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v): return None
        return v
    # Include independent checking, serialization, hashing and certificate I/O.
    # Final result/READY writes and process teardown remain in owner/runner wall time.
    result['process_s'] = time.monotonic()-start
    result['process_s_scope'] = 'startup through CPU certificate, before result and READY publication'
    result['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    atomic_bytes(ep/'result.json', (json.dumps(finite_json(result), indent=2, allow_nan=False)+'\n').encode())
    # Publication is last, after complete paired data and diagnostics are durable.
    if candidate is not None and time.monotonic()-start < seconds:
        atomic_bytes(ep/'READY', ('HPRCPU01\n'+token+'\n').encode())
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        # No READY can be produced on a failed episode.
        print(type(exc).__name__+': '+str(exc), file=sys.stderr)
        sys.exit(2)
