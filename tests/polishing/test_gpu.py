"""Small live GPU contract tests; never run a benchmark corpus.

Usage: PYTHONPATH=polishing python3 tests/polishing/test_gpu.py DRIVER OUT
Select an available device with CUDA_VISIBLE_DEVICES before invocation.
"""
import json, os, pathlib, subprocess, sys, time
import numpy as np
from fixtures import qp, spatial
from hpr_polish.transport import write_model, read_candidate
from hpr_polish.cpu_check import check

exe=pathlib.Path(sys.argv[1]).resolve();root=pathlib.Path(sys.argv[2]).resolve();root.mkdir(exist_ok=False)
worker=pathlib.Path(__file__).resolve().parents[2]/'polishing/worker.py'
records=[];models={}
for kind in ('affine','mixed','dependent','no_equalities','box'):
    models[kind]=qp(kind)
for g in (3,31,63):models['spatial'+str(g)]=(spatial(g),None)
for name,(model,_) in models.items():write_model(root/(name+'.bin'),model)

def trace_audit(log):
    events=[json.loads(s.split('HPRQP_GRAM_TRACE ',1)[1]) for s in log.splitlines() if s.startswith('HPRQP_GRAM_TRACE ')]
    setup=next((s for s in events if s['stage']=='setup'),None)
    if setup is None:return 0
    e=setup.get('e',setup.get('l'));g=np.array(setup['gram']).reshape(e,e);eq=setup.get('eq',list(range(e)))
    rest=[i for i in range(setup['m']) if i not in eq];beta=setup['beta'];defects=[]
    for s in events:
        if s['stage']=='row':
            step=np.linalg.solve(g,np.array(s['rhs'],dtype=float));actual=np.array(s['step'],dtype=float)
            defects.append(np.linalg.norm(actual-step,np.inf)/(1+np.linalg.norm(step,np.inf)))
            y=np.array(s['old_y'],dtype=float);expected=y.copy();expected[eq]-=step/(2*s['sigma'])
            expected[rest]=(np.array(s['slack'])[rest]-np.array(s['raw'])[rest])/(2*s['sigma']*beta)
            defects.append(np.linalg.norm(np.array(s['yb'])-expected,np.inf)/(1+np.linalg.norm(expected,np.inf)))
        if s['stage']=='quadratic':
            y=np.array(s['dy'],dtype=float);v=2*y[eq]@g@y[eq]+2*beta*(y[rest]@y[rest]);defects.append(abs(s['value']-v)/(1+abs(v)))
    assert defects and max(defects)<1e-10,defects
    return max(defects)

def run(name,gram,polish=0,override=None,cap=None):
    key=f'{name}_g{gram}_p{polish}'+('_'+override.stem if override else '')
    out=root/key;out.mkdir();args=[str(exe),str(root/(name+'.bin')),str(out),str(gram),str(polish),sys.executable,str(override or worker)]
    if cap is not None:args.append(str(cap))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',HPRQP_GRAM_TRACE='1')
    tick=time.monotonic();p=subprocess.run(args,capture_output=True,text=True,env=env,timeout=90)
    (out/'stdout.txt').write_text(p.stdout);(out/'stderr.txt').write_text(p.stderr)
    lines=[s for s in p.stdout.splitlines() if s.startswith('{')];assert lines,(p.returncode,p.stderr)
    record=json.loads(lines[-1]);record.update(name=key,process_seconds=time.monotonic()-tick,returncode=p.returncode)
    (out/'result.json').write_text(json.dumps(record,indent=2));records.append(record)
    (root/'PARTIAL.json').write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True)
    assert p.returncode==0,record
    record['trace_defect']=trace_audit(p.stderr)
    if cap is None:
        assert record['status']==6 and 0<=record['error']<=1e-8,record
        optimum=models[name][1]
        if optimum is not None:assert abs(record['objective']-optimum)<1e-7,record
    if record['accepted']:
        episodes=list(out.glob('hprqp-polish-*/episode_*/READY'));assert len(episodes)==1
        ep=episodes[0].parent;c=read_candidate(ep/'candidate.bin',models[name][0]);audit=check(models[name][0],c);assert audit['passed'],audit
        raw=(out/'solution.bin').read_bytes();n=models[name][0].n;m=models[name][0].m
        returned=np.frombuffer(raw[40:],dtype='<f8');assert np.array_equal(returned,np.r_[c['x'],c['y'],c['z'],c['w']])
    return record,out

for name in ('affine','mixed','dependent','no_equalities','box'):
    off,od=run(name,0);on,nd=run(name,2);assert on['gram']==(name in ('affine','mixed'))
    if not on['gram']:assert (od/'solution.bin').read_bytes()==(nd/'solution.bin').read_bytes()
    if name=='mixed':
        auto,ad=run(name,1);assert not auto['gram'];assert (od/'solution.bin').read_bytes()==(ad/'solution.bin').read_bytes()
run('spatial3',0);s,_=run('spatial3',1);assert s['gram']
live,_=run('spatial63',1,1);assert live['gram'] and live['attempts']>0,live
# HPR may legitimately win the race. Accepted payloads are checked in run().
# Failed workers must never change a fixed-length GPU trajectory.
crash=root/'crash.py';crash.write_text('raise RuntimeError("injected worker failure")\n')
base,bd=run('spatial31',0,0,cap=3000);failed,fd=run('spatial31',0,1,override=crash,cap=3000)
assert failed['attempts']>0 and not failed['accepted']
assert base['iterations']==failed['iterations'] and (bd/'solution.bin').read_bytes()==(fd/'solution.bin').read_bytes()
sleeper=root/'sleeper.py';sleeper.write_text('import os,sys,time,pathlib\npathlib.Path(sys.argv[1]).with_name("pid").write_text(str(os.getpid()))\ntime.sleep(60)\n')
cancelled,cd=run('spatial31',0,1,override=sleeper,cap=3000)
assert cancelled['attempts']>0 and not cancelled['accepted']
assert (bd/'solution.bin').read_bytes()==(cd/'solution.bin').read_bytes()
for pidfile in cd.glob('hprqp-polish-*/episode_*/pid'):
    try:os.kill(int(pidfile.read_text()),0)
    except ProcessLookupError:pass
    else:raise AssertionError('worker survived owner completion')
(root/'COMPLETE.json').write_text(json.dumps(dict(passed=True,runs=records),indent=2))
