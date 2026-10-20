import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import shutil

spec=importlib.util.spec_from_file_location('discovery',Path(__file__).resolve().parents[1]/'scripts/find_cudss.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
class Discovery(unittest.TestCase):
    def test_default_make_bootstraps_once_then_compiles_with_cudss(self):
        source=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='hprqp-build-') as tmp:
            p=Path(tmp);(p/'scripts').mkdir();(p/'cuda/bin').mkdir(parents=True)
            (p/'cuda/include').mkdir();(p/'cuda/include/cuda.h').touch()
            shutil.copy2(source/'Makefile',p/'Makefile')
            for name in ('find_cudss.py','write_gpu_config.py'):
                shutil.copy2(source/'scripts'/name,p/'scripts'/name)
            nvcc=p/'cuda/bin/nvcc'
            nvcc.write_text('#!/bin/sh\nprintf "Cuda compilation tools, release 13.0, V13.0\\n"\n')
            nvcc.chmod(0o755)
            # Isolate bootstrap orchestration from the separately tested real
            # NVIDIA download. The compiler stub fails on stale capability=0.
            (p/'scripts/setup_cudss.py').write_text('''\
from pathlib import Path
r=Path.cwd();calls=r/'install_calls'
calls.write_text((calls.read_text() if calls.exists() else '')+'install\\n')
d=r/'.deps/cudss/cuda13';(d/'include').mkdir(parents=True)
(d/'lib').mkdir();(d/'include/cudss.h').touch();(d/'lib/libcudss.so.0').touch()
'''.replace("'install\n'","'install\\n'"))
            compiler=p/'compiler.py'
            compiler.write_text('''\
#!/usr/bin/env python3
from pathlib import Path
import sys
assert '-DHPRQP_HAVE_CUDSS=1' in sys.argv, sys.argv
Path(sys.argv[sys.argv.index('-o')+1]).write_text('cudss=1')
''')
            compiler.chmod(0o755)
            (p/'src').mkdir()
            (p/'src/one.c').touch();(p/'src/two.c').touch()
            goals=['build-cuda/obj/src/one.o','build-cuda/obj/src/two.o']
            args=['make','--no-print-directory','-j2','-C',tmp,
                  'CUDA_ROOT='+str(p/'cuda'),'GPU_SM=90','CC='+str(compiler),*goals]
            subprocess.run(args,check=True,capture_output=True,text=True)
            self.assertEqual((p/'install_calls').read_text(),'install\n')
            for name in goals:self.assertEqual((p/name).read_text(),'cudss=1')
            subprocess.run(args,check=True,capture_output=True,text=True)
            self.assertEqual((p/'install_calls').read_text(),'install\n')

    def test_wheel_soname_without_linker_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'include').mkdir();(p/'include/cudss.h').touch()
            (p/'lib').mkdir();lib=p/'lib/libcudss.so.0';lib.touch()
            with patch.object(d.subprocess,'run',return_value=subprocess.CompletedProcess(
                    [],0,stdout='NEEDED libcudart.so.13')):
                self.assertEqual(d.discover([p],'13'),(str(p/'include'),str(lib)))
                self.assertEqual(d.discover([p],'12'),('',''))

    def test_make_rebuilds_after_capability_change(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/'gpu-config.json';obj=p/'stamp'
            script=root/'scripts/write_gpu_config.py'
            (p/'Makefile').write_text(f'''\
.PHONY: force
all: {obj}
{out}: force
\tpython3 "{script}" --output "{out}" --cudss_library "$(LIB)"
{obj}: {out}
\t@touch "{obj}"
''')
            def run(lib=''):
                subprocess.run(['make','--no-print-directory','-s','-C',tmp,'LIB='+lib],check=True)
                return obj.stat().st_mtime_ns
            first=run()
            self.assertEqual(run(),first)
            lib=p/'libcudss.so.0';lib.write_text('mock')
            second=run(str(lib));self.assertGreater(second,first)
            self.assertEqual(run(str(lib)),second)
            self.assertGreater(run(),second)

    def test_complete_package_and_major_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'include').mkdir();(p/'include/cudss.h').touch()
            self.assertEqual(d.discover([p],'13'),('',''))
            (p/'lib64').mkdir();lib=p/'lib64/libcudss.so';target=p/'lib64/libcudss.so.0.7.1';target.touch();lib.symlink_to(target.name)
            def output(major): return subprocess.CompletedProcess([],0,stdout=f'NEEDED libcudart.so.{major}')
            with patch.object(d.subprocess,'run',return_value=output(12)):
                self.assertEqual(d.discover([p],'13'),('',''))
            with patch.object(d.subprocess,'run',return_value=output(13)):
                self.assertEqual(d.discover([p],'13'),(str(p/'include'),str(lib)))
if __name__=='__main__':unittest.main()
