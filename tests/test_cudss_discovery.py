import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

spec=importlib.util.spec_from_file_location('discovery',Path(__file__).resolve().parents[1]/'scripts/find_cudss.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
class Discovery(unittest.TestCase):
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
