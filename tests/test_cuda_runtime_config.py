"""Driver selection and real ELF/RPATH preflight tests without CUDA hardware."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import configure_cuda_runtime as runtime


def result(code=0, count=1):
    return dict(code=code, device_count=count, message="test", driver_library="test")


class Selection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "compat").mkdir()
        (self.root / "compat/libcuda.so.1").touch()

    def test_system_preferred_even_when_compat_exists(self):
        probe = Mock(return_value=result())
        self.assertEqual(runtime.select(self.root, self.root, "cc", probe)["status"], "system")
        self.assertEqual(probe.call_count, 1)

    def test_insufficient_driver_uses_only_verified_compat(self):
        probe = Mock(side_effect=[result(35, 0), result()])
        actual = runtime.select(self.root, self.root, "cc", probe)
        self.assertEqual(actual["status"], "compat")
        self.assertEqual(actual["driver_dir"], str(self.root / "compat"))
        self.assertEqual(probe.call_count, 2)

    def test_failed_or_hidden_compat_never_qualifies(self):
        for alternate in (result(803, 0), result(100, 0), result(0, 0)):
            with self.subTest(alternate=alternate):
                actual = runtime.select(self.root, self.root, "cc",
                                        Mock(side_effect=[result(35, 0), alternate]))
                self.assertEqual(actual["status"], "failed")
                self.assertEqual(actual["driver_dir"], "")

    def test_missing_compat_fails(self):
        (self.root / "compat/libcuda.so.1").unlink()
        probe = Mock(return_value=result(35, 0))
        self.assertEqual(runtime.select(self.root, self.root, "cc", probe)["status"], "failed")
        self.assertEqual(probe.call_count, 1)

    def test_no_device_crossbuild_is_explicitly_unverified(self):
        for baseline in (result(100, 0), result(0, 0)):
            probe = Mock(return_value=baseline)
            actual = runtime.select(self.root, self.root, "cc", probe)
            self.assertEqual(actual["status"], "unverified_no_device")
            self.assertEqual(actual["driver_dir"], "")
            self.assertEqual(probe.call_count, 1)

    def test_unchanged_include_keeps_timestamp(self):
        path = self.root / "config.mk"
        runtime.write_changed(path, "fixed\n")
        stamp = path.stat().st_mtime_ns
        runtime.write_changed(path, "fixed\n")
        self.assertEqual(path.stat().st_mtime_ns, stamp)


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("cc")
                     and shutil.which("make"), "ELF tests need cc and make")
class NativeProbe(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.cuda = self.work / "cuda"
        for name in ("include", "lib64", "compat", "bin"):
            (self.cuda / name).mkdir(parents=True)
        (self.cuda / "include/cuda.h").touch()
        (self.cuda / "include/cuda_runtime_api.h").write_text(
            "typedef int cudaError_t;\nint cudaGetDeviceCount(int*);\n"
            "const char *cudaGetErrorString(int);\n")
        # Unique dlopen name keeps this test independent of actual installed drivers.
        (self.work / "runtime.c").write_text(
            '#include <dlfcn.h>\n'
            'int cudaGetDeviceCount(int *n) {void *d=dlopen("libhpr_test_driver.so.1",2);'
            '*n=d ? 2:0; return d ? 0:35;}\n'
            'const char *cudaGetErrorString(int e){return e ? "insufficient driver":"success";}\n')
        subprocess.run(["cc", "-shared", "-fPIC", str(self.work / "runtime.c"),
                        "-Wl,-soname,libcudart.so", "-ldl",
                        "-o", str(self.cuda / "lib64/libcudart.so")], check=True)
        (self.work / "driver.c").write_text("int cuInit(int n) {return 0;}\n")
        subprocess.run(["cc", "-shared", "-fPIC", str(self.work / "driver.c"),
                        "-Wl,-soname,libcuda.so.1",
                        "-o", str(self.cuda / "compat/libhpr_test_driver.so.1")], check=True)
        (self.cuda / "compat/libcuda.so.1").symlink_to("libhpr_test_driver.so.1")
        nvcc = self.cuda / "bin/nvcc"
        nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.3'\n")
        nvcc.chmod(0o755)
        shutil.copy(ROOT / "Makefile", self.work / "Makefile")
        shutil.copytree(ROOT / "scripts", self.work / "scripts",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (self.work / "extra.mk").write_text(
            'ifeq ($(HPRQP_GPU_RUNTIME_CHECKED),1)\n'
            'build-cuda/preflight_smoke:\n\t@echo \'$(GPU_LINK_LIBS)\' > link-flags.txt\nendif\n')
        # Recursive Make also needs the isolated test target.
        with (self.work / "Makefile").open("a") as f:
            f.write("\ninclude extra.mk\n")

    def make(self, *args):
        return subprocess.run(["make", "--no-print-directory",
                               "build-cuda/preflight_smoke", "CUDA_ROOT=" + str(self.cuda),
                               "REQUIRE_CUDSS=0", "PYTHON=" + sys.executable, *args],
                              cwd=self.work, text=True, capture_output=True, timeout=30)

    def test_rpath_native_probe_and_public_make_reentry(self):
        # Shell compatibility must not mask the failing default probe.
        old = os.environ.get("LD_LIBRARY_PATH")
        os.environ["LD_LIBRARY_PATH"] = str(self.cuda / "compat")
        try:
            actual = runtime.select(self.cuda, self.cuda / "lib64", "cc")
        finally:
            if old is None:
                os.environ.pop("LD_LIBRARY_PATH", None)
            else:
                os.environ["LD_LIBRARY_PATH"] = old
        self.assertEqual(actual["system_probe"]["code"], 35)
        self.assertEqual(actual["status"], "compat")
        run = self.make()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn(str(self.cuda / "compat"), (self.work / "link-flags.txt").read_text())
        self.assertIn("--no-as-needed", (self.work / "link-flags.txt").read_text())
        before = (self.work / ".deps/cuda-runtime.mk").stat().st_mtime_ns
        run = self.make()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(before, (self.work / ".deps/cuda-runtime.mk").stat().st_mtime_ns)

    def test_failed_preflight_stops_make_before_targets(self):
        (self.cuda / "compat/libhpr_test_driver.so.1").unlink()
        run = self.make()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("CUDA startup validation failed", run.stderr)
        self.assertFalse((self.work / "link-flags.txt").exists())
        self.assertEqual(json.loads((self.work / ".deps/cuda-runtime.json").read_text())["status"],
                         "failed")

    def test_compile_only_is_explicit_and_does_not_embed_unverified_driver(self):
        run = self.make("CUDA_DRIVER_CHECK=0")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("unverified_skipped", run.stderr)
        self.assertNotIn(str(self.cuda / "compat"), (self.work / "link-flags.txt").read_text())

    def test_mixed_compiler_and_toolkit_fail_even_in_compile_only_mode(self):
        run = self.make("CUDA_NVCC=/another/cuda/bin/nvcc", "CUDA_DRIVER_CHECK=0")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Mixed CUDA toolchain", run.stderr)
        self.assertFalse((self.work / "link-flags.txt").exists())


if __name__ == "__main__":
    unittest.main()
