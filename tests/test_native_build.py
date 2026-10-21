"""Direct native build: ABI discovery, source parity and real incremental builds."""
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_build", ROOT / "scripts/build_native.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class SourceContract(unittest.TestCase):
    def test_superlu_sources_match_qualified_generic_and_double_build(self):
        text = (ROOT / "third_party/superlu/SRC/CMakeLists.txt").read_text()
        generic = re.search(r"set\(sources\s+(.*?)\)", text, re.S).group(1).split()
        double = re.search(r"if\(enable_double\).*?list\(APPEND sources\s+(.*?)\)",
                           text, re.S).group(1).split()
        self.assertEqual(tuple(generic), native.SUPERLU_GENERIC)
        self.assertEqual(tuple(double), native.SUPERLU_DOUBLE)
        config = (ROOT / "third_party/superlu/SRC/superlu_config.h").read_text()
        self.assertNotRegex(config, r"(?m)^\s*#define\s+(HAVE_METIS|XSDK_INDEX_SIZE)\b")
        self.assertEqual(len(set(generic + double)), len(generic + double))

    def test_native_sources_and_optimization_match_qualified_build(self):
        text = (ROOT / "polishing/native/CMakeLists.txt").read_text()
        sources = re.search(r"add_library\(hprqp_native_polish STATIC (.*?)\)",
                            text, re.S).group(1).split()
        self.assertEqual(tuple(sources), native.NATIVE_SOURCES)
        self.assertEqual(native.C_FLAGS, ("-O3", "-DNDEBUG", "-std=c99", "-DUSE_VENDOR_BLAS"))
        self.assertEqual(native.CXX_FLAGS,
                         ("-O3", "-DNDEBUG", "-std=gnu++17",
                          "-DEIGEN_DONT_PARALLELIZE", "-DEIGEN_MPL2_ONLY"))


class Discovery(unittest.TestCase):
    def test_missing_compiler_is_fatal_before_dependency_download_decision(self):
        with patch.object(native.shutil, "which", return_value=None), \
                self.assertRaisesRegex(RuntimeError, "working CC tool"):
            native.discover_libraries("/unused")

    def test_missing_crypto_is_fatal_instead_of_missing_blas(self):
        with patch.object(native, "tool", return_value=["/compiler"]), \
                patch.object(native, "run"), \
                patch.object(native, "probe", return_value=(None, "crypto unavailable")), \
                self.assertRaisesRegex(RuntimeError, "OpenSSL.*libcrypto"):
            native.discover_libraries("/unused")

    def test_missing_numerical_libraries_return_none(self):
        with patch.object(native, "tool", return_value=["/compiler"]), \
                patch.object(native, "run"), \
                patch.object(native, "probe", side_effect=[(["/crypto"], "")] +
                             [(None, "not found")] * 4) as probe:
            self.assertIsNone(native.discover_libraries("/unused"))
            self.assertEqual(probe.call_count, 5)

    def test_private_libraries_are_selected_and_hash_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            lib = base / "lib"
            lib.mkdir()
            for name in ("libblas.so.3", "liblapack.so.3", "libgfortran.so.5"):
                (lib / name).write_bytes(name.encode())
            crypto = base / "crypto.so"
            crypto.write_bytes(b"crypto")
            with patch.object(native, "tool", return_value=["/compiler"]), \
                    patch.object(native, "run"), \
                    patch.object(native, "probe", side_effect=[([str(crypto)], ""),
                        ([str(lib / "libblas.so.3"), str(lib / "liblapack.so.3")], "")]) as probe:
                result = native.discover_libraries(base)
            self.assertEqual(result["runtime_dir"], str(lib))
            self.assertEqual(len(result["runtime_files"]), 3)
            self.assertEqual(result["runtime_files"][str(lib / "libgfortran.so.5")],
                             native.digest(lib / "libgfortran.so.5"))
            self.assertEqual(probe.call_count, 2)
            self.assertEqual(probe.call_args.args[2],
                             [str(lib / "liblapack.so.3"), str(lib / "libblas.so.3")])

    def test_broken_installed_private_runtime_is_not_silently_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp) / "lib"
            lib.mkdir()
            for name in ("libblas.so.3", "liblapack.so.3"):
                (lib / name).write_bytes(b"invalid")
            with patch.object(native, "tool", return_value=["/compiler"]), \
                    patch.object(native, "run"), \
                    patch.object(native, "probe", side_effect=[(["/crypto"], ""),
                        (None, "wrong architecture")]) as probe, \
                    self.assertRaisesRegex(RuntimeError, "private BLAS/LAPACK cannot run"):
                native.discover_libraries(tmp)
            self.assertEqual(probe.call_count, 2)

    @unittest.skipUnless(shutil.which("cc") and shutil.which("c++"), "native compilers missing")
    def test_real_abi_probe_needs_no_configuration_generator(self):
        original = native.shutil.which
        def check_tool(name, *args, **kwargs):
            self.assertNotIn("cmake", name.lower())
            return original(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(native.shutil, "which", side_effect=check_tool):
            libraries = native.discover_libraries(tmp)
        if libraries is None:
            self.skipTest("host has no BLAS/LAPACK")
        self.assertEqual(libraries["probe"]["symbols"], ["dgemm_", "dgesdd_", "dsyev_"])
        self.assertTrue(libraries["probe"]["passed"])
        self.assertTrue(Path(libraries["crypto"]).is_file())


@unittest.skipUnless(all(shutil.which(t) for t in ("cc", "c++", "ar")), "toolchain missing")
class IncrementalBuild(unittest.TestCase):
    def fixture(self, base):
        """Tiny sources exercise the actual compiler/archive/link pipeline."""
        slu = base / "third_party/superlu/SRC"
        cpp = base / "polishing/native"
        slu.mkdir(parents=True)
        cpp.mkdir(parents=True)
        (slu / "superlu_config.h").write_text("#define INDEX_BITS 32\n")
        (cpp / "common.hpp").write_text("#define BUILD_VALUE 1\n")
        for name in (*native.SUPERLU_GENERIC, *native.SUPERLU_DOUBLE):
            (slu / name).write_text('#include "superlu_config.h"\n' +
                                   'int test_%s(void){return INDEX_BITS;}\n' % Path(name).stem)
        for name in native.NATIVE_SOURCES:
            (cpp / name).write_text('#include "common.hpp"\n' +
                                   'int test_%s(){return BUILD_VALUE;}\n' % Path(name).stem)
        (cpp / "worker.cpp").write_text('#include "common.hpp"\n#include <cstdio>\n'
            'int main(int argc,char**){if(argc==1){fputs("usage: hprqp_polish_worker",stderr);'
            'return 2;}return BUILD_VALUE;}\n')
        (cpp / "unused.hpp").write_text("unrelated header\n")
        selected = {}
        for name in ("blas", "lapack", "crypto"):
            p = base / (name + ".so")
            p.write_bytes(name.encode())
            selected[name] = str(p)
        selected.update(link_flags=[], runtime_dir=None, runtime_files={},
                        probe={"passed": True})
        return selected

    def test_incremental_header_command_library_and_output_invalidation(self):
        with tempfile.TemporaryDirectory(prefix="hprqp direct build ") as tmp:
            root = Path(tmp) / "source with spaces"
            libraries = self.fixture(root)
            base = root / "build"
            base.mkdir()
            # A user can reuse the previous build directory without cleaning.
            (base / "CMakeCache.txt").write_text("old configuration: preserve me\n")
            gpu = root / "build-cuda"
            with patch.object(native, "ROOT", root):
                exe = native.build(base, libraries, gpu_dir=gpu)
                data = json.loads((base / "NATIVE_BUILD.json").read_text())
                self.assertEqual(data["compiled_objects"], data["source_count"])
                self.assertTrue(data["relinked"])
                old_time = exe.stat().st_mtime_ns
                native.build(base, libraries, gpu_dir=gpu)
                data = json.loads((base / "NATIVE_BUILD.json").read_text())
                self.assertEqual(data["compiled_objects"], 0)
                self.assertFalse(data["relinked"])
                self.assertEqual(exe.stat().st_mtime_ns, old_time)
                # Unused headers do not force a rebuild; used header contents do.
                (root / "polishing/native/unused.hpp").write_text("changed unrelated header\n")
                native.build(base, libraries)
                self.assertEqual(json.loads((base / "NATIVE_BUILD.json").read_text())
                                 ["compiled_objects"], 0)
                (root / "polishing/native/common.hpp").write_text("#define BUILD_VALUE 2\n")
                native.build(base, libraries)
                data = json.loads((base / "NATIVE_BUILD.json").read_text())
                self.assertEqual(data["compiled_objects"], len(native.NATIVE_SOURCES) + 1)
                self.assertTrue(data["relinked"])
                # Command changes are tracked even when they happen to emit the
                # same machine code. Object corruption is also never reused.
                with patch.dict(os.environ, CXX="c++ -DHPRQP_BUILD_TEST=1"):
                    native.build(base, libraries)
                    data = json.loads((base / "NATIVE_BUILD.json").read_text())
                    self.assertEqual(data["compiled_objects"], len(native.NATIVE_SOURCES) + 1)
                    self.assertTrue(data["relinked"])
                    Path(libraries["blas"]).write_bytes(b"new library identity")
                    native.build(base, libraries)
                    data = json.loads((base / "NATIVE_BUILD.json").read_text())
                    self.assertEqual(data["compiled_objects"], 0)
                    self.assertTrue(data["relinked"])
                    obj = base / ".direct/obj/superlu/dmach.o"
                    obj.write_bytes(b"corrupt")
                    native.build(base, libraries, gpu_dir=gpu)
                    self.assertEqual(json.loads((base / "NATIVE_BUILD.json").read_text())
                                     ["compiled_objects"], 1)
            self.assertEqual(native.digest(exe), native.digest(gpu / exe.name))
            self.assertEqual((base / "CMakeCache.txt").read_text(),
                             "old configuration: preserve me\n")
            timer = json.loads((base / ".direct/obj/superlu/superlu_timer.json").read_text())
            self.assertIn("-O0", timer["identity"]["command"])

    def test_failed_compile_preserves_published_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            libraries = self.fixture(root)
            base = root / "build"
            with patch.object(native, "ROOT", root):
                exe = native.build(base, libraries)
                before = native.digest(exe)
                (root / "polishing/native/worker.cpp").write_text("invalid C++ source\n")
                with self.assertRaisesRegex(RuntimeError, "command failed"):
                    native.build(base, libraries)
                self.assertEqual(native.digest(exe), before)
                self.assertFalse(list((base / ".direct").rglob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
