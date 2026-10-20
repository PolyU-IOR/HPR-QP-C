"""Reproduce same-SONAME runtime collisions without a CUDA device."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("cc"),
                     "Linux ELF linker test needs cc")
class RuntimeLink(unittest.TestCase):
    def test_selected_toolkit_precedes_conflicting_library_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            old, selected = p / "old", p / "selected"
            old.mkdir()
            selected.mkdir()
            (p / "old.c").write_text("int old_api(void){return 1;}\n")
            (p / "new.c").write_text("int new_api(void){return 0;}\n")
            (p / "main.c").write_text("int new_api(void); int main(void){return new_api();}\n")
            for source, directory in (("old.c", old), ("new.c", selected)):
                subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-soname,libapi.so.1",
                                str(p / source), "-o", str(directory / "libapi.so.1")], check=True)
            flags = subprocess.check_output(
                ["make", "-s", "--no-print-directory", "-f", "Makefile", "-f", "-",
                 "_test_runtime_flags", "PYTHON=" + sys.executable],
                cwd=ROOT, input="_test_runtime_flags:\n\t@echo '$(GPU_LINK_LIBS)'\n", text=True)
            self.assertIn("-Xlinker --disable-new-dtags", flags)
            environment = dict(os.environ, LD_LIBRARY_PATH=str(old))
            for mode, extra in (("runpath", ["-Wl,--enable-new-dtags"]),
                                ("rpath", ["-Xlinker", "--disable-new-dtags"])):
                executable = p / mode
                subprocess.run(["cc", str(p / "main.c"), "-L" + str(selected),
                                "-l:libapi.so.1", "-Wl,-rpath," + str(selected),
                                *extra, "-o", str(executable)], check=True)
                result = subprocess.run([str(executable)], env=environment, capture_output=True, text=True)
                if mode == "runpath":
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("new_api", result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_link_policy_invalidates_existing_gpu_configuration_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "gpu-config.json"
            base = [sys.executable, str(ROOT / "scripts/write_gpu_config.py"),
                    "--output", str(output), "--cuda", "/selected"]
            subprocess.run(base, check=True)
            previous = output.read_bytes()
            subprocess.run(base + ["--link_policy", "toolkit-rpath-v1"], check=True)
            self.assertNotEqual(previous, output.read_bytes())
            stamp = output.stat().st_mtime_ns
            subprocess.run(base + ["--link_policy", "toolkit-rpath-v1"], check=True)
            self.assertEqual(stamp, output.stat().st_mtime_ns)
            self.assertEqual(json.loads(output.read_text())["link_policy"], "toolkit-rpath-v1")


if __name__ == "__main__":
    unittest.main()
