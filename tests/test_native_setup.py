"""Native dependency reuse, offline behavior and package-integrity contracts."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location(
    "native_setup", Path(__file__).resolve().parents[1] / "scripts/setup_native.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class NativeSetup(unittest.TestCase):
    def test_existing_cmake_is_reused_without_download(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(setup.shutil, "which", return_value="/existing/cmake"), \
                patch.object(setup, "usable_cmake", return_value=True), \
                patch.object(setup, "download") as download:
            self.assertEqual(setup.ensure_cmake(Path(tmp), True), Path("/existing/cmake"))
            download.assert_not_called()

    def test_offline_missing_cmake_does_not_download(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(setup.shutil, "which", return_value=None), \
                patch.object(setup, "usable_cmake", return_value=False), \
                patch.object(setup, "download") as download:
            with self.assertRaisesRegex(RuntimeError, "AUTO_INSTALL_NATIVE_DEPS=0"):
                setup.ensure_cmake(Path(tmp), False)
            download.assert_not_called()

    def test_apt_requires_strong_hash_and_plain_filename(self):
        row = "'http://mirror/native.deb' native.deb 42 SHA256:" + "a" * 64
        self.assertEqual(setup.package_records(row)[0]["size"], 42)
        for bad in (row.replace("SHA256", "MD5Sum"),
                    row.replace("' native.deb", "' ../native.deb"), "no metadata"):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                setup.package_records(bad)

    def test_unsafe_wheel_never_extracts(self):
        for name, mode in (("../escape", 0o100644), ("/absolute", 0o100644),
                           ("cmake/link", 0o120777)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)
                with zipfile.ZipFile(p / "test.whl", "w") as z:
                    item = zipfile.ZipInfo(name)
                    item.external_attr = mode << 16
                    z.writestr(item, b"/etc")
                with self.assertRaises(ValueError):
                    setup.extract_wheel(p / "test.whl", p / "stage")
                self.assertFalse((p / "stage").exists())

    def test_bad_checksum_cannot_publish_native_install(self):
        rows = "\n".join("'http://mirror/p%d.deb' p%d.deb 1 SHA256:%s" %
                         (i, i, "a" * 64) for i in range(5))
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(setup, "supported"), \
                patch.object(setup.shutil, "which", return_value="/tool"), \
                patch.object(setup.subprocess, "check_output", side_effect=["", rows]), \
                patch.object(setup, "download", side_effect=RuntimeError("checksum mismatch")):
            p = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                setup.install_libraries(p)
            self.assertFalse((p / "native").exists())
            self.assertEqual(list(p.iterdir()), [])

    def test_configure_repairs_only_missing_libraries_once(self):
        missing = subprocess.CompletedProcess([], 1)
        okay = subprocess.CompletedProcess([], 0)
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, "ROOT", Path(tmp)), \
                patch.object(setup, "ensure_cmake", return_value=Path("/cmake")), \
                patch.object(setup, "configure", side_effect=[
                    (missing, "HPRQP_NATIVE_BLAS_LAPACK_MISSING"),
                    (okay, "configured\n")]) as configure, \
                patch.object(setup, "install_libraries") as install:
            self.assertEqual(setup.setup(Path(tmp) / "build", True), Path("/cmake"))
            install.assert_called_once()
            self.assertEqual(configure.call_count, 2)
            self.assertTrue((Path(tmp) / ".deps/native_initial_configure.log").exists())
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, "ROOT", Path(tmp)), \
                patch.object(setup, "ensure_cmake", return_value=Path("/cmake")), \
                patch.object(setup, "configure", return_value=(missing, "unrelated compiler error")), \
                patch.object(setup, "install_libraries") as install:
            with self.assertRaisesRegex(RuntimeError, "configuration failed"):
                setup.setup(Path(tmp) / "build", True)
            install.assert_not_called()

    def test_offline_missing_libraries_does_not_install(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, "ROOT", Path(tmp)), \
                patch.object(setup, "ensure_cmake", return_value=Path("/cmake")), \
                patch.object(setup, "configure", return_value=(
                    subprocess.CompletedProcess([], 1), "HPRQP_NATIVE_BLAS_LAPACK_MISSING")), \
                patch.object(setup, "install_libraries") as install:
            with self.assertRaisesRegex(RuntimeError, "AUTO_INSTALL_NATIVE_DEPS=0"):
                setup.setup(Path(tmp) / "build", False)
            install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
