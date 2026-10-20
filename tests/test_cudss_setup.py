"""Pinned archive integrity and local-only extraction tests; no network."""
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/setup_cudss.py"
spec = importlib.util.spec_from_file_location("setup_cudss", SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class Setup(unittest.TestCase):
    def test_bad_checksum_never_installs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "install"
            args = ["setup_cudss.py", "--cuda-root", "/fake/cuda", "--destination", str(dest)]
            with patch("sys.argv", args), patch.object(setup.platform, "system", return_value="Linux"), \
                 patch.object(setup.platform, "machine", return_value="x86_64"), \
                 patch.object(setup.subprocess, "check_output", return_value="release 12.8"), \
                 patch.object(setup, "download", return_value="0"*64):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    setup.main()
            self.assertFalse(dest.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_existing_install_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "install"
            dest.mkdir()
            (dest / "sentinel").write_text("existing installation")
            with patch("sys.argv", ["setup", "--cuda-root", "/cuda", "--destination", str(dest)]), \
                 patch.object(setup.platform, "system", return_value="Linux"), \
                 patch.object(setup.platform, "machine", return_value="x86_64"), \
                 patch.object(setup.subprocess, "check_output", return_value="release 12.8"), \
                 patch.object(setup, "download") as download:
                with self.assertRaisesRegex(SystemExit, "already exists"):
                    setup.main()
                download.assert_not_called()
            self.assertEqual((dest / "sentinel").read_text(), "existing installation")

    def test_unsafe_members_rejected(self):
        for name, link in [("package/../../escape", None), ("/absolute", None),
                           ("package/lib/escape", "/etc"), ("package/lib/escape", "../../../escape")]:
            with self.subTest(name=name, link=link), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)
                with tarfile.open(p / "archive.xz", "w:xz") as tf:
                    m = tarfile.TarInfo(name)
                    if link:
                        m.type = tarfile.SYMTYPE
                        m.linkname = link
                        tf.addfile(m)
                    else:
                        m.size = 1
                        tf.addfile(m, io.BytesIO(b"x"))
                with self.assertRaises(ValueError):
                    setup.extract(p / "archive.xz", p / "stage", "package")
                self.assertFalse((p / "stage").exists())


if __name__ == "__main__":
    unittest.main()
