import argparse
import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dataset", ROOT / "scripts/run_dataset.py")
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)


class Dataset(unittest.TestCase):
    def test_legacy_qcqp_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qc.mps"
            path.write_text("NAME TEST\nQCMATRIX QC\n X X 0.5\nENDATA\n")
            old = {"status": "OPTIMAL"}
            self.assertFalse(dataset.can_resume(old, path))
            old["input_format"] = dataset.INPUT_FORMAT
            self.assertTrue(dataset.can_resume(old, path))
            old["status"] = "EXIT_1"
            self.assertFalse(dataset.can_resume(old, path))

    def test_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(build_dir=None, solver="solve_qp", use_gpu=None)
            for d in ("build", "build-cuda"):
                (root / d).mkdir()
                (root / d / "solve_qp").touch()
            self.assertEqual(dataset.resolve_solver(args, root), root / "build-cuda/solve_qp")
            args.use_gpu = 0
            self.assertEqual(dataset.resolve_solver(args, root), root / "build/solve_qp")
            args.build_dir = str(root / "explicit")
            self.assertEqual(dataset.resolve_solver(args, root), root / "explicit/solve_qp")

    def test_old_executable_cannot_claim_qcqp_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "inputs"; data.mkdir()
            (data / "qc.mps").write_text("NAME QC\nQCMATRIX ROW\n X X 0.5\nENDATA\n")
            exe = root / "solve_qp"
            exe.write_text("#!/usr/bin/env python3\nprint('status: OPTIMAL\\ntime: 1')\n")
            exe.chmod(0o755)
            subprocess.run([sys.executable, str(ROOT / "scripts/run_dataset.py"),
                            str(data), str(root / "out"), "--build-dir", str(root)],
                           check=True, capture_output=True)
            with (root / "out/HPRQP_result.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["status"], "UNVERIFIED_QCQP_READER")
            self.assertEqual(rows[-1]["alg_time"], "0")

    def test_default_command_resume_and_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "inputs"; data.mkdir()
            (data / "tiny.mps").write_text("NAME QC\nQCMATRIX ROW\n X X 0.5\nENDATA\n")
            exe = root / "solve_qp"
            exe.write_text("#!/usr/bin/env python3\nimport json,pathlib,sys\n"
                           "assert not any(x in sys.argv for x in ('--gram','--polishing','--spmv-mode-a'))\n"
                           "p=pathlib.Path(__file__).with_name('calls');p.write_text(p.read_text()+'x' if p.exists() else 'x')\n"
                           "meta=pathlib.Path(sys.argv[sys.argv.index('--result-json')+1])\n"
                           "meta.write_text(json.dumps(dict(schema=1,input_format='mps_qcmatrix_v1',"
                           "row_cones=1,total_time=2,gram_active=1,polish_attempts=1,polish_accepted=1)))\n"
                           "print('status: polished_optimal\\ntime: 1\\nresidual: 1e-10')\n")
            exe.chmod(0o755)
            command = [sys.executable, str(ROOT / "scripts/run_dataset.py"), str(data),
                       str(root / "out"), "--build-dir", str(root)]
            subprocess.run(command, check=True, capture_output=True)
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual((root / "calls").read_text(), "x")
            with (root / "out/HPRQP_result.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["polish_accepted"], "1")
            self.assertEqual(rows[0]["status"], "polished_optimal")
            self.assertEqual(rows[0]["input_format"], dataset.INPUT_FORMAT)
            self.assertEqual(rows[0]["row_cones"], "1")
            self.assertGreater(float(rows[0]["process_time"]), 0)
            self.assertAlmostEqual(float(rows[-2]["alg_time"]), 1)
            self.assertAlmostEqual(float(rows[-2]["total_time"]), 2)
            self.assertEqual(rows[-1]["alg_time"], "1")
            log = (root / "out/HPRQP_log.txt").read_text()
            for key in dataset.METADATA_FIELDS:
                self.assertNotIn(key + ":", log)

    def test_invalid_metadata_not_success(self):
        for content in ("{", '{"schema":2}', '{"schema":1}'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / "inputs"; data.mkdir()
                (data / "tiny.mps").touch()
                exe = root / "solve_qp"
                exe.write_text("#!/usr/bin/env python3\nimport pathlib,sys\n"
                               "p=pathlib.Path(sys.argv[sys.argv.index('--result-json')+1])\n"
                               f"p.write_text({content!r})\n"
                               "print('status: polished_optimal\\ntime: 1')\n")
                exe.chmod(0o755)
                subprocess.run([sys.executable, str(ROOT / "scripts/run_dataset.py"),
                                str(data), str(root / "out"), "--build-dir", str(root)],
                               check=True, capture_output=True)
                with (root / "out/HPRQP_result.csv").open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(rows[0]["status"], "INVALID_RESULT_METADATA")
                self.assertEqual(rows[-1]["alg_time"], "0")

    def test_failed_status_and_missing_historical_clocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.csv"
            dataset.write_csv(path, [{"name": "success", "status": "OPTIMAL", "alg_time": "1200"},
                                    {"name": "fail", "status": "TIME_LIMIT", "alg_time": "2"}], 1000)
            with path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertAlmostEqual(float(rows[-2]["alg_time"]), ((1210 * 1010)**.5 - 10), places=7)
            self.assertEqual(rows[-2]["process_time"], "")
            self.assertEqual(rows[-1]["alg_time"], "1")


if __name__ == "__main__":
    unittest.main()
