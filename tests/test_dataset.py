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

    def test_default_command_resume_and_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "inputs"; data.mkdir()
            (data / "tiny.mps").touch()
            exe = root / "solve_qp"
            exe.write_text("#!/usr/bin/env python3\nimport pathlib,sys\n"
                           "assert not any(x in sys.argv for x in ('--gram','--polishing','--spmv-mode-a'))\n"
                           "p=pathlib.Path(__file__).with_name('calls');p.write_text(p.read_text()+'x' if p.exists() else 'x')\n"
                           "print('status: OPTIMAL\\ntime: 1\\ntotal_time: 2\\nresidual: 1e-10\\n"
                           "gram_active: 1\\npolish_attempts: 1\\npolish_accepted: 1')\n")
            exe.chmod(0o755)
            command = [sys.executable, str(ROOT / "scripts/run_dataset.py"), str(data),
                       str(root / "out"), "--build-dir", str(root)]
            subprocess.run(command, check=True, capture_output=True)
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual((root / "calls").read_text(), "x")
            with (root / "out/HPRQP_result.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["polish_accepted"], "1")
            self.assertGreater(float(rows[0]["process_time"]), 0)
            self.assertEqual(rows[-1]["alg_time"], "1")

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
