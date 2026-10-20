#!/usr/bin/env python3
"""Invalidate Make GPU objects when toolkit or cuDSS capability changes."""
import argparse
import json
import os
from pathlib import Path
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
for key in ("cuda", "nvcc", "arch", "cudss_include", "cudss_library", "link_policy", "driver_dir"):
    p.add_argument("--" + key, default="")
a = p.parse_args()
config = vars(a).copy()
output = Path(config.pop("output"))
for key in ("nvcc", "cudss_library"):
    path = Path(config[key])
    config[key + "_identity"] = ([path.resolve().as_posix(), path.stat().st_size,
                                  path.stat().st_mtime_ns] if config[key] and path.is_file() else None)
header = Path(a.cudss_include) / "cudss.h"
driver = Path(a.driver_dir) / "libcuda.so.1"
config["driver_identity"] = ([str(driver.resolve()), driver.stat().st_size,
                              driver.stat().st_mtime_ns]
                             if a.driver_dir and driver.is_file() else None)
config["cudss_header_identity"] = ([header.stat().st_size, header.stat().st_mtime_ns]
                                   if a.cudss_include and header.is_file() else None)
text = json.dumps(config, sort_keys=True, indent=2) + "\n"
if not output.is_file() or output.read_text() != text:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as f:
        f.write(text)
        temp = f.name
    os.replace(temp, output)
