#!/usr/bin/env python3
"""Select an existing driver that initializes the selected CUDA runtime.

Run separate native processes with the same Linux RPATH policy as our binaries.
Never install drivers, mutate the shell environment, or treat a hidden GPU as a
successful runtime qualification. The build record is local to this host.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

PROBE = r"""
#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdio.h>
int main(void) {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    const char *path = "";
    void *driver = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    Dl_info info;
    if (driver && dladdr(dlsym(driver, "cuInit"), &info)) path = info.dli_fname;
    printf("%d\n%d\n%s\n%s\n", (int)status, count,
           cudaGetErrorString(status), path);
    if (driver) dlclose(driver);
    return 0;
}
"""


def library_dir(root):
    candidates = [root / "lib64", *sorted(root.glob("targets/*/lib"))]
    for directory in candidates:
        if (directory / "libcudart.so").is_file():
            return directory
    raise RuntimeError("Selected toolkit has no development libcudart.so: " + str(root))


def probe(root, libdir, driver_dir, cc):
    with tempfile.TemporaryDirectory(prefix="hprqp-cuda-probe-") as temp:
        source, binary = Path(temp) / "probe.c", Path(temp) / "probe"
        source.write_text(PROBE)
        rpath = os.pathsep.join(str(p) for p in (libdir, driver_dir) if p)
        command = [*shlex.split(cc), str(source), "-I" + str(root / "include"),
                   str(libdir / "libcudart.so"), "-ldl",
                   "-Wl,--disable-new-dtags", "-Wl,-rpath," + rpath,
                   "-o", str(binary)]
        if driver_dir:
            command += ["-Wl,--no-as-needed", str(driver_dir / "libcuda.so.1"),
                        "-Wl,--as-needed"]
        # Qualify a self-contained launch, not a shell's temporary driver override.
        env = dict(os.environ)
        for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
            env.pop(key, None)
        build = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
        if build.returncode:
            raise RuntimeError("CUDA startup probe could not compile:\n" + build.stderr)
        result = subprocess.run([str(binary)], env=env, capture_output=True,
                                text=True, timeout=30)
        lines = result.stdout.splitlines()
        if result.returncode or len(lines) < 3:
            raise RuntimeError("CUDA startup probe could not run:\n" + result.stderr)
        return dict(code=int(lines[0]), device_count=int(lines[1]),
                    message=lines[2], driver_library=lines[3] if len(lines) > 3 else "")


def select(root, libdir, cc, probe_fn=probe):
    baseline = probe_fn(root, libdir, None, cc)
    record = dict(cuda_root=str(root), runtime_dir=str(libdir),
                  driver_dir="", system_probe=baseline)
    if baseline["code"] == 0 and baseline["device_count"] > 0:
        return dict(record, status="system")
    if baseline["code"] == 100 or (baseline["code"] == 0 and baseline["device_count"] == 0):
        return dict(record, status="unverified_no_device")
    compat = root / "compat"
    if (compat / "libcuda.so.1").is_file():
        alternate = probe_fn(root, libdir, compat, cc)
        record["compat_probe"] = alternate
        if alternate["code"] == 0 and alternate["device_count"] > 0:
            return dict(record, driver_dir=str(compat), status="compat")
    return dict(record, status="failed")


def write_changed(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() == text:
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temp = stream.name
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-root", required=True)
    parser.add_argument("--nvcc", help="reject a compiler from a different toolkit")
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--output", required=True, help="JSON receipt")
    parser.add_argument("--make-output", help="generated Make include")
    parser.add_argument("--cmake-output", help="generated CMake include")
    parser.add_argument("--skip", action="store_true", help="explicit compile-only build")
    args = parser.parse_args()
    output = Path(args.output)
    try:
        root = Path(args.cuda_root).resolve()
        if args.nvcc and Path(args.nvcc).resolve() != (root / "bin/nvcc").resolve():
            raise RuntimeError("Mixed CUDA toolchain: nvcc and CUDA_ROOT belong to different "
                               "installations. Select one toolkit for both.")
        libdir = library_dir(root)
        record = (dict(cuda_root=str(root), runtime_dir=str(libdir), driver_dir="",
                       status="unverified_skipped") if args.skip else select(root, libdir, args.cc))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        record = dict(status="failed", driver_dir="", error=str(exc))
    write_changed(output, json.dumps(record, indent=2, sort_keys=True) + "\n")
    status = record["status"]
    if status == "failed":
        print("CUDA startup validation failed:\n" + json.dumps(record, indent=2), file=sys.stderr)
        print("Use CUDA_ROOT=/path/to/a/driver-compatible/toolkit, or ask the administrator "
              "for a supported driver/compatibility package. No driver was installed or changed. "
              "CUDA_DRIVER_CHECK=0 is for compile-only builds, not a runtime fix.", file=sys.stderr)
        return 1
    if args.make_output:
        # Quotes are supplied by the Make recipes; protect Make's own syntax.
        value = record["driver_dir"].replace("$", "$$").replace("#", r"\#")
        write_changed(Path(args.make_output), "_HPRQP_DRIVER_LIBDIR := " + value + "\n")
    if args.cmake_output:
        value = record["driver_dir"].replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
        write_changed(Path(args.cmake_output), 'set(HPRQP_CUDA_DRIVER_DIR "' + value + '")\n')
    if status == "compat":
        print("CUDA startup verified; embedding existing compatibility driver: " + record["driver_dir"])
    elif status == "system":
        print("CUDA startup verified with the system driver.")
    else:
        print("WARNING: CUDA startup unverified (" + status +
              "). Re-run make on the GPU host before use.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
