#!/usr/bin/env python3
"""Find an installed cuDSS for the selected CUDA toolkit; never install packages."""
import argparse
import glob
import importlib.metadata
from pathlib import Path
import re
import subprocess
import sys


def compatible(library, major):
    if not major:
        return True
    try:
        output = subprocess.run(["readelf", "-d", str(library)], capture_output=True,
                                text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return True  # The linker remains authoritative on non-ELF platforms.
    # Component SONAMEs (e.g. cusparse.so.12 in CUDA 13) are independent
    # versions. Only cudart's major identifies a CUDA runtime dependency.
    versions = re.findall(r"libcudart\.so\.(\d+)", output)
    return all(v == major for v in versions)


def discover(roots, major):
    for root in roots:
        root = Path(root)
        header = root / "include/cudss.h"
        # Wheels do not ship symlinks. Accept their SONAME as well as the
        # unversioned linker name shipped by NVIDIA's tar/deb packages.
        libraries = [root / sub / name for sub in
                     ("lib", "lib64", "lib/x86_64-linux-gnu", "lib/aarch64-linux-gnu")
                     for name in ("libcudss.so", "libcudss.so.0")]
        if not header.is_file():
            continue
        for library in libraries:
            if library.is_file() and compatible(library, major):
                # The Makefile passes this basename directly to the linker;
                # nvcc does not accept a versioned .so as a positional input.
                return str(header.parent.resolve()), str(library.absolute())
    return "", ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="")
    p.add_argument("--cuda-root", default="")
    p.add_argument("--field", choices=("include", "library", "libdir", "libname"), required=True)
    a = p.parse_args()
    major = ""
    try:
        output = subprocess.check_output([str(Path(a.cuda_root) / "bin/nvcc"), "--version"],
                                         text=True)
        major = re.search(r"release (\d+)", output).group(1)
    except (OSError, subprocess.CalledProcessError, AttributeError):
        pass
    bundled = Path(__file__).resolve().parents[1] / ".deps" / "cudss" / f"cuda{major}"
    roots = [a.root] if a.root else [bundled, a.cuda_root, "/usr/local", "/usr",
                                    "/usr/local/cudss", "/opt/cudss"]
    if not a.root:
        roots += sorted(glob.glob("/opt/nvidia/cudss/*"), reverse=True)
        # NVIDIA's wheel explicitly identifies its CUDA major version.
        if major:
            for name in (f"nvidia-cudss-cu{major}", f"nvidia-cudss-cu{major}x"):
                try:
                    d = importlib.metadata.distribution(name)
                    for f in d.files or ():
                        if str(f).endswith("/include/cudss.h"):
                            roots.append(str(d.locate_file(f).parent.parent))
                except importlib.metadata.PackageNotFoundError:
                    pass
    include, library = discover([r for r in roots if r], major)
    if a.root and not library:
        print("No compatible cuDSS header/library under " + a.root, file=sys.stderr)
        return 1
    fields = {"include": include, "library": library,
              "libdir": str(Path(library).parent) if library else "",
              "libname": Path(library).name if library else ""}
    print(fields[a.field])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
