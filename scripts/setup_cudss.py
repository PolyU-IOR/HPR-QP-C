#!/usr/bin/env python3
"""Local-only installation of the benchmark-qualified NVIDIA cuDSS.

Called by a default GPU make build, or explicitly with make setup-cudss.
Does not modify CUDA, drivers, system packages, or the Python environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tarfile
import tempfile
from download_utils import open_download

BASE = "https://developer.download.nvidia.com/compute/cudss/redist/"
MANIFEST_URL = BASE + "redistrib_0.7.1.json"
# NVIDIA's published 0.7.1 manifest, version used in release qualification.
SHA256 = {
    ("x86_64", "12"): "946571d9ea164f948e402dd97a14541cb90fbec800336cfa7ae644af5937632f",
    ("x86_64", "13"): "84b34ebe7fad40ec10f2aab2957a63b6070bd8ce16e3ada3e6bcac7317256347",
    ("sbsa", "12"): "a4ddd47ad711243bbea0da2d6520799595cc8d79aa740927e0855c0b8cb69cf8",
    ("sbsa", "13"): "de8295cb6773992bd1ca7e2ad93caae1fd18eba39175f0101b85d53abd3e0179",
}


def download(url, path):
    digest = hashlib.sha256()
    with open_download(url) as src, path.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive, destination, top):
    # Python 3.8 compatible: allow only files/directories and in-tree symlinks.
    # Reject hardlinks, devices, absolute paths and escaping symlink targets.
    with tarfile.open(archive, "r:xz") as tf:
        members = tf.getmembers()
        root = destination.resolve()
        for m in members:
            path = Path(m.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != top:
                raise ValueError("unsafe archive path")
            if not (m.isdir() or m.isfile() or m.issym()):
                raise ValueError("unsupported archive member")
            if m.issym():
                link = Path(m.linkname)
                target = (destination / path.parent / link).resolve()
                if link.is_absolute() or root not in target.parents:
                    raise ValueError("unsafe archive symlink")
        # The archive has a pinned cryptographic identity, in addition to the
        # path checks. Extract into a new, private staging directory only.
        tf.extractall(destination, members=members)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda-root", required=True)
    p.add_argument("--destination")
    a = p.parse_args()
    text = subprocess.check_output([str(Path(a.cuda_root) / "bin/nvcc"), "--version"], text=True)
    match = re.search(r"release (\d+)", text)
    if not match:
        raise SystemExit("Cannot identify selected CUDA toolkit.")
    major = match.group(1)
    machine = platform.machine()
    arch = {"x86_64": "x86_64", "aarch64": "sbsa"}.get(machine)
    if platform.system() != "Linux" or (arch, major) not in SHA256:
        raise SystemExit("Automatic setup supports Linux x86_64/SBSA with CUDA 12 or 13. "
                         "Use NVIDIA's matching development archive and CUDSS_ROOT otherwise.")
    dest = Path(a.destination or Path(__file__).resolve().parents[1] /
                ".deps" / "cudss" / f"cuda{major}").resolve()
    if dest.exists():
        raise SystemExit(f"{dest} already exists; inspect it or use CUDSS_ROOT. Nothing overwritten.")
    name = f"libcudss-linux-{arch}-0.7.1.4_cuda{major}-archive"
    url = BASE + f"libcudss/linux-{arch}/{name}.tar.xz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NVIDIA cuDSS 0.7.1.4 for CUDA {major} into {dest}", flush=True)
    print("NVIDIA license: https://docs.nvidia.com/cuda/cudss/license.html", flush=True)
    with tempfile.TemporaryDirectory(prefix=".cudss-", dir=dest.parent) as temp:
        staging = Path(temp)
        archive = staging / "download.tar.xz"
        digest = download(url, archive)
        if digest != SHA256[(arch, major)]:
            raise RuntimeError("cuDSS archive SHA256 mismatch; nothing installed")
        extract(archive, staging, name)
        package = staging / name
        if not (package / "include/cudss.h").is_file() or not (package / "lib/libcudss.so").is_file():
            raise RuntimeError("cuDSS archive lacks development header/library")
        (package / "HPRQP_INSTALL.json").write_text(json.dumps({
            "url": url, "sha256": digest, "manifest": MANIFEST_URL,
            "cuda_major": major, "architecture": arch, "version": "0.7.1.4",
        }, indent=2) + "\n")
        # Destination must still not exist; do not overwrite an installation.
        if dest.exists():
            raise RuntimeError("cuDSS destination was created concurrently")
        os.rename(package, dest)
    print(f"Installed. Rebuild with the same CUDA_ROOT={a.cuda_root}; no runtime flags needed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit("cuDSS setup: " + str(error))
