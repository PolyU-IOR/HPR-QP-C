#!/usr/bin/env python3
"""Private native build prerequisites: no sudo, system install or package scripts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from download_utils import open_download
from build_native import discover_libraries, build as build_native

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, expected, algorithm="sha256"):
    with open_download(url) as response, Path(path).open("wb") as f:
        shutil.copyfileobj(response, f)
    if digest(path, algorithm) != expected:
        raise RuntimeError("Downloaded prerequisite checksum mismatch: " + str(path.name))


def supported():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Automatic native dependency setup currently supports Linux x86_64. "
                           "Install BLAS/LAPACK using your platform's package manager.")


def package_records(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith("'"):
            continue
        m = re.fullmatch(r"'([^']+)' (\S+) (\d+) (SHA256|SHA512):([0-9a-fA-F]+)", line)
        if not m:
            raise RuntimeError("APT did not supply a strong checksum: " + line)
        url, name, size, algorithm, checksum = m.groups()
        if Path(name).name != name or not name.endswith(".deb"):
            raise RuntimeError("Invalid APT package filename")
        rows.append(dict(url=url, file=name, size=int(size),
                         algorithm=algorithm.lower(), checksum=checksum.lower()))
    if not rows:
        raise RuntimeError("APT package metadata is unavailable; update package indexes "
                           "or install libblas3/liblapack3 with your package manager.")
    return rows


def install_libraries(deps, allow_download=True):
    supported()
    if not allow_download:
        raise RuntimeError("BLAS/LAPACK are missing, "
                           "and AUTO_INSTALL_NATIVE_DEPS=0. Install native libraries for this platform.")
    if not shutil.which("apt-get") or not shutil.which("dpkg-deb"):
        raise RuntimeError("Automatic BLAS/LAPACK setup requires Debian/Ubuntu apt-get "
                           "and dpkg-deb. Install BLAS/LAPACK manually on this platform.")
    destination = deps / "native"
    if destination.exists():
        raise RuntimeError("Local native dependencies already exist but configuration failed. "
                           "Inspect .deps/native and the native library probe; nothing overwritten.")
    # Repository metadata chooses native-architecture packages compatible with
    # the installed distribution. Never execute package maintainer scripts or
    # privately replace libc, libstdc++, the GPU driver, or system packages.
    packages = ["libblas3", "liblapack3", "libgfortran5", "libquadmath0", "libgcc-s1"]
    dependencies = subprocess.check_output(["apt-cache", "depends", *packages], text=True)
    # GCC runtime copyright files can refer to the matching gcc-N-base package.
    packages += sorted(set(re.findall(r"\b(gcc-\d+-base)\b", dependencies)))
    records = package_records(subprocess.check_output(
        ["apt-get", "--print-uris", "download", *packages], text=True))
    if len(records) != len(packages):
        raise RuntimeError("Incomplete APT native dependency package set")
    print("Installing BLAS/LAPACK and their Fortran runtime locally under .deps/native "
          "(no sudo or system-package changes).", flush=True)
    with tempfile.TemporaryDirectory(prefix=".native-", dir=deps) as tmp:
        stage = Path(tmp)
        unpacked = stage / "unpacked"
        package = stage / "package"
        lib = package / "lib"
        lib.mkdir(parents=True)
        for row in records:
            archive = stage / row["file"]
            download(row["url"], archive, row["checksum"], row["algorithm"])
            if archive.stat().st_size != row["size"]:
                raise RuntimeError("APT archive size mismatch")
            subprocess.run(["dpkg-deb", "-x", str(archive), str(unpacked)], check=True)
        # Flatten only the selected native runtime libraries; this produces a
        # relocatable private lib directory for the worker's transitive RPATH.
        prefixes = ("libblas.so", "liblapack.so", "libgfortran.so",
                    "libquadmath.so", "libgcc_s.so")
        for p in sorted(unpacked.rglob("*")):
            if not p.name.startswith(prefixes) or not p.is_file():
                continue
            target = p.resolve()
            if unpacked.resolve() not in target.parents:
                raise RuntimeError("Package library symlink escapes extraction")
            out = lib / p.name
            if out.exists():
                if digest(out) != digest(target):
                    raise RuntimeError("Conflicting native runtime library: " + p.name)
            else:
                shutil.copy2(target, out)
        for name in ("libblas.so.3", "liblapack.so.3", "libgfortran.so.5",
                     "libquadmath.so.0", "libgcc_s.so.1"):
            if not (lib / name).is_file():
                raise RuntimeError("Downloaded native package set is missing " + name)
        notices = unpacked / "usr/share/doc"
        if notices.is_dir():
            shutil.copytree(notices, package / "licenses", symlinks=False)
        (package / "HPRQP_INSTALL.json").write_text(json.dumps({
            "packages": records, "libraries": {p.name: digest(p) for p in lib.iterdir()},
            "scope": "private extracted runtime libraries; no maintainer scripts"},
            indent=2) + "\n")
        os.rename(package, destination)


def setup(build, allow_download):
    deps = ROOT / ".deps"
    deps.mkdir(exist_ok=True)
    with (deps / "native-setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        libraries = discover_libraries(deps / "native")
        if libraries is None:
            install_libraries(deps, allow_download=allow_download)
            libraries = discover_libraries(deps / "native")
            if libraries is None:
                raise RuntimeError("Installed BLAS/LAPACK cannot link or start on this host; "
                                   "inspect .deps/native before retrying")
        return libraries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=str(ROOT / "build-native-polish"))
    parser.add_argument("--gpu-dir")
    parser.add_argument("--auto-install", choices=("0", "1"), default="1")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--target", choices=("hprqp_polish_worker", "hprqp_polish_inspect"),
                        default="hprqp_polish_worker")
    args = parser.parse_args()
    build = Path(args.build_dir).resolve()
    libraries = setup(build, args.auto_install == "1")
    if args.build:
        build_native(build, libraries, target=args.target, gpu_dir=args.gpu_dir)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit("Native dependency setup: " + str(error))
