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
import tarfile
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


def host_distribution():
    """Do not apply a distribution-specific binary bundle to arbitrary Linux."""
    try:
        values = {}
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"').strip("'")
        return values
    except OSError:
        return {}


def install_bundled_libraries(deps):
    bundle = ROOT / "deps/native"
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    host = host_distribution()
    expected = manifest.get("platform", {})
    actual = dict(system=platform.system(), machine=platform.machine(),
                  os_id=host.get("ID"), version_id=host.get("VERSION_ID"))
    if actual != expected:
        return False
    if manifest.get("format") != 1:
        raise RuntimeError("Unsupported bundled native manifest format")
    name = manifest.get("archive", "")
    if not name or Path(name).name != name:
        raise RuntimeError("Invalid bundled native archive filename")
    archive = bundle / name
    if (not archive.is_file() or archive.stat().st_size != manifest.get("size")
            or digest(archive) != manifest.get("sha256")):
        raise RuntimeError("Bundled native archive checksum/size mismatch; restore deps/native")
    files = manifest.get("files", {})
    required = ("libblas.so.3", "liblapack.so.3", "libgfortran.so.5",
                "libquadmath.so.0", "libgcc_s.so.1")
    if (not all("lib/" + n in files for n in required)
            or not all("licenses/" + n + "/copyright" in files
                       for n in ("libblas3", "liblapack3", "libgfortran5",
                                 "libquadmath0", "libgcc-s1"))
            or "SOURCES.md" not in files or "ORIGIN_INSTALL.json" not in files):
        raise RuntimeError("Incomplete bundled native libraries, licenses or provenance")
    destination = deps / "native"
    if destination.exists():
        raise RuntimeError("Local native dependencies already exist but configuration failed. "
                           "Inspect .deps/native; nothing overwritten.")
    print("Installing bundled BLAS/LAPACK and runtime libraries under .deps/native "
          "(offline; no CMake, APT or system-package changes).", flush=True)
    with tempfile.TemporaryDirectory(prefix=".native-", dir=deps) as tmp:
        package = Path(tmp) / "package"
        package.mkdir()
        seen = set()
        total = 0
        try:
            with tarfile.open(archive, "r:xz") as tf:
                for member in tf:
                    path = Path(member.name)
                    if (path.is_absolute() or ".." in path.parts
                            or path.as_posix() != member.name or not member.isfile()
                            or member.name not in files or member.name in seen):
                        raise RuntimeError("Unsafe or unmanifested bundled native archive member")
                    info = files[member.name]
                    total += member.size
                    if member.size != info["size"] or total > manifest["unpacked_bytes"]:
                        raise RuntimeError("Bundled native file size mismatch")
                    target = package / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    if digest(target) != info["sha256"]:
                        raise RuntimeError("Bundled native file checksum mismatch: " + member.name)
                    seen.add(member.name)
        except tarfile.TarError as exc:
            raise RuntimeError("Invalid bundled native archive") from exc
        if seen != set(files) or total != manifest["unpacked_bytes"]:
            raise RuntimeError("Incomplete bundled native archive")
        libraries = {n: digest(package / "lib" / n) for n in required}
        if libraries != manifest.get("libraries"):
            raise RuntimeError("Bundled native library identity mismatch")
        (package / "HPRQP_INSTALL.json").write_text(json.dumps({
            "origin": "bundled", "platform": actual, "archive_sha256": manifest["sha256"],
            "packages": manifest["packages"], "libraries": libraries,
            "scope": "private unmodified runtime; no maintainer scripts; bundled notices/sources",
        }, indent=2) + "\n")
        os.rename(package, destination)
    return True


def install_libraries(deps, allow_download=True):
    supported()
    if install_bundled_libraries(deps):
        return
    if not allow_download:
        raise RuntimeError("BLAS/LAPACK are missing, no matching bundled runtime is available, "
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
