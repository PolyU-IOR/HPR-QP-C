#!/usr/bin/env python3
"""Build the qualified native polishing worker with the C/C++ toolchain only.

No configuration generator or network access is needed.  Numerical sources and
the vendored, 32-bit-index SuperLU configuration are shared with the CMake build.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SUPERLU_GENERIC = (
    "superlu_timer.c", "util.c", "memory.c", "get_perm_c.c", "mmd.c",
    "sp_coletree.c", "sp_preorder.c", "sp_ienv.c", "relax_snode.c",
    "heap_relax_snode.c", "colamd.c", "ilu_relax_snode.c",
    "ilu_heap_relax_snode.c", "mark_relax.c", "mc64ad.c", "qselect.c",
    "input_error.c",
)
SUPERLU_DOUBLE = (
    "dlacon2.c", "dmach.c", "dgssv.c", "dgssvx.c", "dsp_blas2.c",
    "dsp_blas3.c", "dgscon.c", "dlangs.c", "dgsequ.c", "dlaqgs.c",
    "dpivotgrowth.c", "dgsrfs.c", "dgstrf.c", "dgstrs.c", "dcopy_to_ucol.c",
    "dsnode_dfs.c", "dsnode_bmod.c", "dpanel_dfs.c", "dpanel_bmod.c",
    "dreadhb.c", "dreadrb.c", "dreadtriple.c", "dreadMM.c", "dcolumn_dfs.c",
    "dcolumn_bmod.c", "dpivotL.c", "dpruneL.c", "dmemory.c", "dutil.c",
    "dmyblas2.c", "dgsisx.c", "dgsitrf.c", "dldperm.c", "ilu_ddrop_row.c",
    "ilu_dsnode_dfs.c", "ilu_dcolumn_dfs.c", "ilu_dpanel_dfs.c",
    "ilu_dcopy_to_ucol.c", "ilu_dpivotL.c", "ddiagonal.c", "dGetDiagU.c",
)
NATIVE_SOURCES = (
    "model.cpp", "system.cpp", "checker.cpp", "linalg.cpp", "access.cpp",
    "spatial.cpp", "episode.cpp",
)
TARGETS = {
    "hprqp_polish_worker": "polishing/native/worker.cpp",
    "hprqp_polish_inspect": "tests/polishing/native_inspect.cpp",
}
C_FLAGS = ("-O3", "-DNDEBUG", "-std=c99", "-DUSE_VENDOR_BLAS")
CXX_FLAGS = ("-O3", "-DNDEBUG", "-std=gnu++17", "-DEIGEN_DONT_PARALLELIZE",
             "-DEIGEN_MPL2_ONLY")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tool(name, default):
    command = shlex.split(os.environ.get(name, default))
    if not command or not shutil.which(command[0]):
        raise RuntimeError("Native polishing requires a working %s tool (%s)." %
                           (name, os.environ.get(name, default)))
    command[0] = str(Path(shutil.which(command[0])).absolute())
    return command


def run(command, **kwargs):
    """Run argv directly, retaining the actual compiler diagnostic on failure."""
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError("Native build command failed: " + shlex.join(command) +
                           "\n" + result.stdout + result.stderr)
    return result


def tool_identity(command):
    executable = Path(command[0]).resolve()
    return {"command": command, "executable": str(executable),
            "sha256": digest(executable),
            "version": run([*command, "--version"]).stdout[:4096]}


def runtime_flags(directory):
    if not directory:
        return []
    # DT_RPATH is intentional: LAPACK's Fortran dependencies are transitive.
    return ["-Wl,--disable-new-dtags", "-Xlinker", "-rpath", "-Xlinker",
            "$ORIGIN/lib:" + str(Path(directory).resolve())]


CRYPTO_PROBE = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
int main(void) {
  void *lib=dlopen("libcrypto.so.3",RTLD_NOW|RTLD_LOCAL);
  if(!lib) lib=dlopen("libcrypto.so.1.1",RTLD_NOW|RTLD_LOCAL);
  if(!lib) return 2;
  const char *names[]={"EVP_MD_CTX_new","EVP_MD_CTX_free","EVP_sha256",
    "EVP_DigestInit_ex","EVP_DigestUpdate","EVP_DigestFinal_ex"};
  for(unsigned i=0;i<sizeof(names)/sizeof(names[0]);++i)
    if(!dlsym(lib,names[i])) return 3;
  Dl_info info;
  if(!dladdr(dlsym(lib,"EVP_sha256"),&info)) return 4;
  puts(info.dli_fname); dlclose(lib); return 0;
}
'''

LINEAR_PROBE = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
extern void dgemm_(char*,char*,int*,int*,int*,double*,double*,int*,double*,int*,double*,double*,int*);
extern void dgesdd_(char*,int*,int*,double*,int*,double*,double*,int*,double*,int*,double*,int*,int*,int*);
extern void dsyev_(char*,char*,int*,double*,int*,double*,double*,int*,int*);
int main(void) {
  int n=1,info=0,lwork=128,iw[8]; char no='N',all='A',vec='V',lo='L';
  double a=2,b=3,c=0,one=1,zero=0,s=0,u=0,v=0,w[128];
  dgemm_(&no,&no,&n,&n,&n,&one,&a,&n,&b,&n,&zero,&c,&n);
  if(fabs(c-6)>1e-12) return 2;
  dgesdd_(&all,&n,&n,&a,&n,&s,&u,&n,&v,&n,w,&lwork,iw,&info);
  if(info || fabs(s-2)>1e-12) return 3;
  a=3; dsyev_(&vec,&lo,&n,&a,&n,&s,w,&lwork,&info);
  if(info || fabs(s-3)>1e-12) return 4;
  Dl_info blas,lapack;
  if(!dladdr((void*)dgemm_,&blas) || !dladdr((void*)dgesdd_,&lapack)) return 5;
  puts(blas.dli_fname); puts(lapack.dli_fname); return 0;
}
'''


def probe(source, cc, flags, directory):
    """Compile, link and execute a native ABI probe, with no shell or Python libs."""
    with tempfile.TemporaryDirectory(prefix="hprqp-native-probe-") as tmp:
        base = Path(tmp)
        c = base / "probe.c"
        exe = base / "probe"
        c.write_text(source)
        compiled = subprocess.run([*cc, "-std=c99", str(c), "-o", str(exe),
                                   *flags, *runtime_flags(directory), "-lm", "-ldl"],
                                  capture_output=True, text=True)
        if compiled.returncode:
            return None, compiled.stdout + compiled.stderr
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        executed = subprocess.run([str(exe)], capture_output=True, text=True,
                                  env=environment)
        if executed.returncode:
            return None, executed.stdout + executed.stderr + \
                "\nprobe exit status " + str(executed.returncode)
        return executed.stdout.splitlines(), ""


def discover_libraries(native_root):
    """Return verified linker/runtime selections, or None for missing BLAS/LAPACK.

    Missing toolchains or SHA256 providers are fatal rather than reasons to
    download numerical libraries. Existing private libraries take priority.
    """
    cc, cxx = tool("CC", "cc"), tool("CXX", "c++")
    # A compiler that exists but cannot compile/link must fail distinctly from
    # a missing BLAS/LAPACK installation.
    run([*cc, "--version"])
    run([*cxx, "--version"])
    crypto, diagnostic = probe(CRYPTO_PROBE, cc, [], None)
    if not crypto:
        raise RuntimeError("Native polishing requires a working C compiler and "
                           "OpenSSL libcrypto.so.3 or libcrypto.so.1.1 with the EVP "
                           "SHA256 API.\n" + diagnostic)
    private = Path(native_root).resolve() / "lib"
    candidates = []
    if private.is_dir():
        def existing(names):
            return next((private / name for name in names
                         if (private / name).is_file()), None)
        blas = existing(("libblas.so.3", "libblas.so", "libopenblas.so.0", "libopenblas.so"))
        lapack = existing(("liblapack.so.3", "liblapack.so"))
        if blas and lapack:
            candidates.append(([str(lapack), str(blas)], private))
    candidates.extend((flags, None) for flags in (
        ["-llapack", "-lblas"], ["-l:liblapack.so.3", "-l:libblas.so.3"],
        ["-lopenblas"], ["-l:libopenblas.so.0"]))
    diagnostics = []
    for flags, directory in candidates:
        paths, error = probe(LINEAR_PROBE, cc, flags, directory)
        if not paths:
            diagnostics.append(error)
            if directory:
                raise RuntimeError("Installed private BLAS/LAPACK cannot run. Inspect "
                                   + str(private) + ".\n" + error)
            continue
        if len(paths) != 2:
            raise RuntimeError("Native BLAS/LAPACK probe returned invalid library identities")
        blas, lapack = [str(Path(p).resolve(strict=True)) for p in paths]
        crypto_path = str(Path(crypto[0]).resolve(strict=True))
        runtime = {}
        if directory:
            runtime = {str(p): digest(p) for p in sorted(directory.iterdir())
                       if p.is_file()}
        return {"blas": blas, "lapack": lapack, "crypto": crypto_path,
                "link_flags": [lapack, blas],
                "runtime_dir": str(directory) if directory else None,
                "runtime_files": runtime,
                "probe": {"symbols": ["dgemm_", "dgesdd_", "dsyev_"],
                          "passed": True}}
    return None


def atomic_json(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text() == text:
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def dependency_paths(depfile):
    text = Path(depfile).read_text().replace("\\\n", " ")
    if ":" not in text:
        raise RuntimeError("Compiler did not emit a valid dependency file")
    return [Path(p) for p in shlex.split(text.split(":", 1)[1])]


def object_current(obj, identity):
    metadata = obj.with_suffix(".json")
    try:
        saved = json.loads(metadata.read_text())
        return (saved["identity"] == identity and obj.is_file()
                and saved["sha256"] == digest(obj)
                and bool(saved["dependencies"])
                and all(Path(p).is_file() and digest(p) == h
                        for p, h in saved["dependencies"].items()))
    except (OSError, ValueError, KeyError):
        return False


def compile_object(source, obj, compiler, flags, include, compiler_id):
    obj.parent.mkdir(parents=True, exist_ok=True)
    temporary = obj.with_suffix(".o.tmp")
    depfile = obj.with_suffix(".d.tmp")
    command = [*compiler, *flags, *include, "-MMD", "-MF", str(depfile),
               "-MT", str(obj), "-c", str(source), "-o", str(temporary)]
    identity = {"command": command, "tool": compiler_id}
    if object_current(obj, identity):
        return False
    try:
        run(command)
        dependencies = {str(p.resolve()): digest(p) for p in dependency_paths(depfile)}
        if str(source.resolve()) not in dependencies:
            raise RuntimeError("Compiler dependency file omitted its source")
        os.replace(temporary, obj)
        atomic_json(obj.with_suffix(".json"), {"identity": identity,
                    "dependencies": dependencies, "sha256": digest(obj)})
    finally:
        temporary.unlink(missing_ok=True)
        depfile.unlink(missing_ok=True)
    return True


def copy_atomic(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.is_file() and digest(source) == digest(destination):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)


def copy_runtime(libraries, destination):
    if not libraries.get("runtime_dir"):
        return
    for source, checksum in libraries["runtime_files"].items():
        if digest(source) != checksum:
            raise RuntimeError("Native runtime library changed after qualification: " + source)
        copy_atomic(source, Path(destination) / Path(source).name)


def build(build_dir, libraries, target="hprqp_polish_worker", gpu_dir=None):
    """Incremental direct build; return the atomically published executable Path."""
    if target not in TARGETS:
        raise ValueError("Unknown native polishing target: " + target)
    if not libraries:
        raise RuntimeError("Native BLAS/LAPACK libraries have not been qualified")
    base = Path(build_dir).resolve()
    direct = base / ".direct"
    direct.mkdir(parents=True, exist_ok=True)
    with (direct / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cc, cxx, ar = tool("CC", "cc"), tool("CXX", "c++"), tool("AR", "ar")
        cc_id, cxx_id, ar_id = map(tool_identity, (cc, cxx, ar))
        superlu = ROOT / "third_party/superlu/SRC"
        native = ROOT / "polishing/native"
        include = ["-I" + str(p) for p in (native, ROOT / "third_party",
                   ROOT / "third_party/eigen", superlu, ROOT / "src/shared")]
        c_sources = [superlu / name for name in (*SUPERLU_GENERIC, *SUPERLU_DOUBLE)]
        cpp_sources = [native / name for name in NATIVE_SOURCES]
        cpp_sources.append(ROOT / TARGETS[target])
        jobs = []
        for source in c_sources + cpp_sources:
            is_c = source.suffix == ".c"
            flags = list(C_FLAGS if is_c else CXX_FLAGS)
            if source.name in ("superlu_timer.c", "dmach.c"):
                flags.append("-O0")
            obj = direct / "obj" / ("superlu" if is_c else "native") / \
                (source.stem + ".o")
            jobs.append((source, obj, cc if is_c else cxx, flags, include,
                         cc_id if is_c else cxx_id))
        print("Building native C/C++ polishing (no CMake; at most two compiler processes).",
              flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            changed = list(pool.map(lambda args: compile_object(*args), jobs))
        objects = {str(job[1]): digest(job[1]) for job in jobs}
        c_objects = [str(job[1]) for job in jobs[:len(c_sources)]]
        archive = direct / "libsuperlu.a"
        archive_tmp = direct / "libsuperlu.a.tmp"
        archive_identity = {"tool": ar_id, "objects": {p: objects[p] for p in c_objects},
                            "command": [*ar, "qc", str(archive_tmp), *c_objects]}
        archive_record = direct / "archive.json"
        try:
            old = json.loads(archive_record.read_text())
        except (OSError, ValueError):
            old = {}
        if (old.get("identity") != archive_identity or not archive.is_file()
                or old.get("sha256") != digest(archive)):
            archive_tmp.unlink(missing_ok=True)
            try:
                run([*ar, "qc", str(archive_tmp), *c_objects])
                run([*ar, "s", str(archive_tmp)])
                os.replace(archive_tmp, archive)
                atomic_json(archive_record, {"identity": archive_identity,
                                            "sha256": digest(archive)})
            finally:
                archive_tmp.unlink(missing_ok=True)
        library_hashes = {str(Path(libraries[k]).resolve()): digest(libraries[k])
                          for k in ("blas", "lapack", "crypto")}
        for path, checksum in libraries.get("runtime_files", {}).items():
            if digest(path) != checksum:
                raise RuntimeError("Native runtime library changed after qualification: " + path)
            library_hashes[path] = checksum
        executable = base / target
        temporary = direct / (target + ".tmp")
        cpp_objects = [str(job[1]) for job in jobs[len(c_sources):]]
        command = [*cxx, "-O3", "-DNDEBUG", *cpp_objects, str(archive),
                   *libraries["link_flags"], "-lm", "-ldl",
                   *runtime_flags(libraries.get("runtime_dir")), "-o", str(temporary)]
        identity = {"tool": cxx_id, "command": command, "objects": objects,
                    "archive": digest(archive), "libraries": library_hashes,
                    "selection": libraries}
        record = direct / (target + ".json")
        try:
            old = json.loads(record.read_text())
        except (OSError, ValueError):
            old = {}
        copy_runtime(libraries, base / "lib")
        relink = (old.get("identity") != identity or not executable.is_file()
                  or old.get("sha256") != digest(executable))
        if relink:
            try:
                run(command)
                # Both entrypoints reject absent arguments. Reaching their
                # usage error verifies all linked/transitive runtime loading.
                result = subprocess.run([str(temporary)], capture_output=True, text=True,
                                        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"))
                expected = "usage: hprqp_polish_worker" if target == "hprqp_polish_worker" \
                    else "native_inspect mode"
                if result.returncode != 2 or expected not in result.stderr:
                    raise RuntimeError("Built native polishing executable failed startup: " +
                                       result.stdout + result.stderr)
                os.replace(temporary, executable)
                atomic_json(record, {"identity": identity, "sha256": digest(executable)})
            finally:
                temporary.unlink(missing_ok=True)
        if gpu_dir:
            gpu = Path(gpu_dir).resolve()
            copy_runtime(libraries, gpu / "lib")
            copy_atomic(executable, gpu / target)
        atomic_json(base / "NATIVE_BUILD.json", {
            "backend": "direct-c-cxx", "target": target,
            "source_count": len(jobs), "compiled_objects": sum(changed),
            "maximum_compilers": 2, "relinked": relink,
            "sha256": digest(executable), "libraries": library_hashes,
            "record": str(record), "superlu_index_bits": 32,
        })
        return executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=str(ROOT / "build-native-polish"))
    parser.add_argument("--native-root", default=str(ROOT / ".deps/native"))
    parser.add_argument("--gpu-dir")
    parser.add_argument("--target", choices=tuple(TARGETS), default="hprqp_polish_worker")
    args = parser.parse_args()
    libraries = discover_libraries(args.native_root)
    if libraries is None:
        raise RuntimeError("Native BLAS/LAPACK are missing; run make setup-native")
    build(args.build_dir, libraries, args.target, args.gpu_dir)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit("Native direct build: " + str(error))
