# Bundled native runtime provenance

This is an unmodified subset of the Ubuntu 22.04 amd64 runtime qualified in S817.
The binary libraries are copied byte-for-byte from the package set recorded in
ORIGIN_INSTALL.json. No library is patched or recompiled. This bundle is used
only on Linux x86_64 with ID=ubuntu and VERSION_ID=22.04. Existing usable native
libraries take precedence. This does not provide a general Linux binary ABI.

Source packages and complete corresponding-source download locations:

- LAPACK 3.10.0-2ubuntu1 (libblas3, liblapack3):
  https://launchpad.net/ubuntu/+source/lapack/3.10.0-2ubuntu1
  https://archive.ubuntu.com/ubuntu/pool/main/l/lapack/
- GCC 12.3.0-1ubuntu1~22.04.3 (libgfortran5, libquadmath0, libgcc-s1):
  https://launchpad.net/ubuntu/+source/gcc-12/12.3.0-1ubuntu1~22.04.3
  https://archive.ubuntu.com/ubuntu/pool/main/g/gcc-12/

Each original binary package filename, URL, size and SHA512 appears in
ORIGIN_INSTALL.json; every redistributed file is SHA256-pinned in manifest.json.
All original copyright notices, runtime exceptions and changelogs are retained
under licenses/, alongside the standard complete license texts they reference.
Consult each component's notices for its terms; the project's license does not
replace the component licenses. The GCC Runtime Library Exception is retained
in the GCC copyright notices.
