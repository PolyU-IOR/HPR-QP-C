# Offline native prerequisites

The compact archive contains the BLAS/LAPACK and Fortran/GCC runtime libraries
used by the qualified native polishing worker, with original notices, full
license texts and source-package locations. See SOURCES.md and manifest.json.

Plain make first uses working installed libraries. On Ubuntu 22.04 x86_64,
missing native libraries are extracted from this bundled archive into
.deps/native after integrity checks; no APT or network connection is needed.
Other distributions retain the existing package-manager fallback. Plain make
compiles the native worker directly without CMake; cuDSS is separate from this
archive. Do not replace the payload with binaries
from a different platform while retaining this manifest.
