# Native polishing dependencies

The default CPU worker is compiled C++. These sources are vendored so that
building it does not fetch NumPy, SciPy or other Python packages.

| Source | Version | Role | License |
| --- | --- | --- | --- |
| Eigen | 3.4.0 | Sparse/dense containers and arithmetic | MPL-2.0; built with `EIGEN_MPL2_ONLY` |
| SuperLU | 7.0.0 | Sparse LU factorization | BSD |
| pocketfft | commit `4d865cba24359b05b37da4e75aee24498c3b78e7` | Exact tensor sine-transform access | BSD-3-Clause |
| nlohmann/json | 3.11.3 | Private worker request/report format | MIT |

The native MINRES recurrence in `polishing/native/linalg.cpp` is adapted from
SciPy 1.16.2. Its BSD license is retained in `SCIPY_LICENSE.txt`; SciPy is not a
runtime dependency. BLAS/LAPACK can use system libraries or the matching offline
runtime under `deps/native/`; see its manifest, licenses and source provenance.
OpenSSL libcrypto is a system native library. Plain make builds the worker and
these vendored sources directly with the C/C++ compilers; CMake is optional.

`NATIVE_SOURCES.json` records downloaded source URLs, versions and SHA-256
identities. The corresponding licenses are retained in each source directory
and installed with the worker. Unmodified source copies and these notices must
remain together when redistributing this source tree.
