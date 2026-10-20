cmake_minimum_required(VERSION 3.16)

project(hprqp_c VERSION 0.1.0 LANGUAGES C CXX)

include(CheckLanguage)
check_language(CUDA)
set(HPRQP_CUDA_DEFAULT OFF)
if(CMAKE_CUDA_COMPILER)
    set(HPRQP_CUDA_DEFAULT ON)
endif()
option(HPRQP_ENABLE_CUDA "Build the CUDA backend when available" ${HPRQP_CUDA_DEFAULT})
option(HPRQP_REQUIRE_CUDSS "Require cuDSS for the default GPU equality Gram capability" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Release flags for C" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Release flags for C++" FORCE)

set(HPRQP_SHARED_SOURCES
    src/shared/polish.cpp
    src/shared/gram_policy.cpp
    src/shared/binding.c
    src/shared/build.c
    src/shared/qcqp.c
    src/shared/mps_qp.cpp
    src/shared/mps_reader.cpp
    src/shared/solver.c
    src/shared/solver_backend.c
    src/shared/sparse.c
    src/shared/vector.c
)

set(HPRQP_CPU_SOURCES
    src/cpu/control.c
    src/cpu/eigs.c
    src/cpu/iteration.c
    src/cpu/residuals.c
    src/cpu/scale.c
    src/cpu/workspace.c
)

set(HPRQP_GPU_SOURCES
    src/gpu/backend_stub.c
    src/gpu/solver_stub.c
)

if(HPRQP_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "HPRQP_ENABLE_CUDA=ON but no CUDA compiler was found")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Release flags for CUDA" FORCE)
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_VERSION})")
    message(STATUS "CUDA toolkit version: ${CUDAToolkit_VERSION}")

    string(REGEX MATCH "^[0-9]+\\.[0-9]+" HPRQP_NVCC_VERSION_MM "${CMAKE_CUDA_COMPILER_VERSION}")
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" HPRQP_TOOLKIT_VERSION_MM "${CUDAToolkit_VERSION}")
    if(HPRQP_NVCC_VERSION_MM AND HPRQP_TOOLKIT_VERSION_MM AND NOT HPRQP_NVCC_VERSION_MM STREQUAL HPRQP_TOOLKIT_VERSION_MM)
        message(FATAL_ERROR
            "Detected a mixed CUDA toolchain: nvcc reports ${CMAKE_CUDA_COMPILER_VERSION} "
            "but CUDAToolkit reports ${CUDAToolkit_VERSION}. This can cause generated stub "
            "compile failures such as '__cudaLaunch requires 2 arguments'. Ensure PATH, "
            "CUDAToolkit_ROOT, and CMAKE_CUDA_COMPILER all point to the same CUDA installation.")
    endif()

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(HPRQP_DETECTED_CUDA_ARCHITECTURES "")
        find_program(HPRQP_NVIDIA_SMI_EXECUTABLE nvidia-smi)
        if(HPRQP_NVIDIA_SMI_EXECUTABLE)
            execute_process(
                COMMAND ${HPRQP_NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader,nounits
                OUTPUT_VARIABLE HPRQP_NVIDIA_SMI_ARCHS_RAW
                RESULT_VARIABLE HPRQP_NVIDIA_SMI_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(HPRQP_NVIDIA_SMI_RESULT EQUAL 0 AND NOT HPRQP_NVIDIA_SMI_ARCHS_RAW STREQUAL "")
                string(REPLACE "\n" ";" HPRQP_NVIDIA_SMI_ARCH_LINES "${HPRQP_NVIDIA_SMI_ARCHS_RAW}")
                foreach(HPRQP_ARCH_LINE IN LISTS HPRQP_NVIDIA_SMI_ARCH_LINES)
                    string(STRIP "${HPRQP_ARCH_LINE}" HPRQP_ARCH_LINE)
                    if(NOT HPRQP_ARCH_LINE STREQUAL "")
                        string(REPLACE "." "" HPRQP_ARCH_VALUE "${HPRQP_ARCH_LINE}")
                        list(APPEND HPRQP_DETECTED_CUDA_ARCHITECTURES "${HPRQP_ARCH_VALUE}")
                    endif()
                endforeach()
            endif()
        endif()

        if(HPRQP_DETECTED_CUDA_ARCHITECTURES)
            list(REMOVE_DUPLICATES HPRQP_DETECTED_CUDA_ARCHITECTURES)
            list(SORT HPRQP_DETECTED_CUDA_ARCHITECTURES COMPARE NATURAL)
            set(CMAKE_CUDA_ARCHITECTURES "${HPRQP_DETECTED_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures for hprqp" FORCE)
            message(STATUS "Auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        else()
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
                set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures for hprqp" FORCE)
            else()
                set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90" CACHE STRING "CUDA architectures for hprqp" FORCE)
            endif()
            message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    else()
        message(STATUS "Using user-specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    set(HPRQP_GPU_SOURCES
        src/gpu/row_metric.cu
        src/gpu/qp_row_metric.cu
        src/gpu/context.cu
        src/gpu/workspace.cu
        src/gpu/scale.cu
        src/gpu/spmv.cu
        src/gpu/eigs.cu
        src/gpu/control.cu
        src/gpu/residuals.cu
        src/gpu/iteration.cu
        src/gpu/solver.cu
    )
endif()

find_library(MATH_LIBRARY m)

add_library(hprqp
    ${HPRQP_SHARED_SOURCES}
    ${HPRQP_CPU_SOURCES}
    ${HPRQP_GPU_SOURCES}
)

add_library(hprqp::hprqp ALIAS hprqp)
target_compile_definitions(hprqp PRIVATE HPRQP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(hprqp PRIVATE ${CMAKE_DL_LIBS})
if(HPRQP_ENABLE_CUDA)
    set(CUDSS_ROOT "" CACHE PATH "cuDSS development installation for equality Gram")
    get_filename_component(HPRQP_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(HPRQP_CUDA_ROOT "${HPRQP_CUDA_BIN_DIR}" DIRECTORY)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        foreach(field include library)
            execute_process(COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_cudss.py
                --cuda-root "${HPRQP_CUDA_ROOT}" --root "${CUDSS_ROOT}" --field ${field}
                OUTPUT_VARIABLE HPRQP_CUDSS_${field} OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE HPRQP_CUDSS_DISCOVERY_STATUS)
            if(NOT HPRQP_CUDSS_DISCOVERY_STATUS EQUAL 0)
                message(FATAL_ERROR "Invalid CUDSS_ROOT=${CUDSS_ROOT}; install a cuDSS package matching this CUDA toolkit.")
            endif()
        endforeach()
    endif()
    if(Python3_Interpreter_FOUND)
        # Use the validated pair, not independently cached include/library
        # values from another CUDA-major installation.
        set(CUDSS_INCLUDE_DIR "${HPRQP_CUDSS_include}")
        set(CUDSS_LIBRARY "${HPRQP_CUDSS_library}")
    else()
        find_path(CUDSS_INCLUDE_DIR cudss.h HINTS ${CUDSS_ROOT}/include)
        find_library(CUDSS_LIBRARY cudss HINTS ${CUDSS_ROOT}/lib ${CUDSS_ROOT}/lib64)
    endif()
    if(CUDSS_INCLUDE_DIR AND CUDSS_LIBRARY)
        target_compile_definitions(hprqp PRIVATE HPRQP_HAVE_CUDSS=1)
        target_include_directories(hprqp PRIVATE ${CUDSS_INCLUDE_DIR})
        target_link_libraries(hprqp PUBLIC ${CUDSS_LIBRARY} CUDA::cufft CUDA::cusolver)
        message(STATUS "cuDSS enabled: ${CUDSS_LIBRARY}")
    else()
        if(HPRQP_REQUIRE_CUDSS)
            message(FATAL_ERROR "Default GPU Gram requires cuDSS. Run make setup-cudss CUDA_ROOT=${HPRQP_CUDA_ROOT} in the source tree, or set CUDSS_ROOT to a matching development package. Set HPRQP_REQUIRE_CUDSS=OFF only for an intentional build without Gram.")
        endif()
        target_compile_definitions(hprqp PRIVATE HPRQP_HAVE_CUDSS=0)
        message(WARNING "cuDSS unavailable: explicitly building without equality Gram support")
    endif()
endif()

target_include_directories(hprqp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shared
)

set_target_properties(hprqp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME hprqp
)

target_compile_options(hprqp
    PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
)

if(MATH_LIBRARY)
    target_link_libraries(hprqp PUBLIC ${MATH_LIBRARY})
endif()

if(HPRQP_ENABLE_CUDA)
    set_property(TARGET hprqp PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_link_libraries(hprqp PUBLIC CUDA::cudart CUDA::cusparse CUDA::cublas)
    target_compile_definitions(hprqp PUBLIC HPRQP_HAVE_CUDA=1)
else()
    target_compile_definitions(hprqp PUBLIC HPRQP_HAVE_CUDA=0)
endif()

add_library(hprqp_test_support
    tests/test_support.c
)

target_include_directories(hprqp_test_support
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
)

target_link_libraries(hprqp_test_support PUBLIC hprqp)

add_executable(test_sparse_helpers
    tests/test_sparse_helpers.c
)

target_link_libraries(test_sparse_helpers PRIVATE hprqp_test_support)

add_executable(test_mps_reader
    tests/test_mps_reader.c
)

target_link_libraries(test_mps_reader PRIVATE hprqp_test_support)
target_compile_definitions(test_mps_reader PRIVATE HPRQP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_builder_preprocess
    tests/test_builder_preprocess.c
)

target_link_libraries(test_builder_preprocess PRIVATE hprqp_test_support)

add_executable(test_binding_api
    tests/test_binding_api.c
)

target_link_libraries(test_binding_api PRIVATE hprqp)
target_compile_definitions(test_binding_api PRIVATE HPRQP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_workspace
    tests/cpu/test_workspace.c
)

target_link_libraries(test_workspace PRIVATE hprqp_test_support)

add_executable(test_scale
    tests/cpu/test_scale.c
)

target_link_libraries(test_scale PRIVATE hprqp_test_support)

add_executable(test_residuals
    tests/cpu/test_residuals.c
)

target_link_libraries(test_residuals PRIVATE hprqp_test_support)

add_executable(test_solver
    tests/cpu/test_solver.c
)

target_link_libraries(test_solver PRIVATE hprqp_test_support)

add_executable(test_iteration
    tests/cpu/test_iteration.c
)

target_link_libraries(test_iteration PRIVATE hprqp_test_support)

add_executable(test_eigs
    tests/cpu/test_eigs.c
)

target_link_libraries(test_eigs PRIVATE hprqp_test_support)

add_executable(test_control
    tests/cpu/test_control.c
)

target_link_libraries(test_control PRIVATE hprqp_test_support)

add_executable(test_gpu_dispatch
    tests/gpu/test_gpu_dispatch.c
)

target_link_libraries(test_gpu_dispatch PRIVATE hprqp_test_support)

if(HPRQP_ENABLE_CUDA)
    add_executable(test_binding_api_gpu
        tests/gpu/test_binding_api_gpu.c
    )

    target_link_libraries(test_binding_api_gpu PRIVATE hprqp)

    add_executable(test_gpu_workspace
        tests/gpu/test_gpu_workspace.c
    )

    target_link_libraries(test_gpu_workspace PRIVATE hprqp_test_support)

    add_executable(test_gpu_scale
        tests/gpu/test_gpu_scale.c
    )

    target_link_libraries(test_gpu_scale PRIVATE hprqp_test_support)

    add_executable(test_gpu_residuals
        tests/gpu/test_gpu_residuals.c
    )

    target_link_libraries(test_gpu_residuals PRIVATE hprqp_test_support)

    add_executable(test_gpu_eigs
        tests/gpu/test_gpu_eigs.c
    )

    target_link_libraries(test_gpu_eigs PRIVATE hprqp_test_support)

    add_executable(test_gpu_control
        tests/gpu/test_gpu_control.c
    )

    target_link_libraries(test_gpu_control PRIVATE hprqp_test_support)

    add_executable(test_gpu_iteration
        tests/gpu/test_gpu_iteration.cpp
    )

    target_link_libraries(test_gpu_iteration PRIVATE hprqp_test_support)

    add_executable(solve_socp examples/solve_socp.c)
    target_link_libraries(solve_socp PRIVATE hprqp)

    add_executable(test_gpu_spmv_operation tests/gpu/test_gpu_spmv_operation.cu)
    target_include_directories(test_gpu_spmv_operation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu)
    target_link_libraries(test_gpu_spmv_operation PRIVATE CUDA::cusparse CUDA::cudart)

    add_executable(test_gpu_soc tests/gpu/test_gpu_soc.cu)
    target_include_directories(test_gpu_soc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu)
    target_link_libraries(test_gpu_soc PRIVATE hprqp CUDA::cublas CUDA::cusparse CUDA::cudart)

    add_executable(test_gpu_solver
        tests/gpu/test_gpu_solver.c
    )

    target_link_libraries(test_gpu_solver PRIVATE hprqp_test_support)

    add_executable(compare_gpu_unchecked_path
        examples/gpu/compare_gpu_unchecked_path.c
    )

    target_link_libraries(compare_gpu_unchecked_path PRIVATE hprqp)

    add_executable(trace_restart_compare
        examples/gpu/trace_restart_compare.c
    )

    target_link_libraries(trace_restart_compare PRIVATE hprqp)
endif()

add_executable(solve_qp
    examples/solve_qp.c
)

target_link_libraries(solve_qp PRIVATE hprqp)
target_compile_definitions(solve_qp PRIVATE HPRQP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

enable_testing()
add_test(NAME test_sparse_helpers COMMAND test_sparse_helpers)
add_test(NAME test_mps_reader COMMAND test_mps_reader)
add_executable(test_mps_qcqp tests/test_mps_qcqp.cpp)
target_link_libraries(test_mps_qcqp PRIVATE hprqp)
add_test(NAME test_mps_qcqp COMMAND test_mps_qcqp)
add_test(NAME test_builder_preprocess COMMAND test_builder_preprocess)
add_test(NAME test_binding_api COMMAND test_binding_api)
add_test(NAME test_workspace COMMAND test_workspace)
add_test(NAME test_scale COMMAND test_scale)
add_test(NAME test_residuals COMMAND test_residuals)
add_test(NAME test_solver COMMAND test_solver)
add_test(NAME test_iteration COMMAND test_iteration)
add_test(NAME test_eigs COMMAND test_eigs)
add_test(NAME test_control COMMAND test_control)
add_test(NAME test_gpu_dispatch COMMAND test_gpu_dispatch)
if(HPRQP_ENABLE_CUDA)
    add_test(NAME test_binding_api_gpu COMMAND test_binding_api_gpu)
    add_test(NAME test_gpu_workspace COMMAND test_gpu_workspace)
    add_test(NAME test_gpu_scale COMMAND test_gpu_scale)
    add_test(NAME test_gpu_residuals COMMAND test_gpu_residuals)
    add_test(NAME test_gpu_eigs COMMAND test_gpu_eigs)
    add_test(NAME test_gpu_control COMMAND test_gpu_control)
    add_test(NAME test_gpu_iteration COMMAND test_gpu_iteration)
    add_test(NAME test_gpu_solver COMMAND test_gpu_solver)
    add_test(NAME test_gpu_spmv_operation COMMAND test_gpu_spmv_operation)
    add_test(NAME test_gpu_soc COMMAND test_gpu_soc)
endif()

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hprqpConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hprqpConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hprqp
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/hprqpConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

set_target_properties(solve_qp PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}" INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(hprqp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS solve_qp RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(TARGETS hprqp
    EXPORT hprqpTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hprqp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hprqp_binding.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shared
    ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT hprqpTargets
    FILE hprqpTargets.cmake
    NAMESPACE hprqp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hprqp
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hprqpConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/hprqpConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hprqp
)

add_executable(test_qcqp tests/test_qcqp.c)
target_link_libraries(test_qcqp PRIVATE hprqp)
add_test(NAME qcqp COMMAND test_qcqp)

add_executable(test_polish_progress tests/polishing/test_progress.cpp)
target_include_directories(test_polish_progress PRIVATE src/shared)
add_test(NAME polish_progress COMMAND test_polish_progress)
if(HPRQP_ENABLE_CUDA)
    add_executable(polish_native_driver tests/polishing/native_driver.cpp)
    target_link_libraries(polish_native_driver PRIVATE hprqp)
endif()
if(HPRQP_ENABLE_CUDA)
    add_subdirectory(polishing/native)
    set_target_properties(hprqp_polish_worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    add_dependencies(hprqp hprqp_polish_worker)
    install(TARGETS hprqp_polish_worker RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/hprqp)
    install(FILES third_party/eigen/COPYING.MPL2 DESTINATION ${CMAKE_INSTALL_DATADIR}/hprqp/licenses/eigen)
    install(FILES third_party/nlohmann/LICENSE.MIT DESTINATION ${CMAKE_INSTALL_DATADIR}/hprqp/licenses/nlohmann)
    install(FILES third_party/pocketfft/LICENSE.md DESTINATION ${CMAKE_INSTALL_DATADIR}/hprqp/licenses/pocketfft)
    install(FILES third_party/superlu/License.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/hprqp/licenses/superlu)
    install(FILES third_party/SCIPY_LICENSE.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/hprqp/licenses/scipy)
endif()
install(FILES docs/equality_gram_scaling.tex DESTINATION ${CMAKE_INSTALL_DATADIR}/hprqp/docs)

add_executable(test_polish_publication tests/polishing/test_publication.cpp)
target_include_directories(test_polish_publication PRIVATE src/shared)
target_link_libraries(test_polish_publication PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME polish_publication COMMAND test_polish_publication)

add_executable(test_defaults tests/test_defaults.c)
target_link_libraries(test_defaults PRIVATE hprqp)
add_test(NAME defaults COMMAND test_defaults)
add_executable(test_polish_worker_path tests/polishing/test_worker_path.cpp)
target_include_directories(test_polish_worker_path PRIVATE src/shared)
target_link_libraries(test_polish_worker_path PRIVATE ${CMAKE_DL_LIBS})
add_test(NAME polish_worker_path COMMAND test_polish_worker_path)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME dataset_entrypoint COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dataset.py)
    add_test(NAME cudss_discovery COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cudss_discovery.py)
    add_test(NAME cudss_setup COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cudss_setup.py)
endif()
