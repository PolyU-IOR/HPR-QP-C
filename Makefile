

.PHONY: all cpu gpu shared auto-build auto-shared auto-test auto-install auto-clean run run-gpu \
	test test-cpu gpu-test install dataset dataset-gpu \
	clean clean-cpu gpu-clean rebuild gpu-rebuild fresh gpu-fresh help \
	configure gpu-configure configure-shared gpu-configure-shared \
	build build-shared solve gpu-build gpu-build-shared gpu-solve \
	test-shared gpu-test-shared install-shared gpu-install-shared \
	_check-cmake _check-ctest _announce-auto-build _announce-auto-shared \
	_announce-auto-test _announce-auto-install _announce-auto-clean

.DEFAULT_GOAL := all

CMAKE ?= cmake
CTEST ?= ctest
BUILD_DIR ?= build
GPU_BUILD_DIR ?= build-cuda
SHARED_BUILD_DIR ?= build-shared
GPU_SHARED_BUILD_DIR ?= build-cuda-shared
MODEL ?= model.mps
DATASET_DIR ?= .
RESULT_DIR ?= results
INSTALL_PREFIX ?= $(CURDIR)/install
GPU_INSTALL_PREFIX ?= $(INSTALL_PREFIX)-gpu
PYTHON ?= python3
CUDA_ROOT ?=
CMAKE_ARGS ?=
GPU_CMAKE_ARGS ?=
ARGS ?=
BUILD_MAKEFLAGS := $(strip $(MAKEFLAGS) --no-print-directory)

CUDA_AUTO_ROOT := $(strip $(shell \
	if [ -n "$(CUDA_ROOT)" ]; then \
		candidate="$(CUDA_ROOT)"; \
		if [ -x "$$candidate/bin/nvcc" ] && [ -f "$$candidate/include/cuda.h" ]; then \
			printf '%s\n' "$$candidate"; \
		fi; \
	else \
		{ \
			command -v nvcc 2>/dev/null || true; \
			for candidate in /usr/local/cuda /usr/local/cuda-* /opt/cuda /opt/cuda-*; do \
				[ -x "$$candidate/bin/nvcc" ] || continue; \
				printf '%s/bin/nvcc\n' "$$candidate"; \
			done; \
		} | while IFS= read -r candidate; do \
			[ -n "$$candidate" ] || continue; \
			resolved=$$(readlink -f "$$candidate" 2>/dev/null || printf '%s\n' "$$candidate"); \
			root=$$(dirname $$(dirname "$$resolved")); \
			[ -x "$$root/bin/nvcc" ] || continue; \
			[ -f "$$root/include/cuda.h" ] || continue; \
			version=$$("$$root/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1); \
			[ -n "$$version" ] || continue; \
			printf '%s\t%s\n' "$$version" "$$root"; \
		done | sort -u -t "$$(printf '\t')" -k1,1V -k2,2 | tail -n 1 | awk -F '\t' '{print $$2}'; \
	fi))
CUDA_NVCC ?= $(if $(CUDA_AUTO_ROOT),$(CUDA_AUTO_ROOT)/bin/nvcc)
CUDA_TOOLKIT_ROOT ?= $(if $(CUDA_ROOT),$(CUDA_ROOT),$(CUDA_AUTO_ROOT))
CUDA_TOOLKIT_TARGET_DIR ?= $(firstword $(wildcard $(CUDA_TOOLKIT_ROOT)/targets/*))
CUDA_LIB_DIR ?= $(if $(wildcard $(CUDA_TOOLKIT_ROOT)/lib64),$(CUDA_TOOLKIT_ROOT)/lib64,$(CUDA_TOOLKIT_TARGET_DIR)/lib)
CUDA_STUB_DIR ?= $(CUDA_TOOLKIT_TARGET_DIR)/lib/stubs
GPU_ENV_PREFIX = env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
	CUDA_HOME='$(CUDA_TOOLKIT_ROOT)' CUDA_PATH='$(CUDA_TOOLKIT_ROOT)' CUDACXX='$(CUDA_NVCC)' \
	PATH='$(dir $(CUDA_NVCC)):$(PATH)' \
	LD_LIBRARY_PATH='$(CUDA_LIB_DIR)$(if $(wildcard $(CUDA_STUB_DIR)),:$(CUDA_STUB_DIR))$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH))'
CUDA_CONFIG_ARGS := $(if $(CUDA_NVCC),-DCMAKE_CUDA_COMPILER=$(CUDA_NVCC) -DCUDAToolkit_ROOT=$(CUDA_TOOLKIT_ROOT),)

MAKE_JOBS := $(strip $(shell printf '%s\n' '$(MAKEFLAGS)' | sed -n 's/.*-j[[:space:]]*\([0-9][0-9]*\).*/\1/p'))
ifeq ($(MAKE_JOBS),)
MAKE_JOBS := $(strip $(shell printf '%s\n' '$(MAKEFLAGS)' | sed -n 's/.*--jobs[=[:space:]]\([0-9][0-9]*\).*/\1/p'))
endif
ifneq ($(MAKE_JOBS),)
export CMAKE_BUILD_PARALLEL_LEVEL := $(MAKE_JOBS)
endif

AUTO_SHARED_TARGETS := build-shared
AUTO_BUILD_TARGETS := build
AUTO_TEST_TARGETS := test-cpu
AUTO_INSTALL_TARGETS := install-shared
AUTO_CLEAN_DIRS := $(BUILD_DIR) $(SHARED_BUILD_DIR)
ifneq ($(strip $(CUDA_NVCC)),)
AUTO_SHARED_TARGETS += gpu-build-shared
AUTO_BUILD_TARGETS += gpu-build gpu-build-shared
AUTO_TEST_TARGETS += gpu-test
AUTO_INSTALL_TARGETS += gpu-install-shared
AUTO_CLEAN_DIRS += $(GPU_BUILD_DIR) $(GPU_SHARED_BUILD_DIR)
endif

all: auto-build

cpu: build

gpu: gpu-build

shared: auto-shared

run: solve

run-gpu: gpu-solve

fresh: rebuild

gpu-fresh: gpu-rebuild

help:
	@printf '%s\n' \
	  'Common targets:' \
	  '  make                  Build CPU targets by default, or CPU+GPU targets plus shared libs when CUDA is available' \
	  '  make shared           Build CPU shared libs by default, or CPU+GPU shared libs when CUDA is available' \
	  '  make test             Run CPU tests by default, or CPU+GPU tests when CUDA is available' \
	  '  make install          Install CPU shared artifacts by default, or CPU+GPU shared artifacts when CUDA is available' \
	  '  make clean            Remove default build directories for the current machine' \
	  '  make build            Build the CPU solver and tests only' \
	  '  make build-shared     Build the shared CPU library for wrappers' \
	  '  make gpu-build        Build the CUDA solver and tests only' \
	  '  make gpu-build-shared Build the shared CUDA library for wrappers' \
	  '  make solve            Run solve_qp on MODEL using the CPU build' \
	  '  make gpu-solve        Run solve_qp on MODEL using the GPU build' \
	  '  make test-cpu         Run the CPU test suite only' \
	  '  make test-shared      Run the CPU shared-build test suite' \
	  '  make gpu-test         Run the CUDA test suite' \
	  '  make gpu-test-shared  Run the CUDA shared-build test suite' \
	  '  make install-shared   Install the CPU shared library and headers' \
	  '  make gpu-install-shared Install the CUDA shared library and headers' \
	  '  make dataset          Run solve_qp on all .mps files in DATASET_DIR' \
	  '  make dataset-gpu      Run solve_qp on all .mps files in DATASET_DIR with GPU enabled' \
	  '  make clean-cpu        Remove the CPU build and CPU shared-build directories' \
	  '  make gpu-clean        Remove the CUDA build and CUDA shared-build directories' \
	  '  make rebuild          Same as make clean && make' \
	  '  make gpu-rebuild      Clean and rebuild the CUDA tree' \
	  '' \
	  'Short aliases:' \
	  '  make cpu             Same as make build' \
	  '  make gpu             Same as make gpu-build' \
	  '  make shared          Same as make auto-shared' \
	  '  make run             Same as make solve' \
	  '  make run-gpu         Same as make gpu-solve' \
	  '  make fresh           Same as make rebuild' \
	  '  make gpu-fresh       Same as make gpu-rebuild' \
	  '' \
	  'Variables:' \
	  '  BUILD_DIR=dir          Build directory (default: build)' \
	  '  GPU_BUILD_DIR=dir      CUDA build directory (default: build-cuda)' \
	  '  SHARED_BUILD_DIR=dir   Shared CPU build directory (default: build-shared)' \
	  '  GPU_SHARED_BUILD_DIR=dir Shared CUDA build directory (default: build-cuda-shared)' \
	  '  INSTALL_PREFIX=dir     Install prefix for CPU shared installs and default auto-install' \
	  '  GPU_INSTALL_PREFIX=dir Install prefix for CUDA shared installs in auto-install (default: INSTALL_PREFIX-gpu)' \
	  '  MODEL=path             MPS file for make solve (default: model.mps)' \
	  '  DATASET_DIR=path       Folder of .mps files for dataset targets' \
	  '  RESULT_DIR=path        Output folder for dataset CSV/log files' \
	  '  CUDA_ROOT=path         Override the auto-detected CUDA toolkit for gpu-* targets' \
	  '  CUDA_NVCC=path         Override the auto-detected nvcc for gpu-* targets' \
	  '  ARGS="..."             Extra arguments passed to solve_qp or dataset runner' \
	  '' \
	  'GPU configure prefers nvcc from PATH, then /usr/local/cuda* and /opt/cuda*.' \
	  '' \
	  'Advanced variables:' \
	  '  CMAKE_ARGS="..."       Extra arguments passed to CPU configure' \
	  '  GPU_CMAKE_ARGS="..."   Extra arguments passed to CUDA configure'

_check-cmake:
	@command -v $(CMAKE) >/dev/null 2>&1 || { echo 'error: cmake not found in PATH'; exit 1; }

_check-ctest:
	@command -v $(CTEST) >/dev/null 2>&1 || { echo 'error: ctest not found in PATH'; exit 1; }

_announce-auto-build:
	@if [ -n "$(CUDA_NVCC)" ]; then \
		echo 'CUDA toolkit detected; building CPU and GPU targets, including shared libraries.'; \
	else \
		echo 'CUDA toolkit not detected; building CPU targets only.'; \
	fi

_announce-auto-shared:
	@if [ -n "$(CUDA_NVCC)" ]; then \
		echo 'CUDA toolkit detected; building CPU and GPU shared libraries.'; \
	else \
		echo 'CUDA toolkit not detected; building the CPU shared library only.'; \
	fi

_announce-auto-test:
	@if [ -n "$(CUDA_NVCC)" ]; then \
		echo 'CUDA toolkit detected; running CPU and GPU test suites.'; \
	else \
		echo 'CUDA toolkit not detected; running the CPU test suite only.'; \
	fi

_announce-auto-install:
	@if [ -n "$(CUDA_NVCC)" ]; then \
		echo 'CUDA toolkit detected; installing CPU shared artifacts to $(INSTALL_PREFIX) and CUDA shared artifacts to $(GPU_INSTALL_PREFIX).'; \
	else \
		echo 'CUDA toolkit not detected; installing CPU shared artifacts only to $(INSTALL_PREFIX).'; \
	fi

_announce-auto-clean:
	@if [ -n "$(CUDA_NVCC)" ]; then \
		echo 'CUDA toolkit detected; removing CPU and GPU build directories, including shared-build directories.'; \
	else \
		echo 'CUDA toolkit not detected; removing CPU build directories, including the CPU shared-build directory.'; \
	fi

configure: _check-cmake
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_ARGS)

configure-shared: _check-cmake
	$(CMAKE) -S . -B $(SHARED_BUILD_DIR) -DBUILD_SHARED_LIBS=ON $(CMAKE_ARGS)

gpu-configure: _check-cmake
	@if [ -n "$(CUDA_ROOT)" ] && { [ ! -x "$(CUDA_ROOT)/bin/nvcc" ] || [ ! -f "$(CUDA_ROOT)/include/cuda.h" ]; }; then \
		echo 'error: CUDA_ROOT must point to a CUDA toolkit containing bin/nvcc and include/cuda.h'; \
		exit 1; \
	fi
	@if [ -z "$(CUDA_NVCC)" ]; then \
		echo 'error: unable to find nvcc; set CUDA_ROOT=/path/to/cuda or CUDA_NVCC=/path/to/nvcc'; \
		exit 1; \
	fi
	@printf 'Using CUDA compiler: %s\n' '$(CUDA_NVCC)'
	@printf 'Using CUDA toolkit root: %s\n' '$(CUDA_TOOLKIT_ROOT)'
	$(GPU_ENV_PREFIX) $(CMAKE) -S . -B $(GPU_BUILD_DIR) -DHPRQP_ENABLE_CUDA=ON $(CUDA_CONFIG_ARGS) $(GPU_CMAKE_ARGS)

gpu-configure-shared: _check-cmake
	@if [ -n "$(CUDA_ROOT)" ] && { [ ! -x "$(CUDA_ROOT)/bin/nvcc" ] || [ ! -f "$(CUDA_ROOT)/include/cuda.h" ]; }; then \
		echo 'error: CUDA_ROOT must point to a CUDA toolkit containing bin/nvcc and include/cuda.h'; \
		exit 1; \
	fi
	@if [ -z "$(CUDA_NVCC)" ]; then \
		echo 'error: unable to find nvcc; set CUDA_ROOT=/path/to/cuda or CUDA_NVCC=/path/to/nvcc'; \
		exit 1; \
	fi
	@printf 'Using CUDA compiler: %s\n' '$(CUDA_NVCC)'
	@printf 'Using CUDA toolkit root: %s\n' '$(CUDA_TOOLKIT_ROOT)'
	$(GPU_ENV_PREFIX) $(CMAKE) -S . -B $(GPU_SHARED_BUILD_DIR) -DBUILD_SHARED_LIBS=ON -DHPRQP_ENABLE_CUDA=ON $(CUDA_CONFIG_ARGS) $(GPU_CMAKE_ARGS)

build: configure
	env MAKEFLAGS='$(BUILD_MAKEFLAGS)' $(CMAKE) --build $(BUILD_DIR)

build-shared: configure-shared
	env MAKEFLAGS='$(BUILD_MAKEFLAGS)' $(CMAKE) --build $(SHARED_BUILD_DIR)

gpu-build: gpu-configure
	$(GPU_ENV_PREFIX) MAKEFLAGS='$(BUILD_MAKEFLAGS)' $(CMAKE) --build $(GPU_BUILD_DIR)

gpu-build-shared: gpu-configure-shared
	$(GPU_ENV_PREFIX) MAKEFLAGS='$(BUILD_MAKEFLAGS)' $(CMAKE) --build $(GPU_SHARED_BUILD_DIR)

auto-build: _announce-auto-build $(AUTO_BUILD_TARGETS)

auto-shared: _announce-auto-shared $(AUTO_SHARED_TARGETS)

test: auto-test

test-cpu: _check-ctest build
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

test-shared: _check-ctest build-shared
	$(CTEST) --test-dir $(SHARED_BUILD_DIR) --output-on-failure

gpu-test: _check-ctest gpu-build
	$(GPU_ENV_PREFIX) $(CTEST) --test-dir $(GPU_BUILD_DIR) --output-on-failure

gpu-test-shared: _check-ctest gpu-build-shared
	$(GPU_ENV_PREFIX) $(CTEST) --test-dir $(GPU_SHARED_BUILD_DIR) --output-on-failure

auto-test: _announce-auto-test $(AUTO_TEST_TARGETS)

install: auto-install

install-shared: build-shared
	$(CMAKE) --install $(SHARED_BUILD_DIR) --prefix $(INSTALL_PREFIX)

gpu-install-shared: gpu-build-shared
	$(CMAKE) --install $(GPU_SHARED_BUILD_DIR) --prefix $(GPU_INSTALL_PREFIX)

auto-install: _announce-auto-install $(AUTO_INSTALL_TARGETS)

auto-clean: _announce-auto-clean
	rm -rf $(AUTO_CLEAN_DIRS)

solve: build
	./$(BUILD_DIR)/solve_qp $(MODEL) $(ARGS)

gpu-solve: gpu-build
	$(GPU_ENV_PREFIX) ./$(GPU_BUILD_DIR)/solve_qp $(MODEL) --use-gpu 1 $(ARGS)

dataset: build
	$(PYTHON) scripts/run_dataset.py $(DATASET_DIR) $(RESULT_DIR) --build-dir $(BUILD_DIR) $(ARGS)

dataset-gpu: gpu-build
	$(GPU_ENV_PREFIX) $(PYTHON) scripts/run_dataset.py $(DATASET_DIR) $(RESULT_DIR) --build-dir $(GPU_BUILD_DIR) --use-gpu 1 $(ARGS)

clean: auto-clean

clean-cpu:
	rm -rf $(BUILD_DIR) $(SHARED_BUILD_DIR)

gpu-clean:
	rm -rf $(GPU_BUILD_DIR) $(GPU_SHARED_BUILD_DIR)

rebuild: clean auto-build

gpu-rebuild: gpu-clean gpu-build