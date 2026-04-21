REPO_ROOT := $(abspath $(CURDIR)/..)
PYTHON := $(if $(wildcard $(REPO_ROOT)/.venv/bin/python),$(REPO_ROOT)/.venv/bin/python,python3)
PIP_INSTALL_ARGS ?= -e .[cvxpy]

.PHONY: help setup test demo-qabc demo-mps demo-cvxpy

help:
	@printf '%s\n' \
	  'make setup      Build the shared GPU library and install the Python wrapper with CVXPY' \
	  'make test       Run the Python wrapper test suite' \
	  'make demo-qabc  Run the direct QAbc example' \
	  'make demo-mps   Run the MPS example' \
	  'make demo-cvxpy Run the CVXPY example'

setup:
	$(MAKE) --no-print-directory -C $(REPO_ROOT) shared
	$(PYTHON) -m pip install $(PIP_INSTALL_ARGS)

test:
	$(PYTHON) -m unittest discover -s tests

demo-qabc:
	$(PYTHON) examples/build_qabc_demo.py

demo-mps:
	$(PYTHON) examples/solve_mps_demo.py

demo-cvxpy:
	$(PYTHON) examples/cvxpy_demo.py