REPO_ROOT := $(abspath $(CURDIR)/..)
JULIA ?= julia

.PHONY: help setup test demo-jump demo-mps demo-qabc

help:
	@printf '%s\n' \
	  'make setup      Build the shared GPU library and instantiate the Julia environment' \
	  'make test       Run the Julia package tests' \
	  'make demo-qabc  Run the direct QAbc example' \
	  'make demo-jump  Run the JuMP example' \
	  'make demo-mps   Run the direct MPS example'

setup:
	$(MAKE) --no-print-directory -C $(REPO_ROOT) shared
	$(JULIA) --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

test:
	$(JULIA) --project=. -e 'using Pkg; Pkg.test()'

demo-qabc:
	$(JULIA) --project=. examples/qabc_demo.jl

demo-jump:
	$(JULIA) --project=. examples/jump_demo.jl

demo-mps:
	$(JULIA) --project=. examples/solve_mps_demo.jl