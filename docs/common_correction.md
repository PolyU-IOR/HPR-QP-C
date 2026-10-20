# Common frozen correction

For the unscaled native model, minimize `0.5 x'Qx + c'x` subject to
`x in B` and `Ax - b in K`, where B is the box and K combines intervals and
standard SOCs. Use the solver's sign convention `Qx+c-A'y-z=0`.
With private state `u=(x,y)`, every access backend uses

```
g = Qx+c-A'y
p = projection_B(x-g)
s = projection_K(Ax-b-y)
F(u) = (x-p, Ax-b-s).
```

The unit scales are common. At the copied HPR state u0, freeze box and row
projector derivatives E and S. The matrix action is

```
J0 (dx,dy) = ((I-E+EQ)dx - E A' dy,
             (I-S)A dx + S dy).
```

Each correction solves `J0 d = -F(u)` and proposes `u+d`. Projector evaluations
change with the private iterate, but J0 is never refreshed. The same eight-step
limit, 100-times-initial residual-growth guard, relative linear defect check
and native candidate/checker apply to every backend. A candidate can terminate
only after the separate CPU checker rereads its serialized full vectors and
certifies primal feasibility, dual feasibility and objective gap at the owner's
requested tolerance. A small projected residual alone is insufficient.

The canonical candidate takes projected primal p, `w=p`, and normal-cone
slacks from the same two projections; stationarity and objective gap are
rechecked in the original native model units. The immutable model includes
all coefficients, offsets, bounds, cones and objective constant. For converted
QCQP data, this is the solver's conic model. No original-QCQP checker is added.

Sparse LU/SVD, checked band factorization, spatial elimination, nullspace
handling and preconditioning are access adapters for J0. Each returned action
is checked against the original common equations. Structural incompatibility,
resource failure or insufficient accuracy rejects access; it does not select
an alternative nonlinear algorithm. All private vector/residual accumulation
uses the tested extended arithmetic uniformly, with double-precision factors.

The GPU continues its unchanged HPR and Halpern trajectory during an episode.
No partial point, multiplier, penalty or control state returns to the GPU.
Only an independently certified terminal result can replace the returned host
solution. Resource, growth and equation failures leave HPR running.

The live path is `worker.py -> common_correction.episode.polish_common ->
precision_driver.run`. Older helper modules are retained as dependencies or
historical reference; the live dispatcher does not select the old per-family
portfolio. Equality Gram is an independent GPU row metric, as described in
[equality_gram_scaling.tex](equality_gram_scaling.tex).
