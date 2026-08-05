# De-domained spec sheets

Six abstract problem sheets mined from real measured numbers in this repository.
The intended use is to hand each sheet to a generator working in a formal numerical
community (randomized linear algebra, approximation theory, mixed-precision numerics,
manifold optimization, fixed-point acceleration, sparse and graph algorithms,
compiler and HPC) and ask for optimization proposals, without steering the proposer
toward the conventions of the domain the numbers came from. The sheets carry no
domain vocabulary on purpose. Every number is traceable to a file, recorded in the
Domain key at the end.

**Generators must be given sheets 1 through 6 only. Withhold the Domain key.** The
key exists for verifier agents who need to check a sheet against the source, and it
uses domain vocabulary freely.

---

## Sheet 1. Sequential batched extremal eigenproblem

**Object.** Find the p algebraically smallest eigenpairs of a Hermitian operator

    A = Fᴴ diag(d) F + diag(t) + U B Uᴴ

where F is a fixed unitary fast transform (an FFT, applied and inverted in O(n log n)),
d and t are real diagonal vectors, U is tall with a few tens of columns, and B is a
small dense Hermitian block. The operator is never formed as a dense matrix. Only the
action A x is available, at the cost of two transforms plus one skinny dense product.
The problem is batched over a set of independent instances that share F and the sparsity
pattern but carry different d, t, U, B. The same batched problem is solved repeatedly
along an outer sequence in which A drifts slowly from one solve to the next.

**Instances and sizes.** The operator dimension n (the length of x) ranges from about
465 on the smallest committed case to about 6746 on a mid case to about 22000 at the
largest case that still fits one 6 GB device. The extremal count p is 8, 20, or 60 across
the committed battery. The working subspace width m tracks p, measured at 16, 40, and up
to 240 (the pre-restart ceiling is 4p). The batch count runs from 8 to 145 independent
instances, and one dense-mesh case reached 384. The fast transform acts on a real-space
box of 20³ to 27³ per instance.

**What varies slowly.** A is regenerated from the current outer iterate on every outer
step, and it moves little once the outer sequence nears its fixed point. Seeding each
solve with the previous converged invariant subspace cuts the outer step count from about
9 to about 2 when the underlying instance is unchanged. An early outer step runs the inner
eigensolver at a loose 1e-3 residual and tightens on a schedule.

**Measured cost profile.** On one small-batch GPU profile the inner solver splits as skinny
dense products 43 percent, the m×m Hermitian eigensolve 21 percent, tall orthonormalization
14 percent, transforms 12 percent, remainder 10 percent, with a 32 percent launch and sync
gap on top. On a wider case (batch 13, p 60, m up to 240) the inner solver is 97.7 percent
of total wall, split as operator action 32.5 percent, orthonormalization 22.8 percent,
subspace build and Ritz combination 40.8 percent, the small dense eigensolve 3.3 percent.
On an 8-core CPU a representative case spent 22 s of 53 s in operator actions, of which 13.5 s
was transforms.

**Precision requirements.** The final eigen-residual required is 1e-9. The subspace reduction
(the projected small dense solve and any metric normalization) must stay in double precision.
The operator actions and the early inner iterations tolerate single precision while the inner
residual is above 1e-5, after which a double-precision polish removes the draft error to 1e-9
with an identical outer trajectory. The draft costs nothing in final accuracy at any size,
measured to 1e-9 agreement across stopping thresholds from 1e-7 to 1e-10.

**Invariants that must hold.** A must remain exactly the same operator under any reordering
or approximation of its action. Any low-rank or interpolative compression of the U B Uᴴ term
must be refinable to a stated rank at which it reproduces the exact action to 1e-13. The
computed invariant subspace, not the individual vectors, is what the outer sequence consumes,
so a basis with a different phase convention is admissible as long as it spans the same
subspace.

**Approaches already measured to fail.** Running the projected small dense solves and the Ritz
combination entirely in single precision floors the achievable residual at 1e-5 and never
reaches 1e-9, on both an easy and a hard conditioning case, even when the single-precision
scope is narrowed to one product family. Capturing the inner solver as one static kernel graph
replays at 1.0 to 1.1 times, no gain, because the actions are already back to back. Removing
per-round host synchronizations with asynchronous copies measured slower at every size.

**Win condition.** A per-iteration deficit of about 9 times against a mature reference
implementation on the hard batched case is the target to close. A factor on the mid-to-large
batch on a device with real double-precision throughput counts. A factor confined to the
smallest instance does not, since that regime is launch-bound rather than throughput-bound.

---

## Sheet 2. Accelerated fixed point with structured Jacobian

**Object.** Iterate x_{t+1} = g(x_t) to a fixed point. Near the fixed point the Jacobian of g
is I − M, where M has a known spectral envelope and is cheaply and approximately diagonalized
in one particular orthogonal basis. In that basis the long-wavelength modes are amplified like
1/q² as the mode index q goes to zero, so the low-q modes carry the slow, large-amplitude error.
The current treatment is a multisecant acceleration that keeps a short history of residual and
iterate vectors, composed with a diagonal preconditioner q²/(q² + q0²) in the amplifying basis
that flattens the 1/q² envelope. One component of x is pinned and the preconditioner maps it to
zero so it is never touched.

**Instances and sizes.** Iteration counts to a fixed tolerance, after the current fixes, sit in
the 13 to 31 range across the committed battery. Representative counts, well-conditioned to hard,
are 12 to 18, 13 to 19, 16 to 21, and a two-sublattice hard case at 31. A preconditioned variant
for inhomogeneous instances moved a slab family from 27 to 21 and from 21 to 17 while leaving a
homogeneous bulk unchanged at 9. Pathological instances exist. One class floors at about 80
iterations in a coupled two-field state while each field alone converges in 13. Another class
never converges under a 300-iteration cap, limit-cycling with an order-10 oscillation, whenever
a scalar coupling strength crosses a threshold between two settings.

**What varies slowly.** The whole fixed-point solve is embedded in an outer parameter sweep, and
the converged iterate from one sweep point seeds the next. The warm-start chain holds the solve
in the correct basin as the parameter moves. A single field check on the first few iterations
detects when the chain has fallen into the wrong basin.

**Measured cost profile.** The iteration count, not the per-iteration cost, is the object here.
The history length is 8, lifted to 12 for the stiffest scheme. A longer history helps a stiff
response at a memory cost. The preconditioner is a couple of transform pairs per step, cheap
against the per-step operator work. Switching the acceleration scheme moved the hard cases the
most, for example 30 to 16, 27 to 18, and 58 to 31 at a fixed final iterate and objective.

**Precision requirements.** The fixed-point tolerance genuinely required ranges from 1e-5 on the
hard coupled-field instances, where a tighter gate is unreachable because a subset of modes floors
on physical noise, to 1e-10 on clean instances. The objective error left by stopping at residual r
is second order in r, so a residual gate over-tightens on instances where the residual floors while
the objective is already settled. An objective-error gate at 1e-6 (relaxed to 1e-4 on the hard
coupled cases) replaces the residual gate there.

**Invariants that must hold.** Any preconditioner or accelerator must leave the fixed point exactly
unchanged. A bad preconditioner may only cost iterations, never move the solution. The pinned
component must stay pinned, so the preconditioner must send it to zero by construction.

**Approaches already measured to fail.** Step-size control is dominated by the right preconditioner
and is not worth chasing. Probing the amplifying basis with undamped iteration to estimate the
envelope sloshes and must instead probe with the preconditioner on and divide its factor back out.
A better initial iterate does not cut the count, since the count is set by the Jacobian spectrum and
the acceleration, not by the start.

**Win condition.** A reduction in iteration count to the fixed point at an identical final iterate,
on the hard and pathological instances specifically. A scheme that turns a never-converging
limit-cycle into a converging solve, or that pulls the 31 and 80 cases down, counts. A gain only on
the already-easy 9-to-13 instances does not.

---

## Sheet 3. Tall-skinny orthonormalization and small dense solves at scale

**Object.** Per inner iteration, batched over independent instances, orthonormalize a tall matrix
V of shape n × m, form the m × m Hermitian Gram or subspace matrix from V and a second tall matrix
H V, and solve that small dense Hermitian eigenproblem. n is the tall dimension, m the skinny one.
The three steps are a batched tall QR, a batched skinny product, and a batched small Hermitian
eigensolve.

**Instances and sizes.** n ranges from about 465 to about 22000. m ranges from 16 to 240. The batch
count is 8 to 145, up to 384 on the densest committed case. The tall block is complex double
precision, so one instance of V is n × m × 16 bytes and the batch multiplies that.

**Measured cost profile.** On a small-batch GPU profile the three steps are skinny product 43 percent,
small eigensolve 21 percent, tall QR 14 percent of device-busy time. On a narrower-instance profile the
tall QR alone is 44 percent. Three scaling cliffs are documented. First, forming the Gram with a product
that materializes a conjugated copy of the tall block spikes memory by batch × m × n × 16 bytes, about
7 GiB per copy at the dense case, and one run died at 32.2 GiB when the Gram requested another 6.68 GiB.
Second, an O(n²) dense workspace inside the small eigensolve allocates about 37 GB in one shot at n near
22000, a hard ceiling reached at one problem-size step above where linear scaling held. Third, the batched
small Hermitian eigensolve on the GPU falls off a fast path at m > 32 and drops to a per-instance loop,
measured about 83 times slower at the boundary, and the eigensolve share rises from 11 percent at m 16 to
21 percent at m 40. The tall QR on the GPU was measured at 3.9 ms for a small shape against 0.3 ms for a
host round trip of the identical shape, above 10 times, because a batched QR factorization pays a fixed
per-call cost a tiny problem cannot amortize.

**What varies slowly.** V is reseeded from the previous inner iteration and, across the outer sequence,
from the previous outer solve, so the subspace it spans drifts slowly. The shapes n, m, and batch count
are fixed across an outer sequence.

**Precision requirements.** The Gram formation and the small Hermitian eigensolve must stay in double
precision. A single-precision factorization of the near-singular metric produces invalid rotations. The
tall QR may run in any precision that spans the same subspace, since only the span feeds the next step.

**Invariants that must hold.** Orthonormality must hold to double precision at output. The subspace
spanned by V, not its particular basis, is the invariant the next step consumes. The small eigensolve
must return eigenvectors consistent with the double-precision Gram.

**Approaches already measured to fail.** Computing the subspace product in single precision and upcasting
only the result floors the outer solve at 1e-5. Streaming or chunking the tall blocks to keep them off the
device during the small solve is worse than either keeping them fully resident or fully host-side, because
per-transfer overhead dominates at these shapes, and a full host-resident redesign nets roughly flat once
every consumer of the tall blocks is counted. Widening the host-offload of the tall QR past m 16 is a loss
by m 24 and mixed to negative through m 60 at the larger n, because the transfer of the larger tall block
erodes the win.

**Win condition.** A per-iteration factor on the orthonormalization plus small-solve bundle at m up to 240
and n up to 22000, on a device whose small-matrix factorizations are throughput-bound. Removing the 37 GB
O(n²) cliff by tiling, so the largest instance fits a small device, counts on its own. A host-offload gain
of 1.55 to 1.67 times per iteration on the small tall shapes is the demonstrated baseline to beat.

---

## Sheet 4. The hardware and kernel object

**Object.** The inner loop is dominated by three batched kernel families, run tens of times per outer
step. Batched skinny complex double-precision dense products. Batched small Hermitian eigensolves at
m about 16 to 40. Batched 3D fast transforms on 20³ to 27³ boxes. Between them sits a large population of
tiny elementwise and reduction kernels whose dispatch is a 32 to 46 percent launch and sync gap on top of
device-busy time. The question is how to schedule and fuse these families across three device classes.

**Instances and sizes.** See sheets 1 and 3 for n, m, and batch ranges. The transform boxes are 20³ to 27³.
The skinny products are complex double precision.

**Measured cost profile by device.** On a consumer GPU with double precision at 1/64 of single (call it
device A, 6 GB), the batched transform runs 6.5 ms in double and 1.1 ms in single (12 times), and the small
Hermitian eigensolve runs 2.4 ms in double and 0.5 ms in single (10 times). The same double transform runs
13.1 ms and the eigensolve 4.9 ms on an 8-core AVX2 CPU (device B). On a datacenter GPU with real
double-precision throughput (device C), a subspace product measured 1.26 ms against 139.6 ms host-resident,
and full mid cases ran 39.4 s and 46.8 s against 2275 s and 1128 s on device B, 58 and 24 times. On device A
a small case ran at 100 percent utilization while drawing only 25 W of a 60 W budget, direct evidence the
bound is arithmetic throughput, not clock or power. The honest per-iteration deficit against a mature
reference implementation on the hard case is about 9 times, and the full deficit factors as about 9 times
per iteration and 2.3 times more iterations, about 21 times total on device B.

**What varies slowly.** Kernel shapes are fixed across an outer sequence, so any per-shape warmup, autotune,
or capture amortizes over many outer steps.

**Precision requirements.** The transforms that conserve a global invariant, and the subspace reduction, must
stay in double precision. The skinny products and early actions tolerate single precision above the 1e-5 inner
residual. Single-precision transforms are accuracy-fatal against the double-precision reference on this
workload.

**Invariants that must hold.** The kernel schedule must not change the computed result beyond the stated
precision. On the double-precision path the result must bit-match across device classes, which is the
correctness gate that let the datacenter numbers stand as speed, not error.

**Approaches already measured to fail.** A structural rewrite aimed at launch latency does nothing, since the
small-instance gap is double-precision throughput, not dispatch. Capturing the real batched operator action as
one static kernel graph replays at 1.0 to 1.1 times. Capturing a whole inner round's post-eigensolve elementwise
math also replays at 1.0 times, no launch gap anywhere in the round on device A. Half-precision transforms are
ruled out on accuracy.

**Win condition.** Closing the 9-times per-iteration deficit on device B, or reclaiming a meaningful slice of the
32 to 46 percent launch gap on device A on the non-transform fraction (the tiny-kernel glue), where the host and
consumer GPU hurt most. A single-precision-dominant schedule that drafts far deeper and reserves double precision
for a final polish is the named candidate. Any proposal must state which device class it targets, since verdicts
here invert across device classes.

---

## Sheet 5. Jacobian and Hessian estimation of an expensive smooth map

**Object.** Estimate a symmetric second-derivative tensor of a smooth scalar-valued map whose single
forward evaluation is a full fixed-point solve costing seconds to hours. Two shapes occur. A 6 × 6
symmetric tensor obtained by two-sided differencing of an analytic first derivative, at 12 forward solves.
A 3N × 3N Hessian of an N-site system with no exploitable symmetry, at 6N forward solves by two-sided
differencing over each of 3N coordinates.

**Instances and sizes.** The 6 × 6 case runs on cells of about 2 to 9 sites, at 12 forward solves each.
The Hessian case runs at 6·N_home forward solves, where site symmetry can collapse the column count (a
high-symmetry crystal needs one displacement column of six, a lower-symmetry one needs two, reconstructed
by the group action). Per-forward-solve cost spans the full range of the underlying solve, from 0.3 to 97 s
on a small many-instance case, to about 11 min per direction on a hard case whose full solve is about 84 min.

**What varies slowly.** Each finite-difference forward solve is a small perturbation of the base solve and can
warm-start from it. The analytic first derivative is exact and cheap once the base solve is converged, measured
to agree with a full-reconvergence difference to 3.2e-10 relative, four decades below the differencing floor,
and the first-derivative field costs about 0.07 s per step against a multi-second solve.

**Measured cost profile.** The forward solve dominates entirely. Differencing the analytic first derivative
(12 solves for the 6 × 6) is far cheaper than differencing the scalar directly. Exact adjoints and
Hessian-vector products exist. The full second derivative is one adjoint pass over the first-derivative
graph, and a trust-region second-order solver already consumes single Hessian-vector products without ever
forming the Hessian. For the Hessian case, the tensor decays with inter-site distance for a class of instances,
so the matrix is effectively banded in a site-distance metric, which is what makes a graph-coloring or
probing scheme plausible.

**Precision requirements.** A complete forward-solve re-run at 1e-5 to 1e-7 relative is the floor for a
first-derivative finite difference. The analytic first derivative itself is exact to the solve tolerance, so
the accuracy limit is the differencing step and the solve floor, not the derivative.

**Invariants that must hold.** Every probe must evaluate the same forward map. The adjoint and
Hessian-vector product are exact only at the fixed point of the forward solve, so a probe must run the
inner solve to tolerance before the derivative is read. Any reduced set of probes must reconstruct the full
tensor to the differencing floor.

**Approaches already measured to fail.** Fitting a small parameter correction against a single-observable
loss is rank-deficient. The relevant Jacobian had singular values 1.7e-2, 2.8e-4, 4.2e-6, 1.5e-7 (condition
1.1e5), so a 5-point single-observable window determined at most two of four directions and a recovery oracle
left the error in the two weakest directions. The lesson is that a probing or coloring scheme must be driven
by a multi-observable target, not one scalar.

**Win condition.** Fewer forward solves than the current 6N (Hessian) or 12 (6 × 6 tensor) at the same tensor
accuracy, by exploiting the exact Hessian-vector product, the inter-site decay, and a coloring or probing
structure. A graph-coloring probe that reconstructs a banded 3N × 3N Hessian in far fewer than 6N solves is
the clearest win.

---

## Sheet 6. Many similar instances

**Object.** Solve 5 to 20 nearly identical fixed-point instances that differ only in one scalar parameter.
The instances are currently solved serially, each warm-started from its predecessor. The question is whether
to batch them into one wide solve, run them as concurrent shared-device processes, or keep them serial.

**Instances and sizes.** Sweep lengths of 5 to 20. Two sweep shapes occur. One holds the operator dimension,
transform box, and all tensor shapes fixed across the sweep, varying only the scalar, so the batch has zero
raggedness. The other varies the operator dimension and transform box with the scalar, so a batched solve is
ragged. One instance uses about 2.6 GB of a 6 GB device, so two instances fit at once.

**What varies slowly.** The fixed point barely moves between adjacent sweep points, so the warm-start chain
converges each point in far fewer iterations than a cold start. The per-point iteration count differs across
the sweep, with the extreme-parameter points needing more iterations than the mild ones.

**Measured cost profile.** A single small instance underfills the device, at 100 percent utilization but only
25 W of a 60-to-80 W budget and 2.6 of 6 GB, so it is launch and latency bound with real headroom. Concurrent
shared-device processes are estimated at 1.5 to 1.8 times on such a launch-bound instance. A batched wide solve
folds the small per-instance products into one large product and amortizes the launch overhead, the main
structural gain, but requires the solver to carry per-instance state and accelerate each instance independently
while sharing the linear algebra. The clean target is the zero-raggedness sweep, where every point shares shapes
exactly.

**What varies, what is shared.** Shared across a sweep point are the fast transform, the sparsity pattern, and
(for the clean sweep) all shapes. Varied are the scalar parameter, the diagonal and low-rank data of the
operator, and the per-point iteration count. On the ragged sweep the operator dimension and transform box also
vary.

**Precision requirements.** All instances run at the same double-precision tolerance and must agree bit-for-bit
with their serial counterparts where shapes are shared.

**Invariants that must hold.** Batching must not change any instance's fixed point. A batched lockstep solve
must respect each instance's own convergence, either by masking converged instances out or by carrying
per-instance state, so that an easy instance is not over-iterated and a hard instance is not stopped early.

**Approaches already measured to fail.** Dropping the warm-start chain to gain concurrency is close to a wash at
two instances, since it trades the warm-start iteration savings for the concurrency. A naive lockstep batched
solve over-iterates the easy sweep members up to the hardest member's count unless per-instance convergence
masking is added.

**Win condition.** Filling an underused device by batching the clean zero-raggedness sweep into one wide solve,
on a device with real double-precision throughput and enough memory to hold the batch. Concurrency of 1.5 to
1.8 times on the launch-bound small instances is the cheap baseline. The batched wide solve is the larger
structural win and is the one that scales to the datacenter device.

---

## Domain key (for verifiers only, not for generators)

This section maps each sheet's abstract terms to the real gradwave symbols and source
files. It uses domain vocabulary freely.

| Sheet | Abstract term | Real object | Source |
|---|---|---|---|
| 1 | Structured operator A = Fᴴ diag(d) F + diag(t) + U B Uᴴ | Kohn-Sham Hamiltonian: kinetic diag(|k+G|²) in G-space, local potential diagonal in real space via FFT F, nonlocal KB projectors U B Uᴴ | `core/batch.py` `BatchedHamiltonian.apply`, `postscf/exchange.py` |
| 1 | Batch of independent instances | k-points in the irreducible wedge (and spin channels) | `kpoints.monkhorst_pack`, `symmetry.reduce_mesh` |
| 1 | Extremal eigenpairs, count p | lowest nband occupied+buffer bands | `solvers/davidson.py` `davidson_batched` |
| 1 | Subspace width m | Davidson subspace dim ~2·nband, ceiling 4·nband before restart | `solvers/davidson.py` |
| 1 | Outer sequence, A drifts slowly | SCF self-consistency loop | `scf/loop.py` `scf`, `scf/uspp_loop.py` |
| 1 | Warm-start 9→2 outer steps | ASE calculator density/orbital reuse, same-position restart | `docs/manual/performance.md` "Warm-start SCF" |
| 1 | Sizes npw 465 / 6746 / 22000; p 8/20/60; m 16/40/240; boxes 20³-27³ | diamond-C 50 Ry nk=8 npw=465; hematite npw≈6746 nb=60; Si-216 npw≈22k | `docs/manual/performance.md` case studies, `docs/ideas.md` acceleration frontier and size-ceiling |
| 1 | Cost split GEMM 43 / eigh 21 / QR 14 / FFT 12 | Si8 2x2x2 GPU torch.profiler | `docs/ideas.md` lines 1126-1129 |
| 1 | Inner solver 97.7% of wall; h_apply 32.5 / ortho 22.8 / RR 40.8 / eigh 3.3 | hematite α-Fe₂O₃ 10-atom nspin=2 | `docs/manual/performance.md` "large-nb magnetic mineral" |
| 1 | Residual 1e-9; loose 1e-3 early; fp32 draft above 1e-5 | davidson `tol=1e-9` default, adaptive diago schedule, `mixed_precision` | `solvers/davidson.py:367`, `docs/manual/performance.md` "Mixed precision" |
| 1 | fp32 subspace floors at 1e-5 | issue #136, RR-GEMM fp32 rejected | `docs/manual/performance.md` "What does not help" |
| 2 | Fixed-point map g, Jacobian I − M, 1/q² envelope | SCF density mixing, Hartree charge response 4πe²χ/G² | `scf/mixing.py`, `docs/manual/convergence.md` "charge sloshing" |
| 2 | Amplifying basis, preconditioner q²/(q²+q0²) | Kerker preconditioner in G-space; local-TF variant | `scf/mixing.py`, `docs/manual/performance.md` "Local-TF preconditioner" |
| 2 | Multisecant, history 8, lifted 12 | Pulay/Broyden/Johnson mixing, `history` default 8 / 12 | `docs/manual/convergence.md` line 198 |
| 2 | Iteration counts 12-18/13-19/16-21/31; 30/27/58 pre-fix | Si/Cu/Pt; bcc Fe, fcc Ni Stoner, AFM Fe (Johnson vs Pulay) | `docs/manual/convergence.md` lines 163-165, wisdom.md line 258 |
| 2 | Slab 27→21, 21→17; bulk 9 | fcc Al slabs local-TF vs Kerker | `docs/manual/convergence.md` lines 191-194 |
| 2 | Floors at 80, 200-iter | mixed collinear/spinor Ni near Stoner, magnetization channel | `docs/ideas.md` lines 897-910 |
| 2 | Never converges under 300 cap, order-10 oscillation, scalar threshold | DFT+U on Pt(111)+H, U(Pt 5d)=8.949 eV, threshold (4,6] eV | `docs/ideas.md` "Large-U divergence", convergence.md lines 204-217 |
| 2 | Objective gate; error 2nd order in residual; entol 1e-6 / 1e-4 | `scf.convergence: energy`, `entol`, rhotol 1e-5 metals | `docs/manual/convergence.md` lines 63-128 |
| 3 | Tall matrix n×m, QR + Gram + m×m eigh | `_orthonormalize_b`, subspace build `s = matmul(v.conj(), hv.mT)`, `_eigh_subspace` | `solvers/davidson.py` |
| 3 | conj-copy spike 7 GiB, died at 32.2 GiB + 6.68 GiB | 384-k FePt A100 run, `einsum("kig,kjg->kij", v.conj(), hv)` | `docs/ideas.md` "Davidson subspace Gram" lines 1614-1625 |
| 3 | O(n²) 37 GB cliff at n≈22k | 128-atom dense-allocation cliff, complex128 ~7.7 GB × eigh workspace | `docs/ideas.md` "size ceiling" lines 1593-1600 |
| 3 | eigh n>32 cliff ~83×; 11%→21% | cusolverXsyevBatched fallback, pytorch#175585 | `docs/ideas.md` lines 1157-1164 |
| 3 | QR 3.9 ms GPU vs 0.3 ms host, cols≤16 | batched-QR CPU-offload `_qr_offload` | `docs/manual/performance.md` "CUDA batched-QR CPU-offload" |
| 3 | fp32 Cholesky invalid rotations | USPP overlap S near-singular, issue #136 | `docs/manual/performance.md` "Mixed precision" |
| 3 | QR offload 1.55-1.67×/iter | diamond-C 4³/6³ RTX 3050 | `docs/manual/performance.md` table lines 110-114 |
| 4 | Three kernel families | complex-c128 skinny GEMM, m~16-40 Hermitian eigh, 20³-27³ FFT | `core/batch.py`, `solvers/davidson.py` |
| 4 | Device A (SM86, fp64 1/64, 6 GB) | RTX 3050; FFT 6.5/1.1 ms, eigh 2.4/0.5 ms | `docs/manual/performance.md` "GPU limit is precision" table |
| 4 | Device B (8-core AVX2 CPU) | laptop CPU; FFT 13.1 ms, eigh 4.9 ms | same table |
| 4 | Device C (datacenter fp64) | H100/A100; hematite 39.4 s, Cr2O3 46.8 s, GEMM 1.26 vs 139.6 ms | `docs/manual/performance.md` "datacenter fp64", `docs/ideas.md` H100 session |
| 4 | 9× per-iter, 21× total, 2.3× iterations | fcc Pt PAW metal vs QE pw.x | `docs/manual/performance.md` "hard PAW metal" |
| 4 | 32-46% launch gap | Si8/Si2 torch.profiler | `docs/ideas.md` lines 1126-1134 |
| 4 | CUDA graph 1.0-1.1×; whole-round 1.0× | apply-only and post-eigh capture rejected | `docs/manual/performance.md` "CUDA graphs" |
| 5 | 6×6 symmetric tensor, 12 solves | elastic constants C_ij Voigt, six ±h strains, central diff of analytic stress | `postscf/elastic.py`, `api.run_elastic` |
| 5 | 3N×3N Hessian, 6N solves, site symmetry collapses columns | Γ-phonon force constants, `HessianSymmetry` irreducible columns | `postscf/phonons.py`, `postscf/phonons_supercell.py` |
| 5 | Exact adjoint / HVP; trust-region 2nd-order | `relax.method: newton` exact-Hvp Steihaug Newton-CG; force Hessian one HVP | `docs/manual/architecture.md:54`, `docs/manual/io.md:204` |
| 5 | Hessian decays with inter-site distance | Born-von-Karman force constants Φ(R) with acoustic sum rule | `postscf/phonons_supercell.py` |
| 5 | Analytic 1st derivative exact to 3.2e-10; forces 0.07 s | Hellmann-Feynman dE/dθ, force per step | `docs/ideas.md` "differentiable pseudopotential" line 353, performance.md relax case |
| 5 | Per-eval 0.3-97 s / 84 min | delta-gauge s/vol; FePt full SCF, MAE 11 min/direction | `docs/manual/performance.md`, `docs/ideas.md` MAE section |
| 5 | Rank-deficient single-observable loss, sv 1.7e-2..1.5e-7, cond 1.1e5 | EOS-only pseudopotential-correction Jacobian SVD | `docs/ideas.md` lines 356-362 |
| 5 | 1e-5 to 1e-7 floor for 1st derivatives | full SCF re-run FD floor | `docs/manual/wisdom.md:429` |
| 6 | 5-20 similar instances, scalar sweep, warm-start chain | EOS volume scan, spin-spiral angle sweep, model-parameter scan, `start_from` chain | `postscf/eos.py`, `examples/fe_spin_spiral.py`, `docs/ideas.md` "Batched multi-structure SCF" |
| 6 | Zero-raggedness vs ragged sweep | angle sweep (fixed cell/box) vs EOS (cell and FFT box vary with volume) | `docs/ideas.md` lines 1547-1563 |
| 6 | 2.6 of 6 GB, two fit; 25 W of 60-80 W | fcc Pt EOS single point on RTX 3050 | `docs/ideas.md` lines 1522-1533 |
| 6 | Concurrency 1.5-1.8×; batched wide solve = one big GEMM | MPS / batched multi-structure SCF | `docs/ideas.md` "Batched multi-structure SCF" |
| 6 | Warm-start helps EOS, hurts band-path (2.5× slower) | EOS density reuse vs near-degenerate band-path seeds | `docs/manual/performance.md` "Warm-start SCF" |
