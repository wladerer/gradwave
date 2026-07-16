# Plan: density of states and projections

Total, spin-projected, atom- and orbital-projected DOS, and noncollinear
projections (charge, spin texture, and j-resolved SOC). The reference for every
projected quantity is Quantum ESPRESSO's `projwfc.x`.

## What already exists

The two hard ingredients are in the tree for other reasons, so most of this is
reuse rather than new machinery.

- **Binning.** `analysis.dos_frame` gaussian-broadens eigenvalues with k-weights.
  A projected DOS is the same binner with per-state weights instead of ones.
- **Atomic orbitals.** The UPF parser reads `PP_PSWFC` into `upf.pswfc`
  (`AtomicOrbital`), grouped by `l` with `wfc_with_l`. Fully-relativistic pseudos
  also carry `jchi`.
- **Projection primitive.** `core/hubbard.py` builds atomic orbitals on the
  plane-wave sphere (`build_hubbard_projectors`) and projects Kohn-Sham states
  onto them (`becp = <phi_p|psi_kb>` in `occupation_matrices`, S-dressed for
  USPP). A projected DOS keeps `|becp|^2` per band instead of summing into
  occupation matrices.
- **Noncollinear.** `scf_noncollinear` stores spinor coefficients
  `(nk, nb, 2*npw_max)` with `[:npw]` the up component and `[npw:]` the down, and
  already decomposes the spin density into `n, m_x, m_y, m_z`. Per-component
  projection (`bu`, `bd`) is in the density build.
- **SOC projectors.** `core/spinor_proj.py` `build_so_projectors` gives
  `(l, j, mj)` spinor projectors on the doubled axis.
- **nscf mesh.** `bands` / `bands_uspp` solve at arbitrary k on the frozen
  potential, which is the dense mesh a smooth DOS wants.

## Architecture

One compute module `postscf/pdos.py`, thin frame and plot wrappers in
`analysis.py`, an output section in `output.py`, and a JSON block. A single
weight tensor drives every variant,

    W[state, group] = projected weight of a KS state onto an AO group,

where `group` aggregates by atom, `l`, `lm`, spin channel, or `(l, j, mj)`.
Every DOS is then `bin_by_energy(eigs, W[:, group], kweights, broadening)`. One
code path, and spin, orbital, and atom become groupings of the same
projections.

## Phases

### Phase 0, nscf plumbing

Return eigenvectors from the band solver at a denser k-mesh on the frozen
converged potential, reusing `bands` / `bands_uspp`. A DOS on the SCF mesh is
coarse. Add Marzari-Vanderbilt and tetrahedron broadening to the binner
alongside the existing gaussian.

### Phase 1, spin-projected DOS (collinear nspin=2)

Bin each spin channel separately, plot the down channel negative. Nearly free
because nspin=2 already stores per-spin eigenvalues. Ships first with a test and
a `plot_dos` spin mode.

### Phase 2, atomic-orbital basis and projection core (collinear)

Generalize `build_hubbard_projectors` from the +U manifold to every `pswfc`
orbital of every atom. Add Löwdin orthogonalization: build the AO overlap
`O_ij = <phi_i|phi_j>` (`<phi_i|S|phi_j>` for USPP), form `O^{-1/2}`, and report
the spilling parameter (the KS weight the atomic basis fails to capture) as the
honesty metric. Return per-band `|<phi_nlm|psi_nk>|^2`.

### Phase 3, PDOS and LDOS (collinear, NC and USPP)

Aggregate the Phase 2 weights by atom (LDOS), `(atom, l)`, and `(atom, l, m)`,
and by spin for nspin=2. Validate against `projwfc.x` on a psl pseudo, comparing
per-atom l-curves and the spilling.

### Phase 4, noncollinear projections

Project each spinor component onto the scalar AO, `b_up = <phi|psi_up>` and
`b_dn = <phi|psi_dn>` (the `bu` and `bd` already in the density build), then

- charge PDOS `= |b_up|^2 + |b_dn|^2`,
- spin texture by the same Pauli decomposition the density uses,
  `m_z = |b_up|^2 - |b_dn|^2`, `m_x = 2 Re(b_up* b_dn)`,
  `m_y = 2 Im(b_up* b_dn)`, and projection on an arbitrary axis `n·m`.

Magnetic noncollinear breaks the crystal symmetry, so it needs the full k-mesh
(`scf_noncollinear` already refuses the density symmetrizer unless
`nonmagnetic=True`). Validate against QE noncollinear `projwfc.x`.

### Phase 5, SOC, fully-relativistic j-resolved

Swap the scalar AO projector for the `build_so_projectors` spinor projectors to
get `(l, j, mj)` PDOS on the doubled axis, `b_so = <phi_ljmj|psi>`, which is how
QE labels PDOS with `lspinorb`. Budget time for matching the orbital ordering
and normalization conventions.

### Phase 6, output, analysis, plotting

`pdos_frame` / `plot_pdos` in `analysis.py`, a projected-DOS section in
`output.py`, a JSON block, and `gradwave plot out/pdos.json --kind pdos` with a
selectable grouping (atom, l, lm, spin, j).

## Caveats

- `PP_PSWFC` must be present. SG15 ONCV pseudos ship no atomic orbitals, so those
  give total DOS but no PDOS. PseudoDojo and the psl kjpaw/rrkjus sets have them.
  Detect and error clearly, the same limitation QE has.
- USPP and PAW need the S-metric in both the projection and the Löwdin overlap.
  The Hubbard path already does this, so it is reuse, but it has to be threaded
  through or the spilling comes out wrong.
- Löwdin conditioning. Near-linearly-dependent AO sets make `O^{-1/2}`
  ill-conditioned. Floor the overlap eigenvalues and report the floor rather than
  returning silent garbage.
- FR orbital conventions. The `(l, j, mj)` normalization and ordering must match
  QE's for a clean comparison.

## Validation

Internal consistency first, since it needs no external code: the summed PDOS
reproduces the total DOS, the spin channels sum to the total, and the spilling is
small for a complete AO set. Then `projwfc.x` per phase for the quantitative
comparison.
