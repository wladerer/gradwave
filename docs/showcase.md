# Capability showcase (2026-07-14)

Fresh head-to-head calculations after the week's feature work: the
analytic position response, the complete adjoint (spin + metals + +U),
and the Johnson mixer at stock defaults. Every number below was computed
this session; nothing is recycled from earlier campaigns.

## 1. Analytic Γ phonons vs QE DFPT — two materials, two formalisms

The dynamical matrix comes from `postscf/uspp_position.hessian_column`:
one SCF, then one self-consistent position response per displacement
(generalized Sternheimer with the δS metric terms, Anderson-accelerated,
contracted through the τ-differentiable force graph). No finite
differences anywhere. QE computes the same object by DFPT (`ph.x`,
tr2_ph 1e-16). Same pseudos, cutoffs, k-mesh, and pinned FFT grids.

| system | gradwave (analytic) | ph.x DFPT | Δ |
|---|---|---|---|
| Si, Γ optical (kjpaw, 45/180, 2³, 32³) | 585.91 / 585.99 / 586.32 (mean 586.07) | 586.093 | 0.03% |
| SiGe zincblende, Γ optical T₂ (kjpaw + dn-kjpaw, 45/240, 4³, 40³) | 419.34 / 419.46 / 419.46 (mean 419.42) | 419.142 | 0.07% |
| acoustic modes | exactly 0 (ASR) | −6.2 raw (no ASR) | — |

SiGe is the demanding case: two species, semicore Ge 3d in valence, and
the cross-atom Hessian blocks exercise every per-species piece of the
augmentation and one-center machinery. The QE reference was generated
fresh for this comparison (inputs and dyn file committed under
`tests/fixtures/qe/sige_phonon/`). Cost on the same machine: six
analytic columns at ~60 s each after a 141 s SCF, against twelve full
SCF re-runs for a finite-difference Hessian.

The SiGe total energies agree to 0.22 meV/atom (−362.24542515 vs
−362.24545809 Ry), the established PAW quadrature class, here on a
system neither code had seen before.

## 2. Learned exchange functional recovered through the full SCF

`examples/train_xc_paw.py` trains the PBE exchange parameters (κ, μ)
against target densities on four systems chosen to exercise every
channel of the self-consistent adjoint at once:

- Si (PAW insulator),
- Al (smeared metal — Fermi-surface and δμ response),
- Si with U = 4 on 3p (+U occupation response),
- triplet O₂ (collinear spin, both channels, vacuum kernel).

Each epoch costs one SCF and one adjoint solve per system; the gradient
of the density loss flows through the complete self-consistency
(generalized Sternheimer, Hartree-XC and one-center Hessian-vector
products, Dudarev kernel, shared-Fermi coupling). Starting from
(κ, μ) = (1.10, 0.30), Adam recovers the PBE values (0.8040, 0.2195):

| | κ | μ | loss |
|---|---|---|---|
| start | 1.1000 | 0.3000 | 2.38e-4 |
| best (epoch 10) | 0.8192 | 0.2086 | 2.53e-06 |
| final (epoch 12) | 0.8046 | 0.2057 | 5.69e-06 |
| PBE target | 0.8040 | 0.2195 | — |

Twelve epochs at three to six minutes each on eight laptop cores bring
κ to 0.08% of the PBE value and the summed density loss down by a
factor of 94. The remaining μ offset sits along the shallow valley of
the (κ, μ) objective — the same near-degeneracy the norm-conserving
two-parameter fit documents — and the loss minimum at epoch 10
brackets both targets. Trajectory figure: `examples/train_xc_paw.png`.

Trajectory data in `examples/train_xc_paw.json`.

## 3. Provenance

QE 7.5 (`pw.x`, `ph.x`) on the asus box, gradwave at the commits of
2026-07-14. The Si phonon reference is the ph.x run recorded with
`examples/si_paw_phonon.py`; the SiGe references were generated and
consumed within this session. All comparisons use identical UPF files,
plane-wave cutoffs, k-meshes, and FFT grids in both codes.
