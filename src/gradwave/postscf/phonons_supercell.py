"""Supercell finite-displacement phonons — dispersion and DOS on a q-mesh.

The frozen-phonon (phonopy-style) method: build an N×N×N supercell of the
primitive cell, displace atoms, collect the real-space force constants
Φ_μν(R) = ∂²E/∂τ_{0μ}∂τ_{Rν} from the ground-state forces, Fourier-interpolate
to the dynamical matrix D(q) = Σ_R Φ_μν(R)/√(M_μ M_ν) · e^{iq·R}, and diagonalize
at any q. Unlike the analytic Γ path (`postscf.phonons`, DFPT-scoped), this needs
only ground-state forces (`postscf.forces`), so it runs for any q on a plain SCF.

Key cost point: because the force constants have the periodicity of the
PRIMITIVE lattice, only the N_prim atoms of the home cell need to be displaced
(their forces on every supercell image fill Φ_μν(R) for all R). So the SCF count
is 6·N_prim — INDEPENDENT of supercell size (Si 2×2×2 → 12 SCFs, not 96). This
is the Born–von-Kármán translational reduction; it is built into
`force_constants_home` by construction.

Units follow the package (eV, Å, amu); frequencies come out in cm⁻¹ (negative =
imaginary), sharing the exact conversion constant with the Γ path so the two
cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gradwave.postscf.phonons import _SQRT_EV_AMU_ANG2_TO_CM1


@dataclass(frozen=True)
class SupercellMap:
    """Deterministic primitive→supercell layout and the (μ, R) label of each
    supercell site. Site `s` is ordered as `s = t·N_prim + μ` where `t` indexes
    the translation cell (0,0,0) first, so the home-cell atoms are sites
    0…N_prim-1."""

    supercell: tuple            # (n1, n2, n3)
    cell_prim: np.ndarray       # (3,3) primitive cell, rows = lattice vectors [Å]
    cell_super: np.ndarray      # (3,3) supercell cell [Å]
    positions_super: np.ndarray  # (N_sc, 3) Cartesian [Å]
    species_super: list         # (N_sc,) species-of-atom indices
    mu_of_site: np.ndarray      # (N_sc,) int — primitive basis atom of each site
    rint_of_site: np.ndarray    # (N_sc, 3) int — integer lattice translation R
    n_prim: int
    n_sc: int

    @property
    def home_sites(self) -> np.ndarray:
        """Site indices of the home (R=0) cell, one per primitive atom."""
        return np.arange(self.n_prim)


def build_supercell(cell, positions, species_of_atom, supercell) -> SupercellMap:
    """Tile a primitive cell into a diagonal (n1,n2,n3) supercell, tracking the
    (primitive-atom μ, lattice-translation R) label of every supercell site.

    `cell` rows are the primitive lattice vectors [Å]; `positions` are Cartesian
    [Å]; `species_of_atom` are indices into the pseudopotential list.
    """
    cell = np.asarray(cell, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n1, n2, n3 = (int(x) for x in supercell)
    if min(n1, n2, n3) < 1:
        raise ValueError(f"supercell must be positive integers, got {supercell}")
    n_prim = len(positions)
    cell_super = np.diag([n1, n2, n3]) @ cell

    pos, spec, mu, rint = [], [], [], []
    for m0 in range(n1):
        for m1 in range(n2):
            for m2 in range(n3):
                r_cart = np.array([m0, m1, m2], dtype=float) @ cell
                for a in range(n_prim):
                    pos.append(positions[a] + r_cart)
                    spec.append(species_of_atom[a])
                    mu.append(a)
                    rint.append((m0, m1, m2))
    return SupercellMap(
        supercell=(n1, n2, n3), cell_prim=cell, cell_super=cell_super,
        positions_super=np.array(pos), species_super=spec,
        mu_of_site=np.array(mu, dtype=int), rint_of_site=np.array(rint, dtype=int),
        n_prim=n_prim, n_sc=n_prim * n1 * n2 * n3)


def _site_lookup(scmap: SupercellMap) -> dict:
    """(integer R, primitive atom μ) → supercell site index."""
    return {(tuple(scmap.rint_of_site[s]), int(scmap.mu_of_site[s])): s
            for s in range(scmap.n_sc)}


def symmetrize_force_constants(phi_home: np.ndarray,
                               scmap: SupercellMap) -> np.ndarray:
    """Enforce the physical symmetry Φ_μν(R) = Φ_νμ(−R)ᵀ, broken slightly by
    finite-difference noise, by averaging each block with its (−R) partner.
    Improves the acoustic modes and the accuracy of D(q) at every q."""
    n = np.array(scmap.supercell)
    look = _site_lookup(scmap)
    out = np.empty_like(phi_home)
    for s in range(scmap.n_sc):
        r = scmap.rint_of_site[s]
        nu = int(scmap.mu_of_site[s])
        neg_r = tuple((-r) % n)
        for mu in range(scmap.n_prim):
            s2 = look[(neg_r, mu)]
            out[mu, :, s, :] = 0.5 * (phi_home[mu, :, s, :]
                                      + phi_home[nu, :, s2, :].T)
    return out


def apply_acoustic_sum_rule(phi_home: np.ndarray) -> np.ndarray:
    """Enforce Σ_{s} Φ_home[μ,i,s,j] = 0 by correcting each μ's self block
    (the R=0, ν=μ term at site μ). Guarantees three exactly-zero acoustic modes
    at q=0. `phi_home` is (N_prim, 3, N_sc, 3)."""
    phi = phi_home.copy()
    n_prim = phi.shape[0]
    resid = phi.sum(axis=2)  # (N_prim, 3, 3) — Σ over supercell sites
    for a in range(n_prim):
        phi[a, :, a, :] -= resid[a]  # site a is the home-cell copy of atom a
    return phi


def force_constants_home(make_scf, scmap: SupercellMap, h: float = 0.01,
                         acoustic_sum_rule: bool = True, warm_start: bool = True,
                         verbose: bool = False) -> np.ndarray:
    """FD force constants Φ_home (N_prim, 3, N_sc, 3) [eV/Å²].

    Displaces ONLY the N_prim home-cell atoms (±h) and reads the analytic force
    on every supercell site — the translational reduction that makes the SCF
    count 6·N_prim regardless of supercell size. `make_scf(positions,
    start_from=None)` must return a converged SCF result; forces come from
    `postscf.forces`. Each displaced SCF warm-starts from the undisplaced
    reference when `warm_start`."""
    from gradwave.postscf.forces import forces

    pos0 = scmap.positions_super.copy()
    ref = make_scf(pos0) if warm_start else None
    n_prim, n_sc = scmap.n_prim, scmap.n_sc
    phi = np.zeros((n_prim, 3, n_sc, 3))
    for a in range(n_prim):  # home-cell atoms are sites 0…N_prim-1
        for i in range(3):
            pp, pm = pos0.copy(), pos0.copy()
            pp[a, i] += h
            pm[a, i] -= h
            fp = forces(make_scf(pp, start_from=ref)).detach().cpu().numpy()
            fm = forces(make_scf(pm, start_from=ref)).detach().cpu().numpy()
            # Φ_{(a,i),(s,j)} = −∂F_{s,j}/∂τ_{a,i}
            phi[a, i, :, :] = -(fp - fm) / (2.0 * h)
            if verbose:
                print(f"  displaced home atom {a} axis {i}", flush=True)
    phi = symmetrize_force_constants(phi, scmap)
    if acoustic_sum_rule:
        phi = apply_acoustic_sum_rule(phi)
    return phi


def phonon_dos(phi_home, scmap, masses_amu, q_mesh, weights,
               width: float = 6.0, npoints: int = 600):
    """Gaussian-broadened phonon DOS over a q-mesh. `width` in cm⁻¹. Returns
    (frequency_grid [cm⁻¹], dos)."""
    from gradwave.postscf.pdos import _broaden, spectral_grid

    freqs = dispersion(phi_home, scmap, masses_amu, q_mesh)  # (nq, 3N)
    e = freqs.reshape(-1)
    w = np.repeat(np.asarray(weights, dtype=float), freqs.shape[1])
    _window, grid = spectral_grid(e, width, npoints)
    return grid, _broaden(grid, e, w, width)


def dynamical_matrix(phi_home: np.ndarray, scmap: SupercellMap,
                     masses_amu, q_frac) -> np.ndarray:
    """Mass-weighted dynamical matrix D(q) (3N_prim, 3N_prim) complex-Hermitian.

    D_{μi,νj}(q) = (1/√(M_μ M_ν)) Σ_R Φ_{μν}(R)[i,j] e^{2πi q·R}, with the sum
    over every supercell site (each carries one (ν, R)). `q_frac` is in
    fractional coordinates of the primitive reciprocal lattice; `masses_amu` are
    the N_prim primitive-atom masses [amu]."""
    n = scmap.n_prim
    m = np.asarray(masses_amu, dtype=float)
    q = np.asarray(q_frac, dtype=float)
    d = np.zeros((n, 3, n, 3), dtype=complex)
    phases = np.exp(2j * np.pi * (scmap.rint_of_site @ q))  # (N_sc,)
    for s in range(scmap.n_sc):
        nu = scmap.mu_of_site[s]
        d[:, :, nu, :] += phi_home[:, :, s, :] * phases[s]
    msqrt = np.sqrt(m)
    d /= msqrt[:, None, None, None] * msqrt[None, None, :, None]
    d = d.reshape(3 * n, 3 * n)
    return 0.5 * (d + d.conj().T)


def frequencies_at_q(phi_home, scmap, masses_amu, q_frac) -> np.ndarray:
    """Phonon frequencies at one q [cm⁻¹], sorted; negative = imaginary."""
    d = dynamical_matrix(phi_home, scmap, masses_amu, q_frac)
    w2 = np.linalg.eigvalsh(d)  # real (Hermitian), ascending
    return np.sign(w2) * _SQRT_EV_AMU_ANG2_TO_CM1 * np.sqrt(np.abs(w2))


def dispersion(phi_home, scmap, masses_amu, q_frac_list) -> np.ndarray:
    """Frequencies [cm⁻¹] along a list of q-points → (n_q, 3N_prim)."""
    return np.array([frequencies_at_q(phi_home, scmap, masses_amu, q)
                     for q in q_frac_list])
