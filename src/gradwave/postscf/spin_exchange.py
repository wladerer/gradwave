"""Spin-Hamiltonian parameters (Heisenberg J, Dzyaloshinskii-Moriya D, anisotropic
exchange) from the AD constrained-moment torque.

The classical spin model behind atomistic spin dynamics and micromagnetics is

    H = - Σ_{I<J} ê_I · 𝒥_IJ · ê_J  -  Σ_I K_I (ê_I·n̂)²                     (1)

with ê_I unit moment directions and 𝒥_IJ a 3×3 exchange tensor. Its parts are the
physics people want:

    isotropic (Heisenberg)   J_IJ   = ⅓ Tr 𝒥_IJ
    antisymmetric (DMI)      D_IJ   from  ê_I·A·ê_J = D_IJ·(ê_I×ê_J)
    symmetric traceless      Γ_IJ   two-ion (anisotropic) exchange

DFT's job is to parametrize (1). The distinctive thing here is *how*: the torque
T_I = -dW/dê_I is an autograd-exact first derivative of the constrained-DFT energy
(`constrained_moment_scf`, validated to a finite difference at ratio 1.000), so the
exchange tensor is its site-to-site derivative

    𝒥_IJ^{ab} = -∂²W/∂ê_I^a ∂ê_J^b = ∂T_I^a/∂ê_J^b .                       (2)

We get (2) by tilting moment J by a small angle at a collinear reference and reading
the induced *transverse* torque on every other moment I. Because the torque is
already an exact gradient, this is a *single* finite-difference order of an analytic
derivative — not the double energy-difference of conventional energy mapping — so it
carries less noise for the same displacement. Conventional codes instead map many
total energies (fragile) or run a separate Green's-function (LKAG) machinery.

Scope and caveats
-----------------
* **Heisenberg J needs no SOC.** DMI and single-ion anisotropy K need a
  fully-relativistic (SOC) pseudopotential; DMI additionally needs broken inversion
  symmetry (an interface or non-centrosymmetric cell) — it vanishes by symmetry in
  bcc Fe.
* **A small cell folds periodic images together.** Tilting "atom J" tilts all of its
  periodic images at once, so the extracted coupling is the *shell-summed*
  J(q=0)-like quantity, not an individual-shell J_n. Individual shells J₁, J₂, …
  need either a supercell where the shells are distinct atoms, or the reciprocal-
  space route (J(q) from the spin-spiral energy / torque on a q-mesh, Fourier-
  transformed to J(R)). The mean-field Curie temperature k_B T_c = ⅔ Σ_J J_IJ only
  needs the shell sum, so it is the robust small-cell benchmark.
* **The full D vector needs three reference orientations.** One collinear reference
  along n̂ gives the DMI component D·n̂ (the antisymmetric torque in the plane ⊥ n̂);
  repeat along x̂, ŷ, ẑ for the whole vector.
* The fully-analytic route — differentiating the torque through the SCF fixed point
  for 𝒥 without any finite step — is future work (see docs/ideas.md).

Convention: (1) uses unit ê, so J is returned in eV as the energy curvature
d²W/dθ². Comparing to a reference that folds the spin magnitude S into J (many quote
J·S² or J in mRy) needs the matching factor; `heisenberg_couplings` documents this
at the call site.
"""

from __future__ import annotations

import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.postscf.moment_config import atomic_weights, constrained_moment_scf


def _unit(v):
    return v / torch.linalg.norm(v)


def _transverse_basis(ref):
    """Two orthonormal vectors (u, v) spanning the plane ⊥ ref, with u×v = ref, so a
    small tilt lives in span(u, v) and the antisymmetric response maps to D·ref."""
    ref = _unit(ref)
    seed = torch.tensor([1.0, 0.0, 0.0], dtype=ref.dtype, device=ref.device)
    if abs(float(torch.dot(ref, seed))) > 0.9:
        seed = torch.tensor([0.0, 1.0, 0.0], dtype=ref.dtype, device=ref.device)
    u = _unit(seed - torch.dot(seed, ref) * ref)
    v = torch.linalg.cross(ref, u)
    return u, v


def exchange_from_atom(system, xc: NoncollinearXC, j: int, *, m0,
                       ref_dir=(0.0, 0.0, 1.0), delta: float = 0.08, lam: float = 8.0,
                       weights=None, mode: str = "vector", **scf_kwargs):
    """Site-to-site exchange tensors from tilting one moment.

    Hold every moment collinear along `ref_dir`, then tilt moment `j` by `delta`
    along each of the two transverse axes (two constrained SCFs) and read the induced
    transverse torque on all other atoms — the derivative in eq. (2). One call yields
    the 2×2 exchange tensor 𝒥_IJ (in the plane ⊥ ref, basis (u, v)) for *every*
    I ≠ j at once.

    `m0` is the per-atom target magnitude [μB] for the magnitude-robust `vector`
    penalty (e.g. the ferromagnetic |M|); holding the moment fixed is what makes the
    tilt a pure rotation. Returns

        {i: J_tensor_2x2 (torch, eV)},  basis (u, v)

    The reference (unperturbed) torque is subtracted, so a residual off-ground-state
    torque cancels to first order.
    """
    dev = system.positions.device
    ref = _unit(torch.as_tensor(ref_dir, dtype=torch.float64, device=dev))
    u, v = _transverse_basis(ref)
    if weights is None:
        weights = atomic_weights(system)
    na = len(system.species_of_atom)
    m0 = torch.as_tensor(m0, dtype=torch.float64, device=dev)
    if m0.ndim == 0:
        m0 = m0.expand(na)

    def torques(dirs):
        _, info = constrained_moment_scf(system, xc, dirs, lam=lam, weights=weights,
                                         mode=mode, target_mag=m0, **scf_kwargs)
        return info["torque"]                       # (na, 3), descent torque -dW/dê

    ref_dirs = ref.repeat(na, 1)
    t0 = torques(ref_dirs)
    cols = []
    for beta in (u, v):
        d = ref_dirs.clone()
        d[j] = _unit(ref + delta * beta)
        cols.append((torques(d) - t0) / delta)      # ∂T/∂θ_j along beta, (na,3)
    du, dv = cols                                    # response to tilting j along u, v

    out = {}
    for i in range(na):
        if i == j:
            continue
        # J[a][b] = (∂T_i along a) when j tilted along b  → project onto (u, v)
        J = torch.tensor([[float(du[i] @ u), float(dv[i] @ u)],
                          [float(du[i] @ v), float(dv[i] @ v)]], dtype=torch.float64)
        out[i] = J
    return out, (u, v)


def decompose(J_tensor):
    """Split a 2×2 transverse exchange tensor 𝒥_IJ into (J_heisenberg, D_ref, Γ):
    the isotropic Heisenberg scalar [eV], the DMI component along the reference axis
    [eV] (the full vector needs three references), and the 2×2 symmetric-traceless
    anisotropic-exchange part [eV]."""
    J = J_tensor
    J_iso = 0.5 * float(J[0, 0] + J[1, 1])
    D_ref = 0.5 * float(J[0, 1] - J[1, 0])
    sym = 0.5 * (J + J.T)
    gamma = sym - 0.5 * float(J[0, 0] + J[1, 1]) * torch.eye(2, dtype=J.dtype)
    return J_iso, D_ref, gamma


def heisenberg_couplings(system, xc: NoncollinearXC, j: int, *, m0, **kwargs):
    """Convenience: isotropic Heisenberg J_ij [eV] to every other atom, from one
    tilt of atom `j`. Sign convention of eq. (1): J > 0 is ferromagnetic. To compare
    with references quoted as J·S² or in mRy, apply the matching magnitude/unit
    factor (1 mRy = 13.6057 meV; S = M/2 in μ_B if the reference folds S into J)."""
    tensors, _ = exchange_from_atom(system, xc, j, m0=m0, **kwargs)
    return {i: decompose(J)[0] for i, J in tensors.items()}
