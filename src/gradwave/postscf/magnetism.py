"""One-call magnetic characterization: the important quantities and qualities of a
magnetic system in a single routine.

`characterize_magnetism` runs a non-collinear reference SCF for the atomic moments,
then (optionally) extracts the spin-Hamiltonian couplings from the autograd torque
and folds everything into a `MagneticReport`:

    quantities   per-atom moment magnitudes and vectors, total moment, Heisenberg
                 exchange J, DMI vector D, anisotropic exchange, mean-field T_c
    qualities    magnetic ordering (ferro / antiferro / non-collinear / nonmagnetic),
                 the sign/scale of the dominant coupling

It composes the pieces built in `moment_config` (constrained SCF, weights) and
`spin_exchange` (J/D/K from ∂T/∂ê) behind one entry point so a caller does not have
to wire them together. DMI and single-ion anisotropy K need a fully-relativistic
(SOC) pseudo; without one the DMI channel is reported as ~0 (as symmetry requires)
and K is left for the MAE routine (see docs/ideas.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.postscf.moment_config import atomic_weights
from gradwave.postscf.spin_exchange import decompose, exchange_from_atom
from gradwave.scf.noncollinear import scf_noncollinear

KB = 8.617333e-5  # eV/K


@dataclass
class MagneticReport:
    """Structured result of `characterize_magnetism`. Energies in eV, moments in μB,
    temperature in K. `exchange_J`/`dmi` are keyed by neighbor atom index and are
    None when the exchange step was skipped."""

    moment_magnitudes: list       # |M_I| per atom
    moment_vectors: list          # M_I = (Mx, My, Mz) per atom
    total_moment: float           # |Σ_I M_I|
    ordering: str
    exchange_J: dict | None       # {i: J_0i}  isotropic Heisenberg [eV], from ref atom
    dmi: dict | None              # {i: D_0i·n̂}  DMI along the reference axis [eV]
    curie_temperature_mfa: float | None
    ref_atom: int

    def summary(self) -> str:
        lines = [f"Magnetic ordering : {self.ordering}",
                 f"Total moment      : {self.total_moment:.3f} μB",
                 "Atomic moments    : " + ", ".join(
                     f"{m:.3f}" for m in self.moment_magnitudes) + " μB"]
        if self.exchange_J is not None:
            js = ", ".join(f"J_{self.ref_atom}{i} = {J*1000:+.1f}"
                           for i, J in self.exchange_J.items())
            lines.append(f"Exchange [meV]    : {js}")
            if any(abs(d) > 1e-6 for d in (self.dmi or {}).values()):
                ds = ", ".join(f"D_{self.ref_atom}{i}·n̂ = {d*1000:+.3f}"
                               for i, d in self.dmi.items())
                lines.append(f"DMI [meV]         : {ds}")
            if self.curie_temperature_mfa is not None:
                lines.append(f"T_c (mean field)  : {self.curie_temperature_mfa:.0f} K")
        return "\n".join(lines)


def _atomic_moment_vectors(system, m, weights):
    cf = system.grid.volume / system.grid.n_points
    return torch.einsum("axyz,ixyz->ai", weights, m) * cf     # (na, 3) [μB]


def _classify(moment_vectors, mags, exchange_J, mag_tol=0.15):
    if float(max(mags)) < mag_tol:
        return "nonmagnetic"
    if exchange_J is not None and len(exchange_J):
        Jdom = max(exchange_J.values(), key=abs)
        return "ferromagnetic" if Jdom > 0 else "antiferromagnetic"
    # no exchange: infer from the reference moment directions
    mv = [v for v, g in zip(moment_vectors, mags, strict=True) if float(g) > mag_tol]
    ref = mv[0] / torch.linalg.norm(mv[0])
    cosths = [float(torch.dot(v / torch.linalg.norm(v), ref)) for v in mv]
    if all(c > 0.9 for c in cosths):
        return "ferromagnetic (seeded, unverified — run exchange to confirm)"
    if any(c < -0.9 for c in cosths):
        return "antiferromagnetic (seeded)"
    return "non-collinear"


def characterize_magnetism(system, xc: NoncollinearXC, *, seed_scale: float = 1.5,
                           exchange: bool = True, ref_atom: int = 0, lam: float = 8.0,
                           delta: float = 0.08, weights=None, **scf_kwargs):
    """Characterize the magnetism of `system` in one call. Returns a
    `MagneticReport`.

    A non-collinear reference SCF (seeded high-spin along +z, since the bare
    non-collinear SCF is multi-stable) fixes the atomic moments. With
    `exchange=True` the spin-Hamiltonian couplings are extracted by tilting the
    moment on `ref_atom` and reading the induced torque on the others
    (`spin_exchange`), giving the isotropic Heisenberg J, the DMI component along the
    reference axis, and a nearest-neighbor mean-field Curie temperature
    k_B T_c = ⅔ Σ_i J_{ref,i}. Exchange adds ~3 constrained SCFs; set exchange=False
    for a cheap moments-and-ordering pass. Extra scf_kwargs pass through to the SCFs.
    """
    dev = system.positions.device
    na = len(system.species_of_atom)
    if weights is None:
        weights = atomic_weights(system)

    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=dev)
    res = scf_noncollinear(system, xc, mag_vec_init=(seed_scale * z).repeat(na, 1).tolist(),
                           **scf_kwargs)
    M = _atomic_moment_vectors(system, res.m, weights)         # (na, 3)
    mags = torch.linalg.norm(M, dim=-1)
    total = float(torch.linalg.norm(M.sum(0)))

    exchange_J = dmi = tc = None
    if exchange and float(mags.max()) >= 0.15:
        tensors, _ = exchange_from_atom(system, xc, j=ref_atom, m0=mags,
                                        ref_dir=(0, 0, 1), delta=delta, lam=lam,
                                        weights=weights, **scf_kwargs)
        exchange_J, dmi = {}, {}
        for i, Jt in tensors.items():
            J_iso, D_ref, _ = decompose(Jt)
            exchange_J[i], dmi[i] = J_iso, D_ref
        tc = (2.0 / 3.0) * sum(exchange_J.values()) / KB

    ordering = _classify(M, mags, exchange_J)
    return MagneticReport(
        moment_magnitudes=[round(float(x), 3) for x in mags],
        moment_vectors=[[round(float(c), 3) for c in v] for v in M],
        total_moment=total, ordering=ordering, exchange_J=exchange_J, dmi=dmi,
        curie_temperature_mfa=tc, ref_atom=ref_atom)
