"""Result dataclasses shared by the USPP/PAW SCF drivers.

The four SCF drivers historically returned four shapes: ``scf`` an
``SCFResult``, ``scf_noncollinear`` an ``NCResult``, and the two USPP/PAW
drivers two structurally different plain dicts — so every consumer sniffed
which one it got. ``USPPResult`` / ``USPPNCResult`` give the dict paths a
first-class type whose field names match the old dict keys exactly, and all
four result types now carry an explicit ``formalism`` tag ("nc" |
"noncollinear" | "uspp" | "uspp_noncollinear") so downstream code dispatches
on an attribute rather than on isinstance checks.

``_DictBridge`` is TRANSITIONAL: it keeps every dict-style consumer
(``res["rho"]``, ``res.get("hub_sites")``, ``"hub_occ" in res``, ``dict(res)``)
working unchanged while call sites migrate to attribute access. Keys the old
dicts carried only conditionally (spin/Hubbard extras) stay absent from the
dict view while their field is None, so key-probing code sees exactly the old
shape. The ``formalism`` tag is not part of the dict view (the dicts had no
such key).

This module stays import-light (torch + core energies only) so both USPP
drivers can import it without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import torch

from gradwave.core.energies.total import EnergyBreakdown


class _DictBridge:
    """Transitional dict-style view of a result dataclass (see module docs)."""

    # fields mirroring keys the legacy dict carried only when applicable;
    # they read as absent while their value is None
    _conditional_keys = frozenset()

    def _present(self, key) -> bool:
        if key == "formalism" or key not in {f.name for f in fields(self)}:
            return False
        return not (key in self._conditional_keys
                    and getattr(self, key) is None)

    def keys(self):
        return [f.name for f in fields(self) if self._present(f.name)]

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key) -> bool:
        return self._present(key)

    def __getitem__(self, key):
        if not self._present(key):
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key) if self._present(key) else default


@dataclass
class USPPResult(_DictBridge):
    """Converged ``scf_uspp`` state (USPP/PAW, collinear, nspin=1 or 2).

    Field names match the keys of the plain dict this driver returned
    historically; dict-style reads keep working through ``_DictBridge``.
    """

    converged: bool
    n_iter: int
    energies: EnergyBreakdown
    eigenvalues: torch.Tensor  # (nk, nb) [eV]; (nspin, nk, nb) when nspin=2
    occupations: torch.Tensor  # (nk, nb); (nspin, nk, nb) when nspin=2
    coeffs: list  # [(nb, npw_k)] per k; list-of-lists [spin][k] when nspin=2
    rho: torch.Tensor  # TOTAL density (n1,n2,n3) [e/Å³]
    rho_ij_atoms: list  # becsum per atom; [spin][atom] when nspin=2
    becps: list  # ⟨β|ψ⟩ per k; [spin][k] when nspin=2
    history: list
    fermi: float | None
    system: object  # USPPSystem
    nspin: int
    smearing: str
    width: float
    mixer_mult: object  # mixer block multipliers (diagnostics)
    rho_out_spin: list  # RAW map output (pre-mixing) — rig/diagnostics
    hub_occ: list | None = None  # DFT+U per-spin occupation matrices [σ][site]
    hub_sites: list | None = None  # DFT+U site definitions
    rho_spin: list | None = None  # [ρ↑, ρ↓] when nspin=2
    mag_total: float | None = None  # ∫(ρ↑−ρ↓) dr [μB] when nspin=2
    mag_abs: float | None = None  # ∫|ρ↑−ρ↓| dr [μB] when nspin=2
    newton: list | None = None  # newton_polish per-step residual norms
    formalism: str = "uspp"

    _conditional_keys = frozenset(
        {"hub_occ", "hub_sites", "rho_spin", "mag_total", "mag_abs", "newton"})


@dataclass
class USPPNCResult(_DictBridge):
    """Converged ``scf_uspp_noncollinear`` state (USPP/PAW spinor).

    Field names match the keys of the plain dict this driver returned
    historically; dict-style reads keep working through ``_DictBridge``.
    """

    converged: bool
    n_iter: int
    energies: EnergyBreakdown
    fermi: float
    mag_vec: tuple  # ∫ m⃗ dr [μB]
    mag_abs: float  # ∫ |m⃗| dr [μB]
    rho: torch.Tensor
    m: torch.Tensor  # (3, *grid)
    eigenvalues: torch.Tensor  # (nk, nb)
    history: list = field(default_factory=list)
    rho_ij_chan: list | None = None  # becsum in the 4 (n, m⃗) channels
    coeffs: torch.Tensor | None = None  # (nk, nb, 2·npw_max) spinors
    formalism: str = "uspp_noncollinear"
