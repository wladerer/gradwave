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
from typing import TYPE_CHECKING, Any, cast

import torch

from gradwave.core.energies.total import EnergyBreakdown

if TYPE_CHECKING:
    # Annotation-only: keeps this module import-light at runtime (see module
    # docstring) even though there is no actual cycle (scf/uspp_setup.py's own
    # top-level imports never reach back into this module, nor do scf/loop.py's
    # or scf/noncollinear.py's).
    from _typeshed import DataclassInstance

    from gradwave.core.hubbard import HubbardSite
    from gradwave.scf.loop import SCFResult
    from gradwave.scf.noncollinear import NCResult
    from gradwave.scf.uspp_setup import USPPSystem


class _DictBridge:
    """Transitional dict-style view of a result dataclass (see module docs).

    ``_DictBridge`` itself is a plain mixin, not a dataclass -- ``fields()``
    needs a real ``DataclassInstance``, which only the ``@dataclass``
    subclasses (``USPPResult``/``USPPNCResult``) actually are, so ``self`` is
    cast at each call rather than widening ``fields()``'s own parameter type."""

    # fields mirroring keys the legacy dict carried only when applicable;
    # they read as absent while their value is None
    _conditional_keys = frozenset()
    # fields never part of the legacy dict shape — internal diagnostics that
    # dict-style consumers (and dict(res) / checkpoint) should not see even when
    # set. Kept out of the dict view unconditionally.
    _hidden_keys = frozenset({"recorder"})

    def _present(self, key) -> bool:
        self_dc = cast("DataclassInstance", self)
        if (key == "formalism" or key in self._hidden_keys
                or key not in {f.name for f in fields(self_dc)}):
            return False
        return not (key in self._conditional_keys
                    and getattr(self, key) is None)

    def keys(self):
        self_dc = cast("DataclassInstance", self)
        return [f.name for f in fields(self_dc) if self._present(f.name)]

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
    # [(nb, npw_k)] per k (nspin=1); list-of-lists [spin][k] when nspin=2 --
    # scf_uspp collapses the per-spin wrapper away at nspin=1 (`coeffs[0]`).
    coeffs: list[torch.Tensor] | list[list[torch.Tensor]]
    rho: torch.Tensor  # TOTAL density (n1,n2,n3) [e/Å³]
    rho_ij_atoms: list[torch.Tensor] | list[list[torch.Tensor]]  # per atom; [spin][atom] w/ nspin=2
    becps: list[torch.Tensor] | list[list[torch.Tensor]]  # ⟨β|ψ⟩ per k; [spin][k] when nspin=2
    history: list[dict[str, int | float]]  # common.record_iteration's per-iter schema
    fermi: float | None
    system: USPPSystem
    nspin: int
    smearing: str
    width: float
    mixer_mult: dict[int, float] | None  # mixer block multipliers (diagnostics)
    rho_out_spin: list[torch.Tensor]  # RAW map output (pre-mixing) — rig/diagnostics
    hub_occ: list[list[torch.Tensor]] | None = None  # DFT+U per-spin occupation matrices [σ][site]
    hub_sites: list[HubbardSite] | None = None  # DFT+U site definitions
    rho_spin: list[torch.Tensor] | None = None  # [ρ↑, ρ↓] when nspin=2
    mag_total: float | None = None  # ∫(ρ↑−ρ↓) dr [μB] when nspin=2
    mag_abs: float | None = None  # ∫|ρ↑−ρ↓| dr [μB] when nspin=2
    newton: list[float] | None = None  # newton_polish per-step residual norms
    recorder: Any = None  # scf.recorder.SCFRecorder — per-iteration flight recorder
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
    mag_vec: tuple[float, ...]  # ∫ m⃗ dr [μB], 3 components
    mag_abs: float  # ∫ |m⃗| dr [μB]
    rho: torch.Tensor
    m: torch.Tensor  # (3, *grid)
    eigenvalues: torch.Tensor  # (nk, nb)
    system: USPPSystem
    history: list[dict[str, int | float]] = field(default_factory=list)
    rho_ij_chan: list[list[torch.Tensor]] | None = None  # becsum in the 4 (n, m⃗) channels
    coeffs: torch.Tensor | None = None  # (nk, nb, 2·npw_max) spinors
    formalism: str = "uspp_noncollinear"


if TYPE_CHECKING:
    # The four SCF drivers' result shapes, as accepted by postscf/ entry points
    # that dispatch on `res.formalism`/`isinstance` across all of them (forces,
    # stress, discretization_error, pdos, cohp, bader, ...) rather than being
    # tied to one formalism. Not every "res"-taking postscf function accepts
    # every member of this union (e.g. forces() is SCFResult | NCResult only,
    # no USPP path) — this is the maximal shared vocabulary, not a claim that
    # every function takes all four. TYPE_CHECKING-only like SCFResult/NCResult
    # above, for the same import-lightness reason.
    AnyResult = SCFResult | USPPResult | NCResult | USPPNCResult
