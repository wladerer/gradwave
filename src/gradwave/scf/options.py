"""SCF option objects (refactor stage 1).

`scf_uspp` accumulated ~20 keyword parameters, each individually
justified and collectively unreadable. The options are grouped by what
they control, frozen (an SCF run's configuration is immutable), and
constructed from plain kwargs for backward compatibility, so
`scf_uspp(system, xc, etol=1e-9, mixing_scheme="johnson")` keeps working
while `scf_uspp(system, xc, opts=SCFOptions(...))` becomes the readable
form. Defaults here are THE defaults, the loop reads only these objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass(frozen=True)
class MixerOptions:
    scheme: str = "pulay"  # pulay | broyden | johnson
    alpha: float = 0.7
    history: int | None = None  # None → per-scheme default (johnson 12, else 8)
    kerker: bool | None = None  # None → on for smeared systems
    metric: str = "plain"  # plain | coulomb (johnson only)
    w0: float = 0.01  # johnson regularization
    trust_factor: float = 20.0
    adapt_step: bool = False  # opt-in collapse protection (see lessons.md)
    spin_precond: bool = False  # Stoner m-channel preconditioner
    bec_step_scale: float = 0.4


@dataclass(frozen=True)
class SCFOptions:
    smearing: str = "none"
    width: float = 0.1
    max_iter: int = 60
    etol: float = 1e-8
    rhotol: float = 1e-7
    diago_tol: float = 1e-9
    criterion: str = "drho"  # drho | energy
    rho_safety: float = 1e-2
    batched: bool = True
    verbose: bool = True
    mixer: MixerOptions = field(default_factory=MixerOptions)

    @classmethod
    def from_kwargs(cls, **kw) -> "SCFOptions":
        """Build from flat legacy kwargs (mixing_alpha=..., etc.).
        Unknown keys raise, misspelled tolerances must not pass silently."""
        rename = {
            "mixing_alpha": "alpha", "mixing_history": "history",
            "mixing_scheme": "scheme", "mixing_kerker": "kerker",
            "mixing_metric": "metric",
        }
        mix_names = {f.name for f in fields(MixerOptions)}
        scf_names = {f.name for f in fields(SCFOptions)} - {"mixer"}
        mix_kw, scf_kw = {}, {}
        for key, val in kw.items():
            name = rename.get(key, key)
            if name in mix_names:
                mix_kw[name] = val
            elif name in scf_names:
                scf_kw[name] = val
            else:
                raise TypeError(f"unknown SCF option {key!r}")
        return cls(mixer=MixerOptions(**mix_kw), **scf_kw)
