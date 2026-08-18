"""SCF flight recorder — per-outer-iteration convergence diagnostics.

Two jobs: (a) a debugging aid for stuck SCFs, and (b) a training corpus for
learned mixers. Cheap scalar/vector metrics are collected once per SCF
iteration (never per Davidson step — the hot inner loop stays untouched),
summarized into the run output, and optionally dumped as a per-iteration
sidecar trace. Everything stored is a detached CPU scalar or a small vector,
so the recorder is provably off the differentiable path: the SCF drivers run
under ``torch.no_grad`` and the recorder only ever reads already-detached
state.

The three drivers that record are the collinear norm-conserving loop
(``scf.loop.scf``) and the collinear USPP/PAW loop (``scf.uspp_loop.scf_uspp``).
The noncollinear/spinor drivers are OUT OF SCOPE — their magnetization is a
vector field, so the moment/shell heuristics below do not transfer.

Metrics per iteration:
- residual norm (reused from the loop, not recomputed);
- residual decomposed onto ``n_shells`` |G|-shells (one extra r→G FFT of the
  total real-space residual, binned by |G|);
- eigenvalue drift (max |Δε| vs the previous iteration) and a band-reordering
  count (nearest-neighbour index matching of the current to previous
  eigenvalues, per k and spin);
- Fermi level and the smearing entropy term (reused);
- absolute magnetization ∫|ρ↑−ρ↓| for nspin=2 (reused/cheap);
- wall time (reused from the loop's own per-iteration timing).

``summarize()`` gives the compact diagnostics block; ``diagnose()`` gives 0-3
heuristic tags with a one-line reason each; ``to_trace_dict()`` gives the
schema-versioned per-iteration arrays for the sidecar. The thresholds in
``diagnose()`` are heuristics tuned on the test systems — see the inline notes.
"""

from __future__ import annotations

from typing import Any

import torch

from gradwave.core.fftbox import r_to_g
from gradwave.dtypes import CDTYPE

# sidecar schema version (bump on any incompatible field change)
TRACE_VERSION = 1

# diagnose() guard thresholds (see the diagnose() docstring for the justification).
# Both exist to stop a *converged* run's settled tail from tripping a tag: the
# last-5-iteration window that diagnose() reads is the pathological phase for a
# stuck run but the noise floor for a converged one.
#
# _SLOSH_RES_FLOOR_: charge-sloshing only means something while the residual is
# still being driven down. Once |Δρ| has fallen below this, the |G|-shell
# decomposition is dominated by roundoff-level noise and its (non-)monotonicity
# is meaningless, so the sloshing fraction must not be read there. 1e-4 sits
# above the occupation-noise floor of a smeared metal (~1e-5..1e-6) and well
# below any genuinely stuck residual.
_SLOSH_RES_FLOOR = 1e-4
# _MOMENT_COLLAPSE_FLOOR_ (μB): a collapse is a fall to the *nonmagnetic* branch,
# so the final moment must be genuinely near zero, not merely a large fraction
# below the seed. The seed moment on the USPP/PAW path is the smooth-grid moment
# of the atomic (SAD) seed, which for a strongly-seeded 3d metal is many times
# the converged smooth moment (e.g. fcc Ni: seed ~10.8 μB vs converged ~0.79 μB
# smooth), so "fell to <10% of the seed" alone false-positives on every healthy
# FM PAW metal. The smallest held FM moment in the test battery (fcc Ni, ~0.79 μB
# on the smooth grid) sits ~8x above this floor, while a Stoner collapse drives
# the moment below occupation noise (<0.05 μB).
_MOMENT_COLLAPSE_FLOOR = 0.1


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class SCFRecorder:
    """Collects per-iteration SCF diagnostics (detached CPU scalars/vectors).

    Construct once before the SCF loop with the real-grid ``g2`` (|G|²), call
    ``record()`` once per outer iteration, then read ``summarize()`` /
    ``diagnose()`` / ``to_trace_dict()`` after the loop. ``mixing`` is set once
    per run via ``set_mixing()``.
    """

    def __init__(
        self,
        g2: torch.Tensor,
        *,
        nspin: int = 1,
        seed_moment: float = 0.0,
        n_shells: int = 12,
    ) -> None:
        self.nspin = int(nspin)
        self.seed_moment = float(abs(seed_moment))
        self.n_shells = int(n_shells)
        self.iters: list[dict[str, Any]] = []
        self.mixing: dict[str, Any] = {}
        # |G|-shell binning over the flat dense grid, built once. Linear bins in
        # |G| from 0 to |G|max; the lowest bins are the long-wavelength modes the
        # sloshing heuristic keys on. Kept on the grid's device so binning stays
        # a single device-side index_add per iteration.
        gmag = g2.detach().reshape(-1).clamp_min(0.0).sqrt()
        gmax = float(gmag.max())
        edges = torch.linspace(
            0.0, gmax if gmax > 0 else 1.0, self.n_shells + 1, device=gmag.device
        )
        # bucketize → 1..n_shells; shift to 0..n_shells-1 and clamp the top edge
        idx = torch.bucketize(gmag, edges, right=False) - 1
        self._bin_idx = idx.clamp_(0, self.n_shells - 1).to(torch.long)
        self.shell_edges = [float(x) for x in edges.cpu().tolist()]
        self._prev_eig: torch.Tensor | None = None

    def set_mixing(
        self,
        *,
        scheme: str,
        alpha: float,
        kerker: bool,
        history: int,
    ) -> None:
        """Record the mixing configuration once per run (run-level metadata)."""
        self.mixing = {
            "scheme": str(scheme),
            "alpha": float(alpha),
            "kerker": bool(kerker),
            "history": int(history),
        }

    def _shell_fractions(self, drho_r: torch.Tensor) -> list[float]:
        """|Δρ(G)|² binned onto the |G|-shells, normalized to a fraction that
        sums to 1. One extra r→G FFT of the (already detached) real residual."""
        drho_g = r_to_g(drho_r.detach().to(CDTYPE)).reshape(-1)
        power = (drho_g.real**2 + drho_g.imag**2).to(torch.float64)
        shell = torch.zeros(self.n_shells, dtype=torch.float64, device=power.device)
        shell.index_add_(0, self._bin_idx, power)
        total = float(shell.sum())
        if total <= 0.0:
            return [0.0] * self.n_shells
        return [float(x) / total for x in shell.cpu().tolist()]

    # eigenvalues within this window (eV) are treated as degenerate for the
    # reordering count, so a uniform shift of a degenerate manifold (or a tie in
    # the nearest-neighbour match) does not read as a spurious crossing.
    _DEGEN_TOL = 1e-2

    def _eig_drift_and_reorder(self, eigs: torch.Tensor) -> tuple[float | None, int]:
        """(max |Δε| vs previous, band-reordering count) for a stacked
        (nspin, nk, nb) eigenvalue tensor. Reordering: for each (spin, k), match
        each current eigenvalue to its nearest previous one; count bands whose
        nearest-previous index differs from their own — a crossing signature —
        excluding matches within a degeneracy window (uniform shifts of a
        degenerate manifold are not genuine reorderings). None/0 on the first
        iteration (no previous eigenvalues)."""
        cur = eigs.detach().to(torch.float64).cpu()
        prev = self._prev_eig
        self._prev_eig = cur
        if prev is None or prev.shape != cur.shape:
            return None, 0
        drift = float((cur - prev).abs().max())
        nb = cur.shape[-1]
        ident = torch.arange(nb)
        count = 0
        flat_cur = cur.reshape(-1, nb)
        flat_prev = prev.reshape(-1, nb)
        for c, p in zip(flat_cur, flat_prev, strict=True):
            nn = (c[:, None] - p[None, :]).abs().argmin(dim=1)
            # a genuine crossing: the matched previous band nn[i] is well
            # separated from p[i], so matching there rather than to index i is
            # not degeneracy shuffling.
            sep = (p[nn] - p[ident]).abs()
            count += int(((nn != ident) & (sep > self._DEGEN_TOL)).sum())
        return drift, count

    def record(
        self,
        *,
        it: int,
        free_energy: float,
        dE: float | None,
        res_norm: float,
        t_iter: float,
        drho_r: torch.Tensor,
        eigs: list[torch.Tensor],
        fermi: float | None,
        entropy: float,
        mag_abs: float | None = None,
        e_metric: float | None = None,
        e_metric_charge: float | None = None,
        e_metric_mag: float | None = None,
        e_hf_gap: float | None = None,
        subspace_size: int | None = None,
        op_counts: dict[str, int] | None = None,
        t_eig_s: float | None = None,
    ) -> None:
        """Append one outer-iteration record. ``drho_r`` is the total real-space
        SCF residual (ρ_out − ρ_in), ``eigs`` the per-spin eigenvalue tensors,
        ``t_iter`` the loop's own measured wall time for this iteration.

        ``e_metric`` (and its charge/magnetization decomposition) is the
        residual's second-order energy-error estimate 1/2 <r|K_Hxc|r> [eV]
        (``postscf._response.kernel_energy_error``), recorded only when the
        energy-metric convergence gate is active; None otherwise (the cheap
        default path does not compute it). ``e_hf_gap`` is the Harris-Foulkes
        minus Kohn-Sham free energy at this iteration [eV], the zero-machinery
        bracket of the same error (NC driver, plain semilocal path only)."""
        stacked = torch.stack([e.detach() for e in eigs])  # (nspin, nk, nb)
        drift, reorder = self._eig_drift_and_reorder(stacked)
        self.iters.append(
            {
                "it": int(it),
                "free_energy": float(free_energy),
                "dE": None if dE is None else (float(dE) if _is_finite(dE) else None),
                "drho": float(res_norm),
                "t": float(t_iter),
                "fermi": None if fermi is None else float(fermi),
                "entropy": float(entropy),
                "eig_drift": drift,
                "reorder": int(reorder),
                "shell_frac": self._shell_fractions(drho_r),
                "mag_abs": None if mag_abs is None else float(mag_abs),
                "e_metric": None if e_metric is None else float(e_metric),
                "e_metric_charge": (
                    None if e_metric_charge is None else float(e_metric_charge)),
                "e_metric_mag": (
                    None if e_metric_mag is None else float(e_metric_mag)),
                "e_hf_gap": None if e_hf_gap is None else float(e_hf_gap),
                "subspace_size": None if subspace_size is None else int(subspace_size),
                "rss_mb": _rss_mb(),
                "t_eig": None if t_eig_s is None else float(t_eig_s),
                "n_fft": None if op_counts is None else int(op_counts.get("fft", 0)),
                "n_eigh": None if op_counts is None else int(op_counts.get("eigh", 0)),
                "n_hpsi": None if op_counts is None else int(op_counts.get("hpsi", 0)),
            }
        )

    def diagnose(self) -> list[tuple[str, str]]:
        """0-3 heuristic tags, each with a one-line reason. Never raises on
        short or degenerate runs: 1-2 iterations yield no tags.

        Thresholds are HEURISTICS, tuned on the test systems:
        - charge-sloshing: over the last ~5 iterations, >50% of the residual
          weight sits in the 2 lowest |G|-shells AND the residual is not
          monotonically falling AND the residual is still above
          ``_SLOSH_RES_FLOOR`` (a converged run's noise-floor tail has a
          meaningless shell decomposition and roundoff-level non-monotonicity,
          so it must not read as sloshing);
        - moment-collapse: nspin=2 with a nonzero seeded moment whose |M| falls
          across recent iterations to below both 10% of the seed AND
          ``_MOMENT_COLLAPSE_FLOOR`` μB (a collapse is a fall to the nonmagnetic
          branch, m→~0, not merely a large fraction below an inflated SAD seed);
        - level-crossing: a nonzero band-reordering count in a majority of
          recent (it > 1) iterations.
        """
        tags: list[tuple[str, str]] = []
        n = len(self.iters)
        if n < 3:
            return tags
        recent = self.iters[-5:]

        low2 = [sum(i["shell_frac"][:2]) for i in recent]
        res = [i["drho"] for i in recent]
        falling = all(res[j] <= res[j - 1] for j in range(1, len(res)))
        frac = _mean(low2)
        if frac > 0.5 and not falling and max(res) > _SLOSH_RES_FLOOR:
            tags.append(
                (
                    "charge-sloshing",
                    f"{frac * 100:.0f}% of recent residual weight in the 2 "
                    f"longest-wavelength shells with a non-monotone residual",
                )
            )

        if self.nspin == 2 and self.seed_moment > 1e-6:
            mags = [i["mag_abs"] for i in recent if i["mag_abs"] is not None]
            if (
                len(mags) >= 3
                and mags[-1] < mags[0]
                and mags[-1] < 0.1 * self.seed_moment
                and mags[-1] < _MOMENT_COLLAPSE_FLOOR
            ):
                tags.append(
                    (
                        "moment-collapse",
                        f"|M| fell from {mags[0]:.3f} to {mags[-1]:.3f} muB "
                        f"toward zero from a seeded {self.seed_moment:.3f}",
                    )
                )

        reorders = [i["reorder"] for i in recent if i["it"] > 1]
        nz = sum(1 for r in reorders if r > 0)
        if reorders and nz > len(reorders) / 2:
            tags.append(
                (
                    "level-crossing",
                    f"band reordering in {nz} of the last {len(reorders)} iterations",
                )
            )
        return tags[:3]

    def summarize(self) -> dict[str, Any]:
        """The compact diagnostics block for the run summary."""
        if not self.iters:
            return {"n_iter": 0, "diagnosis": []}
        final = self.iters[-1]
        block: dict[str, Any] = {
            "n_iter": len(self.iters),
            "final_residual": float(final["drho"]),
            "diagnosis": [{"tag": t, "reason": r} for t, r in self.diagnose()],
            "sloshing_fraction_final": float(sum(final["shell_frac"][:2])),
            "reordering_total": int(sum(i["reorder"] for i in self.iters)),
            "wall_time_mean_s": round(_mean([i["t"] for i in self.iters]), 4),
        }
        if self.nspin == 2:
            block["final_magnetization_muB"] = (
                None if final["mag_abs"] is None else float(final["mag_abs"])
            )
        if final.get("e_metric") is not None:
            block["energy_metric_eV"] = float(final["e_metric"])
            block["energy_metric_charge_eV"] = (
                None if final.get("e_metric_charge") is None
                else float(final["e_metric_charge"]))
            block["energy_metric_mag_eV"] = (
                None if final.get("e_metric_mag") is None
                else float(final["e_metric_mag"]))
            if final.get("e_hf_gap") is not None:
                block["harris_foulkes_gap_eV"] = float(final["e_hf_gap"])
        return block

    def to_trace_dict(self) -> dict[str, Any]:
        """Schema-versioned, JSON-serializable per-iteration arrays (the sidecar
        payload). One entry per iteration; the shell decomposition rides along as
        the ``shell_fraction`` list per iteration, with ``shell_edges`` in the
        header."""
        return {
            "version": TRACE_VERSION,
            "nspin": self.nspin,
            "n_shells": self.n_shells,
            "shell_edges_invAng": self.shell_edges,
            "seed_moment_muB": self.seed_moment,
            "mixing": self.mixing,
            "diagnosis": [{"tag": t, "reason": r} for t, r in self.diagnose()],
            "iterations": [
                {
                    "iter": i["it"],
                    "free_energy_eV": i["free_energy"],
                    "dE_eV": i["dE"],
                    "drho": i["drho"],
                    "t_s": round(i["t"], 4),
                    "fermi_eV": i["fermi"],
                    "entropy_eV": i["entropy"],
                    "eig_max_drift_eV": i["eig_drift"],
                    "reorder_count": i["reorder"],
                    "subspace_size": i.get("subspace_size"),
                    "rss_mb": i.get("rss_mb"),
                    "t_eig_s": i.get("t_eig"),
                    "n_fft": i.get("n_fft"),
                    "n_eigh": i.get("n_eigh"),
                    "n_hpsi": i.get("n_hpsi"),
                    "shell_fraction": i["shell_frac"],
                    **({"mag_abs_muB": i["mag_abs"]} if self.nspin == 2 else {}),
                    **(
                        {
                            "energy_metric_eV": i["e_metric"],
                            "energy_metric_charge_eV": i["e_metric_charge"],
                            "energy_metric_mag_eV": i["e_metric_mag"],
                            **({"harris_foulkes_gap_eV": i["e_hf_gap"]}
                               if i.get("e_hf_gap") is not None else {}),
                        }
                        if i.get("e_metric") is not None
                        else {}
                    ),
                }
                for i in self.iters
            ],
        }


def _is_finite(x: float) -> bool:
    import math

    return math.isfinite(x)


# Resident-set size sampled per outer iteration for the memory-over-time trace.
# Dependency-free (matches io/runinfo's /proc style — no psutil): statm field 2
# is RSS in pages. Returns None off Linux or if /proc is unreadable, so the
# recorder never fails on a platform without it.
_PAGE_MB = 4096 / 1e6  # overwritten at import if the real page size differs


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/statm") as fh:
            rss_pages = int(fh.read().split()[1])
        return round(rss_pages * _PAGE_MB, 3)
    except (OSError, IndexError, ValueError):
        return None


try:  # resolve the true page size once; fall back to the 4 KiB assumption
    import resource as _resource

    _PAGE_MB = _resource.getpagesize() / 1e6
except Exception:  # noqa: BLE001 — any failure keeps the 4 KiB default
    pass
