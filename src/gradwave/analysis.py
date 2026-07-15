"""pandas/matplotlib utilities over gradwave result files (Layer C).

Everything here consumes the machine-readable JSON that run() writes
(or the summary dict directly) — no live SCF objects, so results can be
analyzed on any machine. Imports of pandas/matplotlib are lazy so the
core package works without them installed.

    from gradwave import analysis
    r = analysis.load("out/scf.json")
    analysis.scf_frame(r)            # per-iteration convergence table
    analysis.eigenvalues_frame(r)    # tidy (spin, k, band, energy, occ)
    analysis.dos_frame(r, width=0.1) # gaussian-broadened DOS
    analysis.plot_scf(r, path="scf.png")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _pd():
    try:
        import pandas
    except ImportError as err:  # pragma: no cover
        raise ImportError("gradwave.analysis needs pandas "
                          "(uv pip install pandas)") from err
    return pandas


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover
        raise ImportError("gradwave.analysis plotting needs matplotlib "
                          "(uv pip install matplotlib)") from err
    return plt


def load(source) -> dict:
    """A summary dict from a path to <task>.json (dicts pass through)."""
    if isinstance(source, dict):
        return source
    return json.loads(Path(source).read_text())


def scf_frame(source):
    """Per-iteration SCF convergence: iter, free_energy_eV, dE_eV, drho,
    plus dF_from_final (|F_it − F_last|, the practical convergence read)."""
    s = load(source)
    pd = _pd()
    trace = s["scf"]["trace"]
    df = pd.DataFrame(trace)
    if len(df):
        df["dF_from_final_eV"] = (df["free_energy_eV"]
                                  - df["free_energy_eV"].iloc[-1]).abs()
    return df


def eigenvalues_frame(source):
    """Tidy eigenvalues: one row per (spin, k, band) with energy [eV],
    occupation and k-weight."""
    s = load(source)
    pd = _pd()
    nspin = s["parameters"]["nspin"]
    kw = s["parameters"]["kweights"]
    eig = np.asarray(s["eigenvalues_eV"], dtype=float)
    occ = np.asarray(s["occupations"], dtype=float)
    if nspin == 1:
        eig, occ = eig[None], occ[None]
    rows = []
    for isp in range(nspin):
        for ik in range(eig.shape[1]):
            for ib in range(eig.shape[2]):
                rows.append((isp, ik, kw[ik], ib, eig[isp, ik, ib],
                             occ[isp, ik, ib]))
    return pd.DataFrame(rows, columns=["spin", "k", "kweight", "band",
                                       "energy_eV", "occupation"])


def bands_frame(source):
    """Tidy band path: one row per (k, band) with the path coordinate x
    and energy [eV] relative to the stored reference. The high-symmetry
    labels ride along as df.attrs['labels']."""
    s = load(source)
    pd = _pd()
    b = s["bands"] if "bands" in s else s
    eig = np.asarray(b["eigenvalues_eV"], dtype=float)
    x = np.asarray(b["x"], dtype=float)
    ref = b.get("reference_eV") or 0.0
    rows = []
    for ik in range(eig.shape[0]):
        for ib in range(eig.shape[1]):
            rows.append((ik, float(x[ik]), ib, eig[ik, ib] - ref))
    df = pd.DataFrame(rows, columns=["k", "x", "band", "energy_eV"])
    df.attrs["labels"] = b["labels"]
    return df


def dos_frame(source, width: float = 0.1, npoints: int = 800, window=None):
    """Gaussian-broadened density of states from the SCF eigenvalues and
    k-weights. width in eV; window (emin, emax) defaults to the spectrum
    padded by 10 widths. Spin channels come back as separate columns."""
    s = load(source)
    pd = _pd()
    nspin = s["parameters"]["nspin"]
    kw = np.asarray(s["parameters"]["kweights"], dtype=float)
    eig = np.asarray(s["eigenvalues_eV"], dtype=float)
    if nspin == 1:
        eig = eig[None]
    if window is None:
        window = (eig.min() - 10 * width, eig.max() + 10 * width)
    grid = np.linspace(window[0], window[1], npoints)
    g_spin = 2.0 if nspin == 1 else 1.0
    cols = {"energy_eV": grid}
    for isp in range(nspin):
        w = np.broadcast_to(kw[:, None], eig[isp].shape).reshape(-1)
        e = eig[isp].reshape(-1)
        d = (np.exp(-0.5 * ((grid[:, None] - e[None, :]) / width) ** 2)
             / (width * np.sqrt(2 * np.pi)) * w[None, :] * g_spin).sum(axis=1)
        cols["dos" if nspin == 1 else ("dos_up", "dos_down")[isp]] = d
    df = pd.DataFrame(cols)
    df.attrs["fermi_eV"] = s["scf"].get("fermi_eV")
    return df


def _finish(fig, ax, path):
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=180)
    return ax


def plot_scf(source, path=None, ax=None):
    """Convergence plot: |F − F_final| and |Δρ| per iteration, log scale."""
    plt = _plt()
    df = scf_frame(source)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.semilogy(df["iter"], df["dF_from_final_eV"].clip(lower=1e-16),
                "o-", ms=3.5, color="#2a78d6", label="|F − F_final| [eV]")
    ax.semilogy(df["iter"], df["drho"], "s-", ms=3.5, color="#1baf7a",
                label="|Δρ|")
    ax.set_xlabel("SCF iteration")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return _finish(ax.figure, ax, path)


def plot_bands(source, path=None, ax=None, window=None):
    """Band structure along the stored path, labels on the x axis, the
    reference energy (VBM or Fermi) at zero."""
    plt = _plt()
    df = bands_frame(source)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for _band, sub in df.groupby("band"):
        ax.plot(sub["x"], sub["energy_eV"], color="#2a78d6", lw=1.1)
    for xt, _lab in df.attrs["labels"]:
        ax.axvline(xt, color="#52514e", lw=0.5, alpha=0.5)
    ax.axhline(0.0, color="#52514e", lw=0.5, ls="--", alpha=0.7)
    ax.set_xticks([xt for xt, _ in df.attrs["labels"]])
    ax.set_xticklabels([lab.replace("G", "Γ") for _, lab in
                        df.attrs["labels"]])
    ax.set_ylabel("E − E_ref [eV]")
    ax.set_xlim(df["x"].min(), df["x"].max())
    if window is not None:
        ax.set_ylim(*window)
    return _finish(ax.figure, ax, path)


def plot_dos(source, path=None, ax=None, width: float = 0.1):
    """Broadened DOS; spin-down plotted negative for nspin=2."""
    plt = _plt()
    df = dos_frame(source, width=width)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(5.4, 3.6))
    if "dos" in df:
        ax.plot(df["energy_eV"], df["dos"], color="#2a78d6")
        ax.fill_between(df["energy_eV"], df["dos"], alpha=0.15,
                        color="#2a78d6")
    else:
        ax.plot(df["energy_eV"], df["dos_up"], color="#2a78d6",
                label="up")
        ax.plot(df["energy_eV"], -df["dos_down"], color="#1baf7a",
                label="down")
        ax.legend(frameon=False)
    if df.attrs.get("fermi_eV") is not None:
        ax.axvline(df.attrs["fermi_eV"], color="#52514e", lw=0.7, ls="--")
    ax.set_xlabel("E [eV]")
    ax.set_ylabel("DOS [states/eV]")
    return _finish(ax.figure, ax, path)
