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


def _is_noncollinear_block(b) -> bool:
    return isinstance(b, dict) and (b.get("noncollinear") or "m_z" in b)


def _pdos_block(source) -> dict:
    """Extract the projected-DOS dict from a ProjectedDOS/SOC ProjectedDOS, a raw
    block, or a JSON summary that carries a top-level ``pdos`` key. Collinear and
    j-resolved share the (total, groups) schema; the noncollinear spin-texture
    block is handled by _noncollinear_block/noncollinear_pdos_frame."""
    if hasattr(source, "to_dict") and hasattr(source, "groups"):
        return source.to_dict()
    if isinstance(source, dict) and "groups" in source and "energy_eV" in source:
        return source
    s = load(source)
    if "pdos" not in s:
        raise ValueError("no projected DOS in this result "
                         "(run with projections enabled)")
    block = s["pdos"]
    if _is_noncollinear_block(block):
        raise ValueError("this is a noncollinear (spin-texture) PDOS; use "
                         "noncollinear_pdos_frame / plot_spin_texture")
    return block


def _noncollinear_block(source) -> dict:
    """The noncollinear PDOS dict from a NoncollinearPDOS, a raw block, or a JSON
    summary carrying a top-level ``pdos`` that is noncollinear."""
    if hasattr(source, "to_dict") and hasattr(source, "m_z"):
        return source.to_dict()
    if _is_noncollinear_block(source):
        return source
    s = load(source)
    block = s.get("pdos")
    if not _is_noncollinear_block(block):
        raise ValueError("no noncollinear projected DOS in this result")
    return block


def pdos_frame(source):
    """Tidy projected-DOS frame. One column per group (and per spin for
    nspin=2, suffixed _up/_down). ``df.attrs`` carries fermi_eV and spilling."""
    pd = _pd()
    block = _pdos_block(source)
    nspin = block["nspin"]
    cols = {"energy_eV": np.asarray(block["energy_eV"], dtype=float)}

    def add(name, arr):
        a = np.asarray(arr, dtype=float)
        if nspin == 1:
            cols[name] = a
        else:
            cols[f"{name}_up"], cols[f"{name}_down"] = a[0], a[1]

    add("total", block["total"])
    for lab, arr in block["groups"].items():
        add(lab, arr)
    df = pd.DataFrame(cols)
    df.attrs.update(fermi_eV=block.get("fermi_eV"),
                    spilling=block.get("spilling"), nspin=nspin)
    return df


def noncollinear_pdos_frame(source):
    """Tidy noncollinear PDOS frame. Columns are the charge n(E) and the spin
    texture m_x/m_y/m_z(E) per group, prefixed ``charge_``/``mx_``/``my_``/``mz_``,
    plus the total charge. ``df.attrs`` carries fermi_eV and spilling."""
    pd = _pd()
    block = _noncollinear_block(source)
    cols = {"energy_eV": np.asarray(block["energy_eV"], dtype=float),
            "total_charge": np.asarray(block["total_charge"], dtype=float)}
    for key, pre in (("charge", "charge"), ("m_x", "mx"), ("m_y", "my"),
                     ("m_z", "mz")):
        for lab, arr in block[key].items():
            cols[f"{pre}_{lab}"] = np.asarray(arr, dtype=float)
    df = pd.DataFrame(cols)
    df.attrs.update(fermi_eV=block.get("fermi_eV"),
                    spilling=block.get("spilling"), noncollinear=True)
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


def plot_phonons(source, path=None):
    """Phonon dispersion (with a DOS side panel when present) from a
    phonons.json: branches along the q-path, high-symmetry labels on the x
    axis, ω = 0 marked (imaginary modes plot as negative)."""
    plt = _plt()
    s = load(source)
    ph = s["phonons"] if "phonons" in s else s
    x = np.asarray(ph["x"], dtype=float)
    freqs = np.asarray(ph["frequencies_cm1"], dtype=float)  # (nq, nbranch)
    labels = ph["labels"]
    has_dos = "dos" in ph
    if has_dos:
        fig, (ax, axd) = plt.subplots(
            1, 2, figsize=(6.8, 4.2), sharey=True,
            gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})
    else:
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for j in range(freqs.shape[1]):
        ax.plot(x, freqs[:, j], color="#2a78d6", lw=1.1)
    for xt, _lab in labels:
        ax.axvline(xt, color="#52514e", lw=0.5, alpha=0.5)
    ax.axhline(0.0, color="#52514e", lw=0.5, ls="--", alpha=0.7)
    ax.set_xticks([xt for xt, _ in labels])
    ax.set_xticklabels([lab.replace("G", "Γ") for _, lab in labels])
    ax.set_ylabel("ω [cm⁻¹]")
    ax.set_xlim(x.min(), x.max())
    if has_dos:
        g = np.asarray(ph["dos"]["frequency_cm1"], dtype=float)
        d = np.asarray(ph["dos"]["dos"], dtype=float)
        axd.plot(d, g, color="#2a78d6", lw=1.1)
        axd.axhline(0.0, color="#52514e", lw=0.5, ls="--", alpha=0.7)
        axd.set_xlabel("DOS")
        axd.set_xticks([])
    return _finish(fig, ax, path)


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


_PDOS_COLORS = ["#2a78d6", "#1baf7a", "#d6892a", "#a24bd6", "#d64b6b",
                "#4bb8d6", "#7a7a2a", "#d62a2a"]


def plot_pdos(source, path=None, ax=None, total=True):
    """Projected DOS, one curve per group, the total behind them. Spin-down is
    plotted negative for nspin=2."""
    plt = _plt()
    df = pdos_frame(source)
    nspin = df.attrs.get("nspin", 1)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(5.8, 3.8))
    e = df["energy_eV"]
    groups = [c[:-3] for c in df.columns if c.endswith("_up") and c != "total_up"] \
        if nspin == 2 else [c for c in df.columns
                            if c not in ("energy_eV", "total")]

    def draw(name, color):
        if nspin == 1:
            ax.plot(e, df[name], color=color, lw=1.2, label=name)
        else:
            ax.plot(e, df[f"{name}_up"], color=color, lw=1.2, label=name)
            ax.plot(e, -df[f"{name}_down"], color=color, lw=1.2)

    if total:
        if nspin == 1:
            ax.fill_between(e, df["total"], color="#c9c9c9", label="total")
        else:
            ax.fill_between(e, df["total_up"], color="#e0e0e0")
            ax.fill_between(e, -df["total_down"], color="#e0e0e0")
    for i, g in enumerate(groups):
        draw(g, _PDOS_COLORS[i % len(_PDOS_COLORS)])
    if df.attrs.get("fermi_eV") is not None:
        ax.axvline(df.attrs["fermi_eV"], color="#52514e", lw=0.7, ls="--")
    if nspin == 2:
        ax.axhline(0.0, color="#52514e", lw=0.5)
    ax.set_xlabel("E [eV]")
    ax.set_ylabel("PDOS [states/eV]")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _finish(ax.figure, ax, path)


def plot_spin_texture(source, path=None, ax=None, group="total", component="z"):
    """Noncollinear PDOS: the charge n(E) filled in grey with the chosen spin
    texture component m_x/m_y/m_z(E) overlaid (positive above, negative below).
    ``group`` picks the group column ('total' is the total charge)."""
    plt = _plt()
    df = noncollinear_pdos_frame(source)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(5.8, 3.8))
    e = df["energy_eV"]
    charge = df["total_charge"] if group == "total" else df[f"charge_{group}"]
    mcol = "total" if group == "total" else group
    m = df[f"m{component}_{mcol}"] if f"m{component}_{mcol}" in df else None
    ax.fill_between(e, charge, color="#c9c9c9", label=f"charge ({group})")
    if m is not None:
        ax.plot(e, m, color="#d64b6b", lw=1.3, label=f"m_{component} ({group})")
        ax.axhline(0.0, color="#52514e", lw=0.5)
    if df.attrs.get("fermi_eV") is not None:
        ax.axvline(df.attrs["fermi_eV"], color="#52514e", lw=0.7, ls="--")
    ax.set_xlabel("E [eV]")
    ax.set_ylabel("PDOS / spin texture [states/eV]")
    ax.legend(frameon=False, fontsize=8)
    return _finish(ax.figure, ax, path)
