"""PW-only validation: FM bcc-Fe, FM fcc-Ni, and a PtCu geometry optimization.

Runs three plane-wave DFT calculations with gradwave and renders a single
self-contained HTML report (inline-SVG matplotlib, no external assets):

  1. bcc-Fe  ferromagnetic collinear SCF (nspin=2) -> total/absolute moment
  2. fcc-Ni  ferromagnetic collinear SCF (nspin=2) -> the weak-moment case
  3. PtCu L1_0 positions-only BFGS relax -> displaced-Cu relaxes back to site

Spin-resolved DOS (kpm_dos) is drawn for both magnets: the exchange splitting
between the up and down channels is the visual centrepiece.

FLAPW is deliberately excluded: gradwave's all-electron FLAPW path has no
spin-polarization, so magnetic moments are not available there.

Run (on asus, 8 threads):
    OMP_NUM_THREADS=8 uv run python experiments/run_pw_magnetic_relax.py \
        --outdir experiments
"""

from __future__ import annotations

import argparse
import html
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradwave.api.relax import run_relax
from gradwave.api.scf import run_scf
from gradwave.api.summary import build_summary
from gradwave.inputs import load_input
from gradwave.io.runinfo import machine_snapshot
from gradwave.postscf.dos import kpm_dos

HERE = Path(__file__).resolve().parent

# reference magnetic moments (muB). Experiment is the headline anchor; the
# delta-gauge all-electron WIEN2k value is the code-to-code anchor.
REF = {
    "Fe": {"expt": 2.22, "wien2k": 2.20, "note": "bcc, spin-only"},
    "Ni": {"expt": 0.61, "wien2k": 0.64, "note": "fcc, weak moment"},
}


def _svg(fig) -> str:
    """Render a matplotlib figure to an inline <svg> string (no XML/doctype)."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    s = buf.getvalue()
    i = s.find("<svg")
    return s[i:] if i >= 0 else s


def run_magnet(name: str, yaml: str) -> dict[str, Any]:
    """Run one FM collinear SCF, return summary numbers + spin-resolved DOS."""
    print(f"\n=== {name}: {yaml} ===", flush=True)
    inp = load_input(str(HERE / yaml))
    t0 = time.perf_counter()
    res = run_scf(inp, verbose=True)
    wall = time.perf_counter() - t0
    summ = build_summary(res, inp, "scf", runtime_s=wall)
    scf = summ["scf"]
    fermi = scf["fermi_eV"]
    m_tot = scf.get("total_magnetization_muB")
    m_abs = scf.get("absolute_magnetization_muB")
    print(f"{name}: converged={scf['converged']} n_iter={scf['n_iter']} "
          f"m_tot={m_tot:.4f} m_abs={m_abs:.4f} muB Ef={fermi:.4f} eV "
          f"E={scf['energies_eV']['total']:.6f} eV  ({wall:.1f}s)", flush=True)

    # spin-resolved DOS: nspin=2 -> (2, nE) on a shared grid, degeneracy 1 each
    e_dos, dos, dinfo = kpm_dos(res, n_moments=1600, n_random=8, n_energies=700)
    dos = np.asarray(dos)
    assert dos.ndim == 2 and dos.shape[0] == 2, f"expected (2,nE) DOS, got {dos.shape}"

    return {
        "name": name,
        "yaml": yaml,
        "converged": bool(scf["converged"]),
        "n_iter": int(scf["n_iter"]),
        "m_tot": float(m_tot),
        "m_abs": float(m_abs),
        "fermi_eV": float(fermi),
        "E_total_eV": float(scf["energies_eV"]["total"]),
        "wall_s": float(wall),
        "final_dE_eV": scf["convergence"]["final_dE_eV"],
        "final_drho": scf["convergence"]["final_drho"],
        "settings": {
            "ecut_eV": inp.ecut, "kmesh": list(inp.kpoints.mesh),
            "smearing": inp.smearing.type, "width_eV": inp.smearing.width,
            "nbands": inp.nbands, "mixing": inp.scf.mixing.scheme,
            "etol_eV": inp.scf.etol, "rhotol": inp.scf.rhotol,
            "symmetry": inp.symmetry,
            "pseudo": inp.pseudo_map,
        },
        "dos": {"e": np.asarray(e_dos).tolist(),
                "up": dos[0].tolist(), "dn": dos[1].tolist()},
    }


def run_ptcu(yaml: str) -> dict[str, Any]:
    print(f"\n=== PtCu relax: {yaml} ===", flush=True)
    inp = load_input(str(HERE / yaml))
    t0 = time.perf_counter()
    relax, atoms, frames = run_relax(inp, verbose=True)
    wall = time.perf_counter() - t0
    traj = relax["trajectory"]
    # the two Cu atoms (indices 2,3) start at z = 1.88 Å; ideal L1_0 site z = c/2.
    c = inp.atoms.cell.array[2, 2]
    ideal_z = c / 2.0
    z0 = [float(inp.atoms.get_positions()[i, 2]) for i in (2, 3)]
    zf = [float(atoms.get_positions()[i, 2]) for i in (2, 3)]
    print(f"PtCu: converged={relax['converged']} n_steps={relax['n_steps']} "
          f"fmax {traj[0]['fmax_eV_ang']:.4f} -> {traj[-1]['fmax_eV_ang']:.4f} "
          f"Cu z {z0} -> {zf} (ideal {ideal_z:.3f})  ({wall:.1f}s)", flush=True)
    return {
        "converged": bool(relax["converged"]),
        "n_steps": int(relax["n_steps"]),
        "optimizer": relax.get("optimizer", inp.relax.optimizer),
        "fmax_target": float(inp.relax.fmax),
        "ideal_z": float(ideal_z),
        "cu_z_initial": z0,
        "cu_z_final": zf,
        "wall_s": float(wall),
        "steps": [t["step"] for t in traj],
        "energy_eV": [float(t["energy_eV"]) for t in traj],
        "fmax": [float(t["fmax_eV_ang"]) for t in traj],
        "scf_iter": [t.get("scf_iter") for t in traj],
        "settings": {
            "ecut_eV": inp.ecut, "ecutrho_eV": inp.ecutrho,
            "kmesh": list(inp.kpoints.mesh), "smearing": inp.smearing.type,
            "width_eV": inp.smearing.width, "symmetry": inp.symmetry,
            "pseudo": inp.pseudo_map, "cell_ang": inp.atoms.cell.array.tolist(),
        },
    }


# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------
def dos_svg(m: dict[str, Any]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = np.asarray(m["dos"]["e"]) - m["fermi_eV"]  # energy relative to E_F
    up = np.asarray(m["dos"]["up"])
    dn = np.asarray(m["dos"]["dn"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.fill_between(e, up, 0, color="#2a78d6", alpha=0.85, lw=0, label="spin ↑")
    ax.fill_between(e, -dn, 0, color="#c1442e", alpha=0.85, lw=0, label="spin ↓")
    ax.axvline(0.0, color="#52514e", lw=1.0, ls="--", alpha=0.9)
    ax.axhline(0.0, color="#333", lw=0.8)
    ax.set_xlim(-9, 5)
    ax.set_xlabel("E − E$_F$  (eV)")
    ax.set_ylabel("DOS  (states/eV, ↑ up / ↓ down)")
    ax.set_title(f"{m['name']} spin-resolved DOS  "
                 f"(m$_{{tot}}$ = {m['m_tot']:.2f} μB)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.25)
    fig.tight_layout()
    return _svg(fig)


def relax_svg(p: dict[str, Any]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = p["steps"]
    fig, (axE, axF) = plt.subplots(1, 2, figsize=(8.8, 3.4))
    e0 = p["energy_eV"][0]
    axE.plot(steps, [(x - e0) * 1e3 for x in p["energy_eV"]], "o-",
             color="#2a78d6", lw=1.6, ms=4)
    axE.set_xlabel("BFGS step")
    axE.set_ylabel("E − E$_0$  (meV)")
    axE.set_title("Energy vs step")
    axE.grid(True, lw=0.3, alpha=0.25)
    axF.semilogy(steps, p["fmax"], "o-", color="#c1442e", lw=1.6, ms=4)
    axF.axhline(p["fmax_target"], color="#52514e", lw=1.0, ls="--", alpha=0.9)
    axF.annotate(f"fmax target {p['fmax_target']:g}", (steps[-1], p["fmax_target"]),
                 textcoords="offset points", xytext=(-6, 6), ha="right",
                 fontsize=8, color="#2a2a28")
    axF.set_xlabel("BFGS step")
    axF.set_ylabel("max force  (eV/Å)")
    axF.set_title("Force convergence")
    axF.grid(True, which="both", lw=0.3, alpha=0.25)
    fig.tight_layout()
    return _svg(fig)


# ----------------------------------------------------------------------------
# HTML assembly
# ----------------------------------------------------------------------------
def _esc(x: Any) -> str:
    return html.escape(str(x))


def build_html(snap: dict[str, Any], fe: dict, ni: dict, ptcu: dict) -> str:
    code = snap["code"]
    cpu = snap.get("cpu", {})
    git = code.get("git_full") or code.get("git") or "unknown"
    dirty = " (dirty)" if code.get("git_dirty") else ""
    threads = torch.get_num_threads()
    host = snap["host"]
    host_str = f"{host['hostname']} · {host['os']} · {host['arch']}"
    cpu_model = cpu.get("model", cpu.get("brand", "n/a"))
    cpu_cores = cpu.get("cores_physical", cpu.get("count", "?"))
    cpu_str = f"{cpu_model} · {cpu_cores} cores"

    def moment_row(m, ref):
        r = REF[ref]
        err_e = m["m_tot"] - r["expt"]
        pct = 100.0 * err_e / r["expt"]
        conv = "✓" if m["converged"] else "✗"
        return (f"<tr><td><b>{_esc(m['name'])}</b></td>"
                f"<td>{m['m_tot']:.3f}</td><td>{m['m_abs']:.3f}</td>"
                f"<td>{r['expt']:.2f}</td><td>{r['wien2k']:.2f}</td>"
                f"<td>{err_e:+.3f}</td><td>{pct:+.1f}%</td>"
                f"<td>{m['E_total_eV']:.5f}</td><td>{m['fermi_eV']:.4f}</td>"
                f"<td>{m['n_iter']}</td><td>{conv}</td></tr>")

    def settings_line(s):
        return (f"ecut {s['ecut_eV']:.1f} eV · k {s['kmesh']} · "
                f"{s['smearing']} {s['width_eV']} eV · nbands {s['nbands']} · "
                f"mixing {s['mixing']} · etol {s['etol_eV']:g} · "
                f"sym {s['symmetry']} · {next(iter(s['pseudo'].values()))}")

    fe_set = settings_line(fe["settings"])
    ni_set = settings_line(ni["settings"])
    ps = ptcu["settings"]
    ptcu_set = (f"ecut {ps['ecut_eV']:.1f} eV / ecutrho {ps['ecutrho_eV']:.1f} eV "
                f"· k {ps['kmesh']} · {ps['smearing']} {ps['width_eV']} eV · "
                f"sym {ps['symmetry']} · PAW {list(ps['pseudo'].values())}")

    z0 = ptcu["cu_z_initial"]
    zf = ptcu["cu_z_final"]
    ideal = ptcu["ideal_z"]
    ptcu_geo = "".join(
        f"<tr><td>Cu {i}</td><td>{z0[i]:.4f}</td><td>{zf[i]:.4f}</td>"
        f"<td>{ideal:.4f}</td><td>{zf[i]-ideal:+.4f}</td>"
        f"<td>{(zf[i]-ideal)/(z0[i]-ideal)*100:.1f}%</td></tr>"
        for i in range(len(z0)))

    return f"""<h1>Plane-wave validation: magnetic moments &amp; geometry relaxation</h1>
<p class="lead">Ferromagnetic bcc-Fe and fcc-Ni (collinear spin, nspin=2) and a
PtCu L1<sub>0</sub> positions-only geometry optimization, all on the plane-wave
path. FLAPW is out of scope (no spin-polarization).</p>

<h2>1. Provenance &amp; settings</h2>
<table class="prov">
<tr><td>timestamp</td><td>{_esc(snap['timestamp'])}</td></tr>
<tr><td>host</td><td>{_esc(host_str)}</td></tr>
<tr><td>commit</td><td><code>{_esc(git)}</code>{dirty}</td></tr>
<tr><td>gradwave</td><td>{_esc(code.get('gradwave'))}</td></tr>
<tr><td>python / torch</td><td>{_esc(code.get('python'))} / {_esc(code.get('torch'))}</td></tr>
<tr><td>CPU</td><td>{_esc(cpu_str)}</td></tr>
<tr><td>torch threads</td><td>{threads}</td></tr>
</table>
<ul class="settings">
<li><b>Fe</b>: {_esc(fe_set)}</li>
<li><b>Ni</b>: {_esc(ni_set)}</li>
<li><b>PtCu</b>: {_esc(ptcu_set)}</li>
</ul>

<h2>2. Magnetic moments</h2>
<table class="data">
<thead><tr><th>system</th><th>m<sub>tot</sub> (μB)</th><th>m<sub>abs</sub> (μB)</th>
<th>expt</th><th>WIEN2k</th><th>Δ vs expt</th><th>%</th>
<th>E<sub>tot</sub> (eV)</th><th>E<sub>F</sub> (eV)</th><th>iters</th><th>conv</th></tr></thead>
<tbody>
{moment_row(fe, 'Fe')}
{moment_row(ni, 'Ni')}
</tbody></table>
<p class="cap">Reference moments: experiment is the headline anchor; the WIEN2k
column is the all-electron delta-gauge code-to-code anchor. Δ and % are against
experiment. Absolute magnetization m<sub>abs</sub> = ∫|m(r)|dr &ge; the net
moment m<sub>tot</sub> = N↑ − N↓.</p>

<h2>3. Spin-resolved DOS — the exchange splitting</h2>
<p>Spin-up (blue, plotted up) and spin-down (red, plotted down) densities of
states from a Kernel-Polynomial-Method expansion of the converged potential,
aligned to E<sub>F</sub> = 0. Fe shows a large exchange splitting (the ↑ band is
filled well below E<sub>F</sub> while ↓ straddles it); Ni's splitting is small,
consistent with its weak ~0.6 μB moment.</p>
<div class="fig">{dos_svg(fe)}</div>
<div class="fig">{dos_svg(ni)}</div>

<h2>4. PtCu geometry optimization</h2>
<p>The two Cu atoms start ~0.08 Å above their ideal L1<sub>0</sub> site
(z = c/2 = {ideal:.3f} Å); positions-only BFGS relaxes them back toward it.</p>
<div class="fig">{relax_svg(ptcu)}</div>
<table class="data">
<thead><tr><th>atom</th><th>z<sub>initial</sub> (Å)</th><th>z<sub>final</sub> (Å)</th>
<th>ideal (Å)</th><th>residual (Å)</th><th>residual / initial offset</th></tr></thead>
<tbody>{ptcu_geo}</tbody></table>
<p class="cap">converged = <b>{ptcu['converged']}</b> · n_steps = {ptcu['n_steps']}
· optimizer {ptcu['optimizer']} · fmax {ptcu['fmax'][0]:.4f} →
{ptcu['fmax'][-1]:.4f} eV/Å (target {ptcu['fmax_target']:g}).</p>

<h2>5. Honest caveats</h2>
<ul>
<li><b>Convergence level.</b> These are validation-grade, not production-converged.
Fe/Ni use ecut = 60 Ry and an 8×8×8 k-mesh on the 1-atom primitive; PtCu uses
ecut = 50 Ry (PAW ecutrho 400 Ry) and 8×8×8. A production study would converge
ecut and k-mesh to sub-meV total-energy / sub-0.01 μB moment tolerance and check
finite-size effects; the moments here are reported as-run.</li>
<li><b>Smearing broadens the metal moment.</b> A Gaussian width of 0.1 eV
partially fills the minority band at E<sub>F</sub>, which reduces the net moment
relative to the zero-broadening limit — the effect is strongest for the weak-moment
Ni case, where the moment is a small difference of large occupations and is most
sensitive to the smearing width and k-mesh density.</li>
<li><b>Symmetry off for the magnets.</b> A collinear moment breaks the paramagnetic
space group, so IBZ reduction / density symmetrization is disabled for Fe and Ni
(enabled for the non-magnetic PtCu relax, where the equal Cu displacement keeps the
tetragonal symmetry along the whole path).</li>
<li><b>FLAPW excluded.</b> gradwave's all-electron FLAPW path has no
spin-polarization, so magnetic moments are not available there. All magnetic
results above are from the plane-wave pseudopotential path exclusively; FLAPW was
not and cannot be used for this comparison.</li>
</ul>
<p class="foot">Report generated by
<code>experiments/run_pw_magnetic_relax.py</code>.</p>
"""


HEAD = """<style>
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
max-width:900px;margin:2rem auto;padding:0 1.2rem;color:#1a1a18;line-height:1.5}
h1{font-size:1.6rem;border-bottom:2px solid #2a78d6;padding-bottom:.3rem}
h2{font-size:1.2rem;margin-top:2rem;color:#243b53}
.lead{font-size:1.05rem;color:#444}
table{border-collapse:collapse;margin:.8rem 0;font-size:.9rem;width:100%}
table.data th,table.data td{border:1px solid #d0d0cc;padding:.35rem .6rem;text-align:right}
table.data th:first-child,table.data td:first-child{text-align:left}
table.data thead{background:#eef4fb}
table.prov td{padding:.2rem .6rem;border:none}
table.prov td:first-child{color:#666;width:11rem}
.settings{font-size:.85rem;color:#333}
.fig{overflow-x:auto;margin:1rem 0;border:1px solid #eee;border-radius:6px;
padding:.4rem;background:#fff}
.fig svg{max-width:100%;height:auto}
.cap{font-size:.83rem;color:#555}
.foot{font-size:.8rem;color:#999;margin-top:2rem;border-top:1px solid #eee;padding-top:.5rem}
code{background:#f4f4f2;padding:.1rem .3rem;border-radius:3px;font-size:.85em}
</style>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    snap = machine_snapshot()
    fe = run_magnet("Fe", "fe_fm.yaml")
    ni = run_magnet("Ni", "ni_fm.yaml")
    ptcu = run_ptcu("ptcu_relax.yaml")

    # dump the captured numbers (DOS arrays included) for provenance / reuse
    dump = {"machine": snap, "Fe": fe, "Ni": ni, "PtCu": ptcu}
    (outdir / "pw_magnetic_relax_data.json").write_text(json.dumps(dump, indent=1))

    body = build_html(snap, fe, ni, ptcu)
    report = outdir / "pw_magnetic_relax_report.html"
    report.write_text("<!doctype html><html><head><meta charset='utf-8'>"
                      "<title>PW magnetic &amp; relax validation</title>"
                      + HEAD + "</head><body>" + body + "</body></html>")
    print(f"\nwrote {report}")
    print(f"wrote {outdir / 'pw_magnetic_relax_data.json'}")


if __name__ == "__main__":
    main()
