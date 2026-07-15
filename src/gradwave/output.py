"""Human-readable output writer (Layer C).

One plain-text file per task, written next to the machine-readable JSON.
The JSON is the parsing target; this file is for eyes. Sections appear
in run order: header, structure, parameters, SCF trace, energies, then
task-specific blocks (relax trajectory, band summary).
"""

from __future__ import annotations

from pathlib import Path

_W = 78


def _rule(ch="-"):
    return ch * _W


def _sec(title):
    return f"\n{_rule('=')}\n  {title}\n{_rule('=')}"


def _kv(label, value, unit=""):
    unit = f" {unit}" if unit else ""
    return f"  {label:<28s}{value}{unit}"


def _structure_lines(struct):
    lines = [_sec("structure")]
    lines.append("  cell [Å]:")
    for row in struct["cell_ang"]:
        lines.append("    " + "".join(f"{x:14.8f}" for x in row))
    lines.append("  positions [Å]:")
    for sym, pos in zip(struct["species"], struct["positions_ang"],
                        strict=True):
        lines.append(f"    {sym:<4s}" + "".join(f"{x:14.8f}" for x in pos))
    return lines


def _parameters_lines(par):
    lines = [_sec("parameters")]
    lines.append(_kv("formalism", par["formalism"]))
    lines.append(_kv("xc", par["xc"]))
    lines.append(_kv("ecut", f"{par['ecut_eV']:.2f}", "eV"))
    if par.get("ecutrho_eV"):
        lines.append(_kv("ecutrho", f"{par['ecutrho_eV']:.2f}", "eV"))
    mesh = "x".join(str(n) for n in par["kmesh"])
    if par.get("nk"):
        mesh += f"  ({par['nk']} points after reduction)"
    lines.append(_kv("k-mesh", mesh))
    lines.append(_kv("spin channels", par["nspin"]))
    if par["smearing"] != "none":
        lines.append(_kv("smearing",
                         f"{par['smearing']} (width {par['width_eV']} eV)"))
    lines.append(_kv("symmetry", "on" if par["symmetry"] else "off"))
    for sym, fname in par["pseudos"].items():
        lines.append(_kv(f"pseudo {sym}", fname))
    return lines


def _scf_lines(scf):
    lines = [_sec("self-consistency")]
    trace = scf.get("trace", [])
    if trace:
        lines.append(f"  {'iter':>4s} {'free energy [eV]':>20s} "
                     f"{'dE [eV]':>12s} {'|drho|':>12s}")
        for h in trace:
            de = h.get("dE_eV")
            lines.append(
                f"  {h['iter']:>4d} {h['free_energy_eV']:>20.10f} "
                f"{'' if de is None else format(de, '>12.3e'):>12s} "
                f"{h['drho']:>12.3e}")
    tag = "converged" if scf["converged"] else "NOT CONVERGED"
    lines.append(f"\n  {tag} in {scf['n_iter']} iterations")
    e = scf["energies_eV"]
    lines.append(_sec("energies [eV]"))
    for key in ("kinetic", "hartree", "xc", "local", "nonlocal", "ewald",
                "hubbard", "onecenter", "smearing"):
        if key in e and abs(e[key]) > 0 or key in (
                "kinetic", "hartree", "xc", "local", "nonlocal", "ewald"):
            lines.append(_kv(key, f"{e[key]:20.10f}"))
    lines.append("  " + "-" * 50)
    lines.append(_kv("total energy E", f"{e['total']:20.10f}"))
    lines.append(_kv("free energy F = E - sigma*S", f"{e['free_energy']:20.10f}"))
    if scf.get("fermi_eV") is not None:
        lines.append(_kv("Fermi energy", f"{scf['fermi_eV']:.6f}", "eV"))
    if scf.get("gap_eV") is not None:
        lines.append(_kv("band gap", f"{scf['gap_eV']:.4f}", "eV"))
    if scf.get("total_magnetization_muB") is not None:
        lines.append(_kv("total magnetization",
                         f"{scf['total_magnetization_muB']:.4f}", "muB"))
        lines.append(_kv("absolute magnetization",
                         f"{scf['absolute_magnetization_muB']:.4f}", "muB"))
    return lines


def _eigenvalue_lines(summary, max_k=None):
    eig = summary.get("eigenvalues_eV")
    occ = summary.get("occupations")
    if eig is None:
        return []
    lines = [_sec("eigenvalues [eV] (occupations)")]
    nspin = summary["parameters"]["nspin"]
    spins = [eig] if nspin == 1 else eig
    occs = [occ] if nspin == 1 else occ
    for isp, (es, fs) in enumerate(zip(spins, occs, strict=True)):
        if nspin == 2:
            lines.append(f"\n  spin {'up' if isp == 0 else 'down'}:")
        shown = es if max_k is None else es[:max_k]
        for ik, (ek, fk) in enumerate(zip(shown, fs, strict=False)):
            lines.append(f"  k {ik + 1}  (weight "
                         f"{summary['parameters']['kweights'][ik]:.6f})")
            for j in range(0, len(ek), 4):
                lines.append("    " + "".join(
                    f"{ev:12.5f} ({f:6.4f})"
                    for ev, f in zip(ek[j:j + 4], fk[j:j + 4], strict=True)))
        if max_k is not None and len(es) > max_k:
            lines.append(f"  ... {len(es) - max_k} more k-points in the JSON")
    return lines


def _relax_lines(relax):
    lines = [_sec("relaxation")]
    lines.append(f"  {'step':>4s} {'energy [eV]':>20s} {'fmax [eV/Å]':>14s}")
    for step in relax.get("trajectory", []):
        lines.append(f"  {step['step']:>4d} {step['energy_eV']:>20.10f} "
                     f"{step['fmax_eV_ang']:>14.6f}")
    tag = "converged" if relax["converged"] else "NOT CONVERGED"
    lines.append(f"\n  {tag} after {relax['n_steps']} steps "
                 f"(fmax {relax['fmax_eV_ang']:.6f} eV/Å)")
    lines.append("\n  final positions [Å]:")
    for sym, pos in zip(relax["species"], relax["positions_ang"],
                        strict=True):
        lines.append(f"    {sym:<4s}" + "".join(f"{x:14.8f}" for x in pos))
    return lines


def _bands_lines(bands):
    lines = [_sec("band structure")]
    labels = " - ".join(lab for _x, lab in bands["labels"])
    lines.append(_kv("path", labels))
    lines.append(_kv("k-points", len(bands["x"])))
    nb = len(bands["eigenvalues_eV"][0])
    lines.append(_kv("bands", nb))
    if bands.get("reference_eV") is not None:
        lines.append(_kv("reference (E=0)", f"{bands['reference_eV']:.6f}",
                         "eV"))
    lines.append("  full dispersion data in the JSON; plot with "
                 "`gradwave plot bands.json`")
    return lines


def format_output(summary: dict) -> str:
    """The full human-readable report for a task summary dict."""
    code = summary["code"]
    lines = [
        _rule("="),
        f"  gradwave {code['version']} — {summary['task']} run",
        f"  {code['created']}",
        _rule("="),
    ]
    lines += _structure_lines(summary["structure"])
    lines += _parameters_lines(summary["parameters"])
    if "scf" in summary:
        lines += _scf_lines(summary["scf"])
        lines += _eigenvalue_lines(summary, max_k=8)
    if "relax" in summary:
        lines += _relax_lines(summary["relax"])
    if "bands" in summary:
        lines += _bands_lines(summary["bands"])
    if "runtime_s" in summary:
        lines.append("")
        lines.append(_kv("wall time", f"{summary['runtime_s']:.1f}", "s"))
    if "outputs" in summary:
        for name, rel in summary["outputs"].items():
            lines.append(_kv(name, rel))
    lines.append(_rule("="))
    return "\n".join(lines) + "\n"


def write_output(summary: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_output(summary))
    return path
