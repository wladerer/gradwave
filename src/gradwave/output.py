"""Human-readable output writer (Layer C).

One plain-text file per task, written next to the machine-readable JSON.
The JSON is the parsing target; this file is for eyes. Sections open
with a light rule carrying the section name; parameters sit in two
columns; every field degrades gracefully when absent so older JSONs
still render.
"""

from __future__ import annotations

from pathlib import Path

_W = 72


def _sec(title):
    pad = _W - len(title) - 4
    return f"\n── {title} {'─' * max(pad, 4)}"


def _cols(pairs, width=34):
    """Two-column 'label  value' layout from a list of (label, value)."""
    cells = [f"{lab:<12s}{val}" for lab, val in pairs]
    lines = []
    for i in range(0, len(cells), 2):
        left = cells[i]
        right = cells[i + 1] if i + 1 < len(cells) else ""
        lines.append(f"   {left:<{width}s}{right}".rstrip())
    return lines


def _structure_lines(struct):
    lines = [_sec("structure")]
    cell = struct["cell_ang"]
    for i, row in enumerate(cell):
        label = "cell [Å]" if i == 0 else ""
        lines.append(f"   {label:<12s}" + "".join(f"{x:12.6f}" for x in row))
    facts = []
    if struct.get("volume_ang3"):
        facts.append(f"volume {struct['volume_ang3']:.3f} Å³")
    if struct.get("spacegroup"):
        facts.append(f"space group {struct['spacegroup']}")
    if facts:
        lines.append("   " + " · ".join(facts))
    lines.append("")
    for sym, pos in zip(struct["species"], struct["positions_ang"],
                        strict=True):
        lines.append(f"   {sym:<4s}" + "".join(f"{x:12.6f}" for x in pos))
    return lines


def _parameters_lines(par):
    lines = [_sec("parameters")]
    mesh = "×".join(str(n) for n in par["kmesh"])
    if par.get("nk"):
        mesh += f" → {par['nk']} k"
    pairs = [
        ("formalism", par["formalism"]),
        ("xc", par["xc"].upper()),
        ("ecut", f"{par['ecut_eV']:.2f} eV"),
        ("ecutrho", f"{par['ecutrho_eV']:.2f} eV"
         if par.get("ecutrho_eV") else "—"),
        ("k-mesh", mesh),
        ("spin", str(par["nspin"])),
        ("smearing", par["smearing"] if par["smearing"] == "none"
         else f"{par['smearing']} ({par['width_eV']} eV)"),
        ("symmetry", "on" if par["symmetry"] else "off"),
    ]
    if par.get("n_electrons"):
        pairs.append(("electrons", f"{par['n_electrons']:g}"))
    if par.get("nbands"):
        pairs.append(("bands", str(par["nbands"])))
    if par.get("fft_grid"):
        pairs.append(("FFT grid",
                      "×".join(str(n) for n in par["fft_grid"])))
    if par.get("npw"):
        pairs.append(("plane waves", f"{par['npw']} (k 1)"))
    lines += _cols(pairs)
    for sym, fname in par["pseudos"].items():
        lines.append(f"   {'pseudo ' + sym:<12s}{fname}")
    return lines


def _scf_lines(scf, runtime=None):
    lines = [_sec("self-consistency")]
    trace = scf.get("trace", [])
    if trace:
        lines.append(f"   {'it':>4s}   {'F [eV]':>18s}   {'ΔE [eV]':>10s}"
                     f"   {'|Δρ|':>10s}")
        for h in trace:
            de = h.get("dE_eV")
            de_s = "—" if de is None else f"{abs(de):10.2e}"
            lines.append(f"   {h['iter']:>4d}   {h['free_energy_eV']:>18.10f}"
                         f"   {de_s:>10s}   {h['drho']:>10.2e}")
    tag = "converged" if scf["converged"] else "NOT CONVERGED"
    tail = f"   {tag} in {scf['n_iter']} iterations"
    if runtime is not None:
        tail += f" · {runtime:.1f} s"
    lines += ["", tail]
    return lines


def _energy_lines(scf):
    e = scf["energies_eV"]
    lines = [_sec("energy [eV]")]
    shown = [("kinetic", "kinetic"), ("hartree", "hartree"), ("xc", "xc"),
             ("local", "local pp"), ("nonlocal", "nonlocal pp"),
             ("ewald", "ewald"), ("onecenter", "one-center (PAW)"),
             ("hubbard", "hubbard U"), ("smearing", "smearing −σS")]
    for key, label in shown:
        val = e.get(key, 0.0)
        core = key in ("kinetic", "hartree", "xc", "local", "nonlocal",
                       "ewald")
        if core or abs(val) > 0:
            lines.append(f"   {label:<18s}{val:>20.10f}")
    lines.append(f"   {'':<18s}{'─' * 20:>20s}")
    lines.append(f"   {'total E':<18s}{e['total']:>20.10f}")
    lines.append(f"   {'free energy F':<18s}{e['free_energy']:>20.10f}")
    facts = []
    if scf.get("fermi_eV") is not None:
        facts.append(f"Fermi {scf['fermi_eV']:.4f} eV")
    if scf.get("gap_eV") is not None:
        facts.append(f"gap {scf['gap_eV']:.4f} eV")
    if scf.get("total_magnetization_muB") is not None:
        facts.append(f"m {scf['total_magnetization_muB']:.4f} μB "
                     f"(|m| {scf['absolute_magnetization_muB']:.4f})")
    if facts:
        lines += ["", "   " + " · ".join(facts)]
    return lines


def _error_lines(err):
    lines = [_sec("basis-set error estimate")]
    if not err.get("available", True):
        lines.append(f"   unavailable — {err.get('reason', 'out of coverage')}")
        return lines
    pairs = [
        ("Ecut", f"{err['ecut_eV']:.1f} eV"),
        ("Ecut large", f"{err['ecut_large_eV']:.1f} eV"),
        ("δE", f"{err['denergy_eV']:.4e} eV"),
        ("δE/atom", f"{err['denergy_meV_per_atom']:.3f} meV"),
        ("F → limit", f"{err['free_energy_extrapolated_eV']:.10f} eV"),
        ("∫|δρ|/e⁻", f"{err['drho_L1_per_electron']:.3e}"),
    ]
    if "force_error_max_eV_ang" in err:
        pairs.append(("δF max", f"{err['force_error_max_eV_ang']:.3e} eV/Å"))
        pairs.append(("δF rms", f"{err['force_error_rms_eV_ang']:.3e} eV/Å"))
    if "gap_eV" in err:
        pairs.append(("gap", f"{err['gap_eV']:.4f} eV"))
        pairs.append(("gap → limit", f"{err['gap_extrapolated_eV']:.4f} eV"))
        pairs.append(("δgap", f"{err['dgap_eV']:.4e} eV"))
    scf = err.get("scf_convergence")
    if scf is not None:
        pairs.append(("δE scf", f"{scf['denergy_eV']:.3e} eV"))
        pairs.append(("∫|δρscf|/e⁻", f"{scf['residual_L1_per_electron']:.2e}"))
    sm = err.get("smearing")
    if sm is not None:
        pairs.append(("δE smear", f"{sm['dsmearing_eV']:.3e} eV"))
        pairs.append(("E → σ=0", f"{sm['energy_extrapolated_eV']:.8f} eV"))
    lines += _cols(pairs)
    lines.append(f"   {err['note']}")
    if scf is not None:
        mode = "dielectric-screened" if scf["screened"] else "unscreened upper bound"
        lines.append(f"   SCF error: {mode}; converged energy ≈ "
                     f"{scf['energy_converged_estimate_eV']:.8f} eV")
    if sm is not None and sm.get("note"):
        lines.append(f"   smearing: {sm['note']}")
    return lines


def _eigenvalue_lines(summary, max_k=8):
    eig = summary.get("eigenvalues_eV")
    occ = summary.get("occupations")
    if eig is None:
        return []
    lines = [_sec("eigenvalues [eV] (occupation)")]
    par = summary["parameters"]
    nspin = par["nspin"]
    spins = [eig] if nspin == 1 else eig
    occs = [occ] if nspin == 1 else occ
    for isp, (es, fs) in enumerate(zip(spins, occs, strict=True)):
        if nspin == 2:
            lines.append(f"   spin {'up' if isp == 0 else 'down'}:")
        for ik, (ek, fk) in enumerate(zip(es[:max_k], fs, strict=False)):
            w = par["kweights"][ik] if par.get("kweights") else None
            head = f"   k {ik + 1}"
            if w is not None:
                head += f" · weight {w:.6f}"
            lines.append(head)
            for j in range(0, len(ek), 4):
                lines.append("     " + "  ".join(
                    f"{ev:9.4f} ({f:5.3f})"
                    for ev, f in zip(ek[j:j + 4], fk[j:j + 4], strict=True)))
        if len(es) > max_k:
            lines.append(f"   … {len(es) - max_k} more k-points in the JSON")
    return lines


def _relax_lines(relax):
    lines = [_sec("relaxation")]
    lines.append(f"   optimizer {relax.get('optimizer', '?')} · "
                 f"target fmax {relax.get('fmax_target_eV_ang', '?')} eV/Å")
    lines.append("")
    lines.append(f"   {'step':>4s}   {'E [eV]':>18s}   {'fmax [eV/Å]':>12s}")
    for step in relax.get("trajectory", []):
        lines.append(f"   {step['step']:>4d}   "
                     f"{step['energy_eV']:>18.10f}   "
                     f"{step['fmax_eV_ang']:>12.6f}")
    tag = "converged" if relax["converged"] else "NOT CONVERGED"
    lines += ["", f"   {tag} after {relax['n_steps']} steps · "
              f"fmax {relax['fmax_eV_ang']:.6f} eV/Å"]
    if relax.get("max_displacement_ang") is not None:
        lines.append(f"   max displacement from start: "
                     f"{relax['max_displacement_ang']:.4f} Å")
    lines.append("")
    lines.append("   final positions [Å]:")
    for sym, pos in zip(relax["species"], relax["positions_ang"],
                        strict=True):
        lines.append(f"   {sym:<4s}" + "".join(f"{x:12.6f}" for x in pos))
    return lines


def _magnetism_lines(mag):
    lines = [_sec("magnetism")]
    lines.append(f"   ordering: {mag['ordering']}")
    lines.append(f"   total moment: {mag['total_moment_muB']:.3f} μB")
    moms = ", ".join(f"{m:.3f}" for m in mag["atomic_moments_muB"])
    lines.append(f"   atomic moments [μB]: {moms}")
    if mag.get("exchange_J_meV"):
        js = ", ".join(f"J_{i} = {J:+.1f}" for i, J in mag["exchange_J_meV"].items())
        lines.append(f"   Heisenberg exchange [meV]: {js}")
        dmi = mag.get("dmi_meV") or {}
        if any(abs(d) > 1e-3 for d in dmi.values()):
            ds = ", ".join(f"D_{i} = {d:+.3f}" for i, d in dmi.items())
            lines.append(f"   DMI [meV]: {ds}")
        if mag.get("curie_temperature_mfa_K") is not None:
            lines.append(f"   mean-field T_c (nn): {mag['curie_temperature_mfa_K']} K")
    lines.append("")
    return lines


def _bands_lines(bands):
    lines = [_sec("band structure")]
    labels = " – ".join(lab for _x, lab in bands["labels"])
    pairs = [("path", labels),
             ("k-points", str(len(bands["x"]))),
             ("bands", str(len(bands["eigenvalues_eV"][0])))]
    if bands.get("reference_eV") is not None:
        pairs.append(("E = 0 at", f"{bands['reference_eV']:.6f} eV"))
    lines += _cols(pairs)
    lines.append("   dispersion data in the JSON · "
                 "plot with `gradwave plot bands.json`")
    return lines


def _pdos_lines(pdos):
    """Projected-DOS summary: spilling and the integrated Löwdin weight per group
    (electrons per group, summed over the spectrum). Spin-resolved for nspin=2."""
    import numpy as np

    lines = ["", "   projected DOS"]
    if not pdos.get("available", True):
        lines.append(f"   unavailable · {pdos.get('reason', '')}")
        return lines
    e = np.asarray(pdos["energy_eV"], dtype=float)
    nspin = int(pdos.get("nspin", 1))
    lines.append(f"   group_by {pdos['group_by']} · "
                 f"spilling {pdos['spilling']:.4f} · "
                 f"integrated Löwdin weight [electrons]")

    def integ(arr):
        a = np.asarray(arr, dtype=float)
        return np.trapezoid(a, e, axis=-1)

    if nspin == 2:
        lines.append(f"   {'group':<20s}{'up':>10s}{'down':>10s}{'net':>10s}")
        for lab, arr in sorted(pdos["groups"].items()):
            up, dn = integ(arr)
            lines.append(f"   {lab:<20s}{up:>10.4f}{dn:>10.4f}{up - dn:>+10.4f}")
    else:
        lines.append(f"   {'group':<28s}{'weight':>12s}")
        for lab, arr in sorted(pdos["groups"].items()):
            lines.append(f"   {lab:<28s}{float(integ(arr)):>12.4f}")
    return lines


def format_output(summary: dict) -> str:
    """The full human-readable report for a task summary dict."""
    code = summary["code"]
    created = code["created"].replace("T", " ")
    head = f"gradwave {code['version']} · {summary['task']} run · {created}"
    lines = [head, "─" * min(len(head), _W)]
    lines += _structure_lines(summary["structure"])
    lines += _parameters_lines(summary["parameters"])
    if "scf" in summary:
        lines += _scf_lines(summary["scf"], summary.get("runtime_s"))
        lines += _energy_lines(summary["scf"])
        if "error_estimate" in summary:
            lines += _error_lines(summary["error_estimate"])
        lines += _eigenvalue_lines(summary)
    if "pdos" in summary:
        lines += _pdos_lines(summary["pdos"])
    if "relax" in summary:
        lines += _relax_lines(summary["relax"])
    if "bands" in summary:
        lines += _bands_lines(summary["bands"])
    if "magnetism" in summary:
        lines += _magnetism_lines(summary["magnetism"])
    tail = []
    if "runtime_s" in summary and "scf" not in summary:
        tail.append(f"wall time {summary['runtime_s']:.1f} s")
    if "outputs" in summary:
        tail.append("files " + " · ".join(summary["outputs"].values()))
    if tail:
        lines += ["", "   " + "  |  ".join(tail)]
    lines.append("")
    return "\n".join(lines) + "\n"


def write_output(summary: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_output(summary))
    return path
