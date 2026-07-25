"""Write and run a matched QE ``pw.x`` SCF for a Mineral, parse energy / wall /
k-count. The cell handed to QE is byte-for-byte the same lattice + positions
gradwave uses (ibrav=0, CELL_PARAMETERS angstrom). Antiferromagnetic order is
expressed the QE way: each magnetic element is split into up/down SUB-SPECIES
that share the same UPF, with opposite starting_magnetization -- this is what
lets QE detect the magnetic space group and reduce k comparably to gradwave's
Shubnikov fold."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from structures import MASS, Mineral

RY_EV = 13.605693122994


def _qe_species(m: Mineral):
    """Return (labels_per_atom, species_cards, starting_mag_lines).

    Magnetic atoms of the same element but opposite sign become distinct QE
    types (Fe1/Fe2, ...). Non-magnetic species keep a single type."""
    cards, mag_lines, label_of_atom = [], [], []
    type_index = {}  # (symbol, signbucket) -> (label, qe_type_number)

    def bucket(sign):
        return 0 if sign == 0 else (1 if sign > 0 else 2)

    for i in range(m.nat):
        sym = m.symbols[m.species[i]]
        pseudo = m.pseudos[m.species[i]]
        sgn = m.start_mag[i]
        key = (sym, bucket(sgn))
        if key not in type_index:
            n = len(type_index) + 1
            suffix = "" if bucket(sgn) == 0 else str(bucket(sgn))
            label = f"{sym}{suffix}"
            type_index[key] = (label, n)
            cards.append(f"  {label}  {MASS[sym]:.3f}  {pseudo}")
            mag_lines.append(f"  starting_magnetization({n}) = {sgn:.3f}")
        label_of_atom.append(type_index[key][0])
    return label_of_atom, cards, mag_lines, len(type_index)


def write_input(m: Mineral, nbands: int, pseudo_dir: str, workdir: Path,
                mixing_beta: float = 0.3) -> Path:
    labels, cards, mag_lines, ntyp = _qe_species(m)
    ecutwfc = m.ecut_ry
    ecutrho = m.rho_factor * m.ecut_ry
    lines = [
        "&control", "  calculation = 'scf'", f"  prefix = '{m.name}'",
        "  outdir = './tmp'", f"  pseudo_dir = '{pseudo_dir}'",
        "  verbosity = 'high'", "  tprnfor = .false.", "  tstress = .false.", "/",
        "&system", "  ibrav = 0", f"  nat = {m.nat}", f"  ntyp = {ntyp}",
        f"  ecutwfc = {ecutwfc:.1f}", f"  ecutrho = {ecutrho:.1f}",
        f"  nbnd = {nbands}",
        "  occupations = 'smearing'", "  smearing = 'gaussian'",
        f"  degauss = {m.degauss_ry:.10f}",
        "  input_dft = 'PBE'",
    ]
    if m.nspin == 2:
        lines.append("  nspin = 2")
        lines += mag_lines
    lines += [
        "/", "&electrons", "  conv_thr = 1.0d-9",
        "  mixing_mode = 'plain'", f"  mixing_beta = {mixing_beta}",
        "  electron_maxstep = 200", "/",
        "ATOMIC_SPECIES", *cards,
        "CELL_PARAMETERS angstrom",
    ]
    for row in m.cell:
        lines.append("  " + "  ".join(f"{x:.12f}" for x in row))
    lines.append("ATOMIC_POSITIONS angstrom")
    for lab, r in zip(labels, m.positions, strict=True):
        lines.append(f"  {lab}  " + "  ".join(f"{x:.10f}" for x in r))
    k = m.kmesh
    lines += ["K_POINTS automatic", f"  {k[0]} {k[1]} {k[2]} 0 0 0", ""]
    workdir.mkdir(parents=True, exist_ok=True)
    inp = workdir / "pw.in"
    inp.write_text("\n".join(lines))
    return inp


def _wall_seconds(text: str) -> float | None:
    # final line: "     PWSCF        :   1m23.45s CPU   1m30.12s WALL"
    m = re.search(r"PWSCF\s*:.*?CPU\s+(.+?)\s+WALL", text)
    if not m:
        return None
    tok = m.group(1).strip()
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)\s*([hms])", tok):
        total += float(num) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total or None


def run(m: Mineral, nbands: int, pseudo_dir: str, workdir: Path,
        nranks: int = 8) -> dict:
    write_input(m, nbands, pseudo_dir, workdir)
    cmd = ["mpirun", "-np", str(nranks), "pw.x", "-in", "pw.in"]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, env=env)
    out = proc.stdout
    (workdir / "pw.out").write_text(out)
    if proc.stderr:
        (workdir / "pw.err").write_text(proc.stderr)
    res: dict = {"ranks": nranks, "converged": "convergence has been achieved" in out,
                 "job_done": "JOB DONE" in out}
    et = re.findall(r"!\s+total energy\s*=\s*([-\d.]+)\s*Ry", out)
    res["etot_eV"] = float(et[-1]) * RY_EV if et else None
    # energy breakdown (Ry -> eV); QE "one-electron" == kin + local + nonlocal
    for key, pat in (
        ("e_one_electron_eV", r"one-electron contribution\s*=\s*([-\d.]+)\s*Ry"),
        ("e_hartree_eV", r"hartree contribution\s*=\s*([-\d.]+)\s*Ry"),
        ("e_xc_eV", r"xc contribution\s*=\s*([-\d.]+)\s*Ry"),
        ("e_ewald_eV", r"ewald contribution\s*=\s*([-\d.]+)\s*Ry"),
    ):
        mm = re.search(pat, out)
        res[key] = float(mm.group(1)) * RY_EV if mm else None
    res["wall_s"] = _wall_seconds(out)
    nk = re.search(r"number of k points\s*=\s*(\d+)", out)
    res["nk_irr"] = int(nk.group(1)) if nk else None
    fft = re.search(r"FFT dimensions:\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\)", out)
    res["fft_dims"] = [int(fft.group(i)) for i in (1, 2, 3)] if fft else None
    for key, pat in (("tot_mag", r"total magnetization\s*=\s*([-\d.]+)"),
                     ("abs_mag", r"absolute magnetization\s*=\s*([-\d.]+)")):
        hits = re.findall(pat, out)
        res[key] = float(hits[-1]) if hits else None
    niter = re.findall(r"iteration #\s*(\d+)", out)
    res["n_iter"] = int(niter[-1]) if niter else None
    ef = re.findall(r"the Fermi energy is\s*([-\d.]+)\s*ev", out)
    res["fermi_eV"] = float(ef[-1]) if ef else None
    if not res["job_done"]:
        res["tail"] = out[-1500:]
    return res
