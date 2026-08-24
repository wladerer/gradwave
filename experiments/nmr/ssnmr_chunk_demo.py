"""End-to-end demo of the shielding k-streaming fix on routine silicates.

Runs the norm-conserving PW SCF + analytic q→0 bare shielding (sigma_shielding_dq)
on a chosen material and reports peak RSS + σ_iso per site. The point: cells whose
full-mesh dense k-contexts do not fit in RAM (the 28-42 GB OOM that blocked routine
ssNMR) now run with CHUNK>=1, which streams those contexts one k at a time.

Env: MAT (Si|quartz|cristobalite|forsterite, default quartz), ECUT (eV, default 600),
     K (default 2), CHUNK (0=eager whole-mesh, N=chunk_k=N stream), PDIR.

Structures use the corrected α-quartz O=(0.4133,0.2672,0.2144) (Si-O 1.606/1.613 Å) and
the sg-62 forsterite from the earlier ssNMR structure diagnosis.
"""
import os
import resource
import time
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal

from gradwave.api.scf import run_scf
from gradwave.inputs import Input, KPointsParams, NmrParams
from gradwave.postscf.kgeometry_nmr import sigma_shielding_dq

PMAP = {"Si": "Si_ONCV_PBE-1.2.upf", "O": "O_ONCV_PBE-1.2.upf",
        "Mg": "Mg_ONCV_PBE-1.2.upf"}
EXP_29SI = {"Si": -81.1, "quartz": -107.4, "cristobalite": -108.5, "forsterite": -61.9}


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB


def structures():
    return {
        "Si": bulk("Si", "diamond", a=5.431),
        "quartz": crystal(
            ["Si", "O"], basis=[(0.4697, 0.0, 1 / 3), (0.4133, 0.2672, 0.2144)],
            spacegroup=152, cellpar=[4.9137, 4.9137, 5.4047, 90, 90, 120]),
        "cristobalite": crystal(
            ["Si", "O"], basis=[(0.3006, 0.3006, 0.0), (0.2392, 0.1049, 0.1789)],
            spacegroup=92, cellpar=[4.978, 4.978, 6.948, 90, 90, 90]),
        "forsterite": crystal(
            ["Mg", "Mg", "Si", "O", "O", "O"],
            basis=[(0, 0, 0), (0.27751, 0.25, 0.99143), (0.09400, 0.25, 0.42627),
                   (0.09133, 0.25, 0.76575), (0.44736, 0.25, 0.22150),
                   (0.16333, 0.03317, 0.27751)],
            spacegroup=62, cellpar=[10.1971, 5.9806, 4.7540, 90, 90, 90]),
    }


def main():
    name = os.environ.get("MAT", "quartz")
    ecut = float(os.environ.get("ECUT", "600"))
    k = int(os.environ.get("K", "2"))
    chunk = int(os.environ.get("CHUNK", "1"))
    pdir = Path(os.path.expanduser(os.environ.get(
        "PDIR", "~/github/gradwave/tests/fixtures/qe/pseudos")))
    atoms = structures()[name]
    pmap = {s: PMAP[s] for s in set(atoms.get_chemical_symbols())}
    inp = Input(atoms=atoms, pseudo_dir=pdir, pseudo_map=pmap, ecut=ecut, task="nmr",
                symmetry=False, kpoints=KPointsParams(mesh=[k, k, k]),
                nmr=NmrParams(task="shielding"))
    print(f"# {name}: {len(atoms)} atoms {atoms.get_chemical_formula()} "
          f"ecut={ecut} k{k}{k}{k} CHUNK={chunk} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    t0 = time.time()
    res = run_scf(inp, verbose=False)
    print(f"# SCF {time.time()-t0:.0f}s peak={_peak_gb():.1f}GB", flush=True)

    kw = {} if chunk == 0 else {"chunk_k": chunk}
    t1 = time.time()
    sig = sigma_shielding_dq(res, **kw).detach().cpu().numpy()  # (nsite,3,3) ppm
    print(f"# shielding {time.time()-t1:.0f}s peak={_peak_gb():.1f}GB "
          f"(CHUNK={chunk})", flush=True)

    syms = atoms.get_chemical_symbols()
    si_iso = []
    for i in range(sig.shape[0]):
        s = 0.5 * (sig[i] + sig[i].T)
        iso = float(np.trace(s) / 3.0)
        eig = np.linalg.eigvalsh(s)
        span = float(eig.max() - eig.min())
        print(f"  {syms[i]}{i}: sigma_iso={iso:+.2f} ppm  span={span:.1f}", flush=True)
        if syms[i] == "Si":
            si_iso.append(iso)
    if si_iso:
        print(f"  => mean sigma_iso(29Si)={sum(si_iso)/len(si_iso):+.2f} ppm  "
              f"(exp delta {EXP_29SI.get(name, '?')} ppm)", flush=True)


if __name__ == "__main__":
    main()
