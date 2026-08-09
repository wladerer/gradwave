# Electrocatalysis run: *H and *CO on Pt(111) & Au(111)

CHE-based adsorption energetics for four adsorbate–surface pairs, structured so
**one pair runs end-to-end first** (the debug/validation target), then all four,
then two stretch goals (r2SCAN/ISDF, differentiable descriptors). Built to run on
a rented H100 (see the connection notes at the bottom).

The walkthrough with results and figures is the manual chapter,
`docs/manual/electrocatalysis.md`. Figures were rendered from the relaxed
geometries with tinykit, e.g. `tk viz results/Pt_CO_hcp_relaxed.xyz --supercell 2
2 1 --rotation -75 -8 0 --orthographic -o fig.png`.

## What's here

```
pseudos/        PAW psl.1.0.0 PBE — Pt Au C O H (the production set)
pseudos_nc/     ONCV PBE — for the r2SCAN/ISDF stretch (meta-GGA is NC-only)
structures.py   build slabs + adsorbate sites + gas refs → structures/*.xyz
structures/     16 adsorbate configs + 2 slabs + 3 gas refs (+ manifest.json)
config.py       cutoffs, k-mesh, smearing, calculator factory (+ DEBUG profile)
che.py          CHE thermodynamics: ΔE, ΔG, HER descriptor / limiting potential
run_pair.py     ONE pair end-to-end (clean slab → sites → gas → ΔG); resumable
run_all.py      all four pairs + summary table
r2scan_isdf.py  STRETCH: r2SCAN(+ISDF) single-points vs PBE on the best geometry
differentiable.py STRETCH: autograd descriptor gradient (strain-tunes-adsorption)
results/        per-pair JSON + BFGS logs (written incrementally)
```

## Method

- **PAW psl.1.0.0 (PBE)**, `ecutwfc = 50 Ry`, `ecutrho = 400 Ry` (from the pseudo
  headers; max wfc 47 Ry / rho 401 Ry). Downloaded Au + H PAW from the QE repo;
  Pt/C/O were in the gradwave fixtures — one consistent family.
- **Slab**: fcc(111), 2×2×4 (1/4 ML), ~15 Å vacuum, **bottom 2 layers fixed**.
  `a_PBE`: Pt 3.968, Au 4.159 Å. k-mesh `(4,4,1)` (→ `(6,6,1)` for production).
- **Metals**: cold (Marzari–Vanderbilt) smearing, 0.15 eV. Non-magnetic (nspin=1).
- **Gas refs**: H2, H2O, CO, Γ-only, `smearing="none"`.
- **CHE**: `*H`: `* + ½H2 → H*`; `*CO`: `* + CO → CO*`. ΔG adds standard 298 K
  ZPE+TS aggregates (`che.DG_CORR`) — swap for frequency-derived values for
  quantitative work (a `vib.py` Hessian hook is the natural add-on).

## Acceleration / convergence settings (audited vs docs/manual/*)

Tuned per `wisdom.md` / `performance.md` / `convergence.md`:

- **`precond="local_tf"`** (not the default kerker) — the key change. Constant
  Kerker over-damps the vacuum region of a slab/molecule box; local Thomas–Fermi
  tracks local density (capped at Kerker in the bulk). Al-slab iters 21→17, 27→21,
  energy bit-identical; the gain grows with vacuum fraction.
- **`mixing_scheme="johnson"`** — pinned (the PAW default already resolves to it;
  johnson 13 vs pulay 17 iters on fcc Pt PAW).
- **Metals: `rhotol=1e-5`, `etol=1e-6`** — a smeared metal floors the density
  residual at occupation noise while the free energy is settled; gate on the
  energy tail, don't fight the 1e-7 floor. (The doc's `scf.convergence: energy`
  gate isn't exposed on the ASE calculator.) Gas insulators keep tight tol.
- **Kept (correct for a PAW-metal relaxation):** `use_symmetry=True` (IBZ 5–14×),
  `davidson` (chebyshev is NC-only → would error on PAW, and slower on H100),
  `reuse_wavefunctions` + `extrapolation="reuse"` (part of the 1.37× PAW-metal
  relaxation speedup), `nbands=None` (20% metal headroom — do NOT trim; buffer
  bands hold the smearing tail), `cold`/0.15 eV, BFGS optimizer (FIRE is 8.5×
  slower), ions-only fixed-cell relax.
- **OFF on purpose:** `mixed_precision` (regresses every metal; no win on H100
  fp64), `compile_xc` (few %, 1-min first trace — not worth it for a single relax),
  CUDA graphs / fp32 Rayleigh–Ritz (measured no-ops or floors).

**Two runtime checks (the docs' explicit caveats):**
1. **Fractional occupations must appear** at 0.15 eV on the k-mesh — else the
   "metal" is silently a fixed-occupation insulator. If not, raise the k-mesh.
2. **k-mesh convergence.** `(4,4,1)` is the fast first-pass (audit-approved, not
   flagged). For production accuracy on a 2×2 metal surface, re-check the best
   site at **`(6,6,1)`** (edit `config.KPTS_SLAB`) — adsorption energies on metals
   can be k-noisy; the difference should be < ~30 meV or bump further.

## Run order (do this on the box)

```bash
# 0. structures are committed; rebuild only if you change geometry:
uv run python structures.py

# 1. FULL RUN-THROUGH on one pair first — catch everything here:
uv run python run_pair.py Pt H

# 2. all four pairs + summary:
uv run python run_all.py

# 3. stretch (if time):
uv run python r2scan_isdf.py Pt CO      # does r2SCAN change CO/Pt binding?
uv run python differentiable.py Pt H    # dE_ads/dε via autograd
```

Local sanity of the whole flow (tiny, CPU, minutes): add `--debug` to `run_pair`
/ `run_all`.

## Expectations / sanity checks

- `*H` on Pt(111): ΔG_H* near 0 (Pt is the HER benchmark, |ΔG_H*| ≲ 0.1 eV);
  fcc/hcp hollow or top competitive. Au binds H weakly (ΔG_H* > 0, poor HER).
- `*CO` on Pt(111): strong binding, PBE **over-binds** (top vs hollow site
  ordering is the famous "CO puzzle" — PBE wrongly favors hollow). This is exactly
  why the **r2SCAN stretch** is interesting: it should reduce over-binding and can
  flip the site preference toward top. Au binds CO weakly.
- If Pt *H ΔG_H* is wildly off (≫0.3 eV) or CO doesn't bind, suspect the k-mesh
  (too coarse), smearing, or a pseudo mismatch — debug before trusting the rest.

## Cost / GPU notes (from the H100 handoff)

- fp64 is real on H100 — stay in fp64 (`mixed_precision` barely helps).
- Memory ≈ npw × nbands; trim `nbands` to ~1.2× occupied for big cells. These
  cells are small (16–18 atoms), so memory is not the constraint — throughput is.
- One heavy GPU job at a time. `nvidia-smi` before launching.
- Detach long runs so an SSH drop can't kill them:
  ```bash
  setsid bash -c 'stdbuf -oL uv run python run_all.py > /root/run_all.log 2>&1; \
    echo EXIT=$? >> /root/run_all.log' </dev/null >/dev/null 2>&1 &
  ```
- **`pytest` on the box: always `-n0`** (192 cores → xdist hangs).
- **CPU threads on a big uniform box:** gradwave caps intra-op threads at
  `min(cores,8)` (tuned for hybrid P/E laptops). On a uniform many-core server
  (e.g. 2× Xeon 8480+, 112 cores) the parallel setup phase (G-sphere build,
  atomic-superposition density) can use more, so launch with
  `GRADWAVE_NUM_THREADS=32` — within one NUMA socket (56 cores) to avoid
  cross-socket traffic. Do NOT bake this into `config.py`: >8 threads *regress*
  on hybrid CPUs (asus/laptop). It's a minor lever — the SCF is GPU-bound and the
  dominant one-time cost is torch/CUDA import (unthreadable); the calc-reuse in
  `run_pair` is the real per-stage win.

## Connect (vast.ai, mirrors ~/Downloads/h100_bifeo3_handoff.md)

```bash
SSH="ssh -p <PORT> -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes"
$SSH root@<IP> 'mkdir -p /root/gw-electrocat'
rsync -a -e "$SSH" --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
  <this-worktree>/ root@<IP>:/root/gw-electrocat/
$SSH root@<IP> 'cd /root/gw-electrocat && uv sync'   # CUDA torch
$SSH root@<IP> 'cd /root/gw-electrocat/experiments/electrocat && \
  setsid bash -c "stdbuf -oL uv run python run_pair.py Pt H > /root/pth.log 2>&1; \
    echo EXIT=\$? >> /root/pth.log" </dev/null >/dev/null 2>&1 &'
# poll: $SSH root@<IP> 'tail -8 /root/pth.log'
```
Strip vast banners with `grep -vE "Welcome|Have fun|AI agents|READ /etc|vast-cap"`.
Terminate the instance when outputs are rsync'd back (metered).
