# Electrocatalysis run: *H and *CO on Pt(111) & Au(111)

CHE-based adsorption energetics for four adsorbate–surface pairs, structured so
**one pair runs end-to-end first** (the debug/validation target), then all four,
then two stretch goals (r2SCAN/ISDF, differentiable descriptors). Built to run on
a rented H100 (see the connection notes at the bottom).

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
