<!-- DRAFT: results tables filled from results/nixos once the asus matrix completes -->
# Low-q damping of the transverse magnetization channels in the spinor SCF

Research note, branch `research/transverse-damping`. Executes recommendation 4
of the noncollinear-convergence campaign (`research/noncollinear-convergence`),
which localized the spinor residual floor to two independent parts. The
transverse magnetization channels (m perpendicular to the moment) are amplified
by the mixing map at about 3x per iteration from machine zero to a ~1e-4
saturation, with the power in the lowest |G| shells, the magnon-soft sector. The
longitudinal channel keeps a separate near-Stoner floor of a few 1e-4. The pin
prototype held the transverse channels at 1e-14 for 80 iterations, which proved
the transverse part is a mixed-state instability, not band-solve noise. This
study asks whether a reverse-Kerker damping of the transverse channels kills the
amplification without freezing the physical moment evolution, and how far that
closes the gap to rhotol 1e-5.

## Design decisions

Three questions had to be answered before the measurement, namely the frame, the
damping form, and the implementation path.

### Frame

Transverse is defined relative to the current instantaneous total moment
direction. Each iteration the hook reads the G=0 magnetization coefficient of the
input density (the integrated total moment), normalizes it to a unit vector
n_hat, and splits the m-residual at every G into a component along n_hat (the
longitudinal, or parallel, part) and the perpendicular remainder (the
transverse part). Only the transverse part is damped. When the total moment is
near zero the frame falls back to the lab z axis. The lab frame (transverse is
the m_x, m_y channels, parallel is m_z) is measured on the z-seeded Ni cases as a
control, where the two frames should agree because the moment sits on z.

### Damping form

The damping is the Kerker mirror kernel D(G) = G^2 / (G^2 + q0^2) applied to the
transverse residual. D goes to zero as G goes to zero, so the soft long-wavelength
modes take small steps, and D goes to one at high G, so the short-wavelength
modes are untouched. This is the reverse of charge Kerker, which suppresses the
low-G charge modes because the Hartree kernel amplifies them. Here the exchange
has near-zero restoring force at q to 0, so the mixer must step small there.

The G=0 coefficient is never damped, in any direction. The uniform component of m
is the total moment, which has to stay free to grow, rotate, and select a branch.
Damping G=0 would freeze branch selection. This is the campaign's "reverse-Kerker
on the magnon-soft sector, G != 0 only".

The crossover q0 is scanned over three values bracketing the observed soft-sector
extent. The campaign measured 0.5 to 0.8 of the transverse residual power in the
bottom two of twelve linear |G| shells. For these cells the density sphere runs
to |G|max about 24 1/Ang, so the bottom two shells reach about 4 1/Ang. The scan
uses q0 = 2, 4, 8 1/Ang, which places the D = 0.5 crossover below, at, and above
that soft-sector edge.

The null hypothesis is a flat transverse step reduction, a constant factor
alpha_perp < 1 applied to the whole transverse residual at G != 0, with no G
dependence. If the flat form matches the Kerker form, the reverse-Kerker
structure is unnecessary and the G dependence is not what matters.

### Implementation

The driver exposes `scf_noncollinear(..., mixer_hook=...)`, called each iteration
with the raw packed (vin, vout) before `mixer.step`. The packed layout is
[rho, m_x, m_y, m_z], each `ng` complex G-space coefficients. The hook rewrites
vout in place so the residual the mixer sees, r = vout - vin, is damped on the
transverse low-G modes. The mixer takes a step proportional to that residual, so
scaling a residual component scales the step there. The charge block and the
longitudinal m component pass through untouched.

The hook alone is enough. No mixer class had to be wrapped or replaced, and no
src file was edited. The convergence gate reads the raw residual `res_norm =
||vout - vin|| * vol` computed BEFORE the hook fires (`scf/noncollinear.py:762`,
gate at `:787`, hook at `:808`), so the damping cannot lower the gated residual
artificially. A run only passes rhotol if the true residual over all channels,
transverse included, actually falls. Convergence stays honest.

Files. `damping.py` is the `TransverseDampingHook`. `probe.py` is the campaign's
`NCConvergenceProbe`, extended with a nearest-atom Voronoi per-atom moment
tracker for the canted kill-criterion. `systems.py` reuses the campaign cells and
fixture pseudos. `run.py` is the measurement matrix. `analyze.py` prints the
tables. Traces are under `results/<host>/`.

## Results

<!-- FILL: floor table, amplification comparison, q0 scan, flat null, canted -->

## Recommendation

<!-- FILL: go/no-go and the exact hook a production src implementation needs -->
