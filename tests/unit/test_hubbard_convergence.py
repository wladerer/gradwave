"""DFT+U convergence aids: occupation-matrix damping (``occ_mix``) and the
linear U-ramp (``u_ramp_iters``).

Large U on a metallic +U manifold diverges by occupation-matrix flip-flop: the
one-step-lagged n_hub produces a U/2-scale level shift far above the smearing
width, so the occupations at E_F flip all-or-nothing each iteration and the SCF
never settles (see docs/manual/hubbard-u.md, issue #206). Two remedies are wired
into BOTH collinear drivers (scf.loop / scf.uspp_loop):

  * occ_mix (β): mix n_hub across iterations, n = (1-β)·n_prev + β·n_new;
  * u_ramp_iters: ramp U_eff linearly from 1/N to full over the first N steps.

β=1.0 and ramp off (the defaults) reproduce today's raw one-step lag bit-for-bit.
These tests cover the mechanism-level helpers, the input surface, and an SCF-level
fixed-point check that damping/ramping reach the SAME converged energy as the
undamped run (so they are convergence aids, not physics changes)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.scf.common import (
    hubbard_u_ramp_scale,
    mix_hubbard_occ,
    validate_hubbard_conv,
)
from tests.helpers import PSEUDOS

# ---------------------------------------------------------- mechanism helpers

def test_u_ramp_scale_linear_then_holds():
    """The ramp climbs 1/N → 1 over the first N iterations then holds at 1.0,
    and is exactly 1.0 at every iteration when the ramp is off."""
    n = 5
    scales = [hubbard_u_ramp_scale(it, n) for it in range(1, 9)]
    assert scales[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert all(s == 1.0 for s in scales[4:])  # holds at full U past iteration N
    # ramp off: 1.0 for every iteration
    assert all(hubbard_u_ramp_scale(it, 0) == 1.0 for it in range(1, 6))


def test_occ_mix_identity_is_bit_for_bit():
    """β=1.0 (or no previous matrix) returns the fresh matrices UNCHANGED — the
    same object, so the default path is provably today's raw one-step lag."""
    prev = [[torch.ones(3, 3, dtype=torch.complex128)]]
    new = [[torch.zeros(3, 3, dtype=torch.complex128)]]
    out = mix_hubbard_occ(prev, new, 1.0)
    assert out[0][0] is new[0][0]  # identity: the fresh object, untouched
    assert mix_hubbard_occ(None, new, 0.3)[0][0] is new[0][0]  # no prev → fresh


def test_occ_mix_convex_combination():
    """A partial mix is the exact convex combination per spin channel/site."""
    prev = [[torch.full((2, 2), 4.0, dtype=torch.complex128)]]
    new = [[torch.zeros(2, 2, dtype=torch.complex128)]]
    out = mix_hubbard_occ(prev, new, 0.25)
    assert torch.allclose(out[0][0], torch.full((2, 2), 3.0, dtype=torch.complex128))


def test_occ_mix_contracts_flip_flop():
    """The core divergence signature: an occupation matrix that flips between two
    states A and B every iteration. Undamped (β=1) the carried matrix keeps
    flip-flopping with undiminished amplitude; damped (β<1) the successive carried
    matrices contract toward the A/B midpoint (the fixed point of the average), so
    the amplitude of the oscillation shrinks geometrically."""
    a = torch.zeros(1, 1, dtype=torch.complex128)
    b = torch.ones(1, 1, dtype=torch.complex128)
    mid = 0.5 * (a + b)

    def amplitude(beta, steps=12):
        carried = [[a.clone()]]
        amps = []
        for k in range(steps):
            fresh = [[b.clone()]] if k % 2 == 0 else [[a.clone()]]
            carried = mix_hubbard_occ(carried, fresh, beta)
            amps.append(float((carried[0][0] - mid).abs()))
        return amps

    undamped = amplitude(1.0)
    damped = amplitude(0.3)
    # undamped: the carried matrix IS the fresh one, amplitude stays 0.5 forever
    assert all(abs(x - 0.5) < 1e-12 for x in undamped)
    # damped: the steady-state swing (peak over the settled tail) is far below
    # the undamped 0.5 — the analytic limit-cycle amplitude for β=0.3 is
    # β/(1-(1-β)²)-0.5 ≈ 0.088, so the flip-flop is contracted ~6x.
    assert max(damped[-4:]) < 0.15


# --------------------------------------------------------------- input surface

def _base(extra: str = "") -> str:
    # diamond C (PseudoDojo pseudo carries PP_PSWFC for the p manifold)
    return f"""
structure:
  cell: [[0.0, 1.7835, 1.7835], [1.7835, 0.0, 1.7835], [1.7835, 1.7835, 0.0]]
  positions: {{frac: [[0, 0, 0], [0.25, 0.25, 0.25]]}}
  species: [C, C]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: PD_C_PBE_std.upf}}
ecut: 340.0
kpoints:
  mesh: [2, 2, 2]
scf:
  etol: 1.0e-9
  rhotol: 1.0e-8
{extra}"""


def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return p


def test_hubbard_mapping_form_parses_conv_aids(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base(
        "hubbard:\n"
        "  manifolds:\n    - {species: C, l: 1, u: 4.5}\n"
        "  occ_mix: 0.3\n  u_ramp_iters: 5\n")))
    assert inp.hubbard.enabled
    assert len(inp.hubbard.manifolds) == 1
    assert inp.hubbard.occ_mix == 0.3
    assert inp.hubbard.u_ramp_iters == 5


def test_hubbard_list_form_keeps_defaults(tmp_path):
    from gradwave.inputs import load_input

    inp = load_input(_write(tmp_path, _base(
        "hubbard:\n  - {species: C, l: 1, u: 4.5}\n")))
    assert inp.hubbard.occ_mix == 1.0 and inp.hubbard.u_ramp_iters == 0


@pytest.mark.parametrize("extra, needle", [
    ("hubbard:\n  manifolds:\n    - {species: C, l: 1, u: 4.0}\n  occ_mix: 0.0\n",
     "occ_mix"),
    ("hubbard:\n  manifolds:\n    - {species: C, l: 1, u: 4.0}\n  occ_mix: 1.5\n",
     "occ_mix"),
    ("hubbard:\n  manifolds:\n    - {species: C, l: 1, u: 4.0}\n  u_ramp_iters: -2\n",
     "u_ramp_iters"),
    ("hubbard:\n  manifolds:\n    - {species: C, l: 1, u: 4.0}\n  bogus: 1\n",
     "unknown key"),
    ("hubbard:\n  occ_mix: 0.3\n", "must be a list"),
])
def test_hubbard_conv_validation_errors(tmp_path, extra, needle):
    from gradwave.inputs import InputError, load_input

    with pytest.raises(InputError, match=needle):
        load_input(_write(tmp_path, _base(extra)))


def test_validate_hubbard_conv_rejects_bad_values():
    for beta in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="occ_mix"):
            validate_hubbard_conv(beta, 0)
    for n in (-1, 2.5):
        with pytest.raises(ValueError, match="u_ramp_iters"):
            validate_hubbard_conv(0.5, n)
    validate_hubbard_conv(1.0, 0)  # defaults are valid
    validate_hubbard_conv(0.3, 5)


# ------------------------------------------------- SCF-level fixed-point checks

@pytest.mark.standard
def test_occ_mix_and_ramp_reach_same_fixed_point(tmp_path):
    """occ_mix<1 and a U-ramp are convergence aids, not physics: on a system
    that converges under all three settings, the raw one-step lag (β=1, no ramp),
    the damped run (β=0.3), and the ramped run (u_ramp_iters=4) must land on the
    SAME converged +U energy and Dudarev E_U (the fixed point is unchanged)."""
    from gradwave import api
    from gradwave.inputs import load_input

    def run(extra_conv: str):
        body = _base(
            "hubbard:\n"
            "  manifolds:\n    - {species: C, l: 1, u: 4.0}\n" + extra_conv)
        (tmp_path / "in.yaml").unlink(missing_ok=True)
        return api.run_scf(load_input(_write(tmp_path, body)), verbose=False)

    base = run("")                        # β=1.0, ramp off (today's behavior)
    damped = run("  occ_mix: 0.3\n")      # occupation-matrix damping
    ramped = run("  u_ramp_iters: 4\n")   # linear U-ramp

    assert base.converged and damped.converged and ramped.converged
    for other in (damped, ramped):
        assert abs(float(other.energies.free_energy)
                   - float(base.energies.free_energy)) < 1e-6
        assert abs(float(other.energies.hubbard)
                   - float(base.energies.hubbard)) < 1e-6


@pytest.mark.standard
def test_ramp_reports_full_u_energy(tmp_path):
    """The ramp must not report a partial-U energy: with u_ramp_iters=4 the final
    Dudarev E_U equals the full-U (no-ramp) value, confirming convergence was
    blocked until the ramp completed (a run reporting E_U at 1/4 U would differ
    by ~4x)."""
    from gradwave import api
    from gradwave.inputs import load_input

    def e_hub(extra_conv: str):
        body = _base(
            "hubbard:\n"
            "  manifolds:\n    - {species: C, l: 1, u: 6.0}\n" + extra_conv)
        (tmp_path / "in.yaml").unlink(missing_ok=True)
        r = api.run_scf(load_input(_write(tmp_path, body)), verbose=False)
        assert r.converged
        return float(r.energies.hubbard)

    assert abs(e_hub("  u_ramp_iters: 4\n") - e_hub("")) < 1e-6


def test_scf_rejects_bad_occ_mix_directly():
    """The driver validates the convergence aids up front (not only the input
    layer): a bad β raised straight through scf() before any SCF work."""
    from gradwave.core.hubbard import HubbardManifold
    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system
    from tests.helpers import RY

    c = parse_upf(PSEUDOS / "PD_C_PBE_std.upf")
    system = setup_system(3.567 * np.eye(3), np.zeros((1, 3)), [0], [c],
                          ecut=25 * RY, kmesh=(1, 1, 1), use_symmetry=False)
    with pytest.raises(ValueError, match="occ_mix"):
        scf(system, PBE(), hubbard=[HubbardManifold(0, l=1, u=4.0)],
            hub_occ_mix=0.0, verbose=False, max_iter=1)
