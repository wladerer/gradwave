"""Stage-2 refactor gate: each mixer, fed a recorded sequence of real
(rho_in, rho_out) pairs from a Si SCF, must reproduce its recorded
outputs bit-exactly. Consolidating the mixer classes must not move a
single ulp. Regenerate the fixture ONLY for an intentional algorithm
change (scratchpad script gen_mixer_traj.py in the session notes;
seed 3, alpha 0.6, history 5)."""

from pathlib import Path

import torch

from gradwave.scf.mixing import BroydenMixer, JohnsonMixer, PulayMixer

TRAJ = torch.load(Path(__file__).parents[1] / "fixtures" / "golden"
                  / "mixer_traj.pt", weights_only=False)


def _replay(cls, name, **kw):
    pairs = TRAJ["pairs"]
    n = pairs[0][0].shape[0]
    torch.manual_seed(3)
    m = cls(torch.linspace(0, 8, n, dtype=torch.float64),
            alpha=0.6, history=5, check_g0=False, **kw)
    for (rin, rout), ref in zip(pairs, TRAJ["outputs"][name], strict=True):
        out = m.step(rin, rout)
        # 1e-13 rather than bit-exact: threaded BLAS solves jitter at the
        # ulp level between processes; algorithmic changes move 1e-6+
        assert float((out - ref).abs().max()) < 1e-13, \
            f"{name}: trajectory moved"


def test_pulay_trajectory():
    _replay(PulayMixer, "pulay")


def test_pulay_kerker_trajectory():
    _replay(PulayMixer, "pulay_kerker", kerker=True)


def test_broyden_trajectory():
    _replay(BroydenMixer, "broyden")


def test_johnson_trajectory():
    _replay(JohnsonMixer, "johnson")


def test_johnson_metric_trajectory():
    _replay(JohnsonMixer, "johnson_metric", metric_w=TRAJ["metric_w"])
