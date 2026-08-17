"""The bench telemetry additions: mixer subspace depth + per-iteration RSS land
in the flight-recorder trace, and the harness produces an analyzable RunRecord.
"""

import numpy as np
import pytest

from gradwave.bench.harness import run_case
from gradwave.bench.suite import _PSEUDO_DIR, RY, BenchCase, _fcc
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.mixing import PulayMixer

pytestmark = pytest.mark.standard


def _tiny_si_case():
    a = 5.431
    upf = parse_upf(_PSEUDO_DIR / "Si_ONCV_PBE-1.2.upf")

    def build():
        return setup_system(_fcc(a), np.array([[0.0, 0, 0], [a / 4] * 3]), [0, 0],
                            [upf], ecut=18 * RY, kmesh=(2, 2, 2), nbands=8)

    return BenchCase("Si-tiny", "insulator", build,
                     dict(smearing="none", max_iter=18, etol=1e-5, rhotol=1e-4, verbose=False),
                     dict(hardness="insulator", n_atoms=2))


def test_mixer_exposes_subspace_size():
    m = PulayMixer(np.ones(4), alpha=0.7, history=8)
    assert m.subspace_size == 0                      # nothing accrued yet


def test_recorder_trace_carries_subspace_and_rss():
    rec = run_case(_tiny_si_case(), {"mixing_scheme": "pulay", "mixing_alpha": 0.7, "kerker": True})
    assert rec.outcome["error"] is None, rec.outcome["error"]
    iters = rec.trace["iterations"]
    assert len(iters) >= 3
    # both new fields present on every iteration
    assert all("subspace_size" in it and "rss_mb" in it for it in iters)
    # the Pulay subspace builds up past zero as history accrues
    assert max(it["subspace_size"] or 0 for it in iters) > 0
    # RSS is a positive number on Linux (None-tolerant elsewhere)
    rss = [it["rss_mb"] for it in iters if it["rss_mb"] is not None]
    assert not rss or all(x > 0 for x in rss)
    # op-counts + eigensolve phase time populate and are non-trivial
    assert all(it["n_fft"] and it["n_hpsi"] and it["n_eigh"] for it in iters)
    assert all(it["t_eig_s"] is not None and it["t_eig_s"] >= 0 for it in iters)


def test_run_record_is_analyzable():
    method = {"mixing_scheme": "broyden", "mixing_alpha": 0.5, "kerker": True}
    rec = run_case(_tiny_si_case(), method)
    assert rec.case == "Si-tiny" and rec.hardness == "insulator"
    assert "wall_s" in rec.outcome
    assert rec.provenance.get("host")
