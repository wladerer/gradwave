"""Ordering classification and report formatting for the magnetic-characterization
routine (postscf/magnetism). Pure logic — no SCF — so it runs in the fast gate."""

import torch

from gradwave.postscf.magnetism import MagneticReport, _classify

Z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)


def test_classify_nonmagnetic():
    M = 0.01 * torch.stack([Z, Z])
    assert _classify(M, torch.linalg.norm(M, dim=-1), None) == "nonmagnetic"


def test_classify_from_exchange_sign():
    M = torch.stack([Z, Z])
    mags = torch.linalg.norm(M, dim=-1)
    assert _classify(M, mags, {1: +0.02}) == "ferromagnetic"
    assert _classify(M, mags, {1: -0.02}) == "antiferromagnetic"
    # the dominant (largest |J|) coupling decides
    assert _classify(M, mags, {1: -0.03, 2: +0.01}) == "antiferromagnetic"


def test_classify_from_directions_without_exchange():
    par = torch.stack([Z, Z])                                  # parallel
    anti = torch.stack([Z, -Z])                                # antiparallel
    obl = torch.stack([Z, torch.tensor([1.0, 0, 0], dtype=torch.float64)])
    assert _classify(par, torch.linalg.norm(par, dim=-1), None).startswith("ferromagnetic")
    assert _classify(anti, torch.linalg.norm(anti, dim=-1), None).startswith("antiferromagnetic")
    assert _classify(obl, torch.linalg.norm(obl, dim=-1), None) == "non-collinear"


def test_report_summary_contains_key_fields():
    r = MagneticReport(
        moment_magnitudes=[1.0, 1.0], moment_vectors=[[0, 0, 1.0], [0, 0, 1.0]],
        total_moment=2.0, ordering="ferromagnetic", exchange_J={1: 1.434},
        dmi={1: 0.0}, curie_temperature_mfa=11090.0, ref_atom=0)
    s = r.summary()
    assert "ferromagnetic" in s
    assert "2.000 μB" in s
    assert "J_01 = +1434.0" in s
    assert "T_c" in s and "11090 K" in s
