"""MAEResult save/load round-trip: every field survives, meta included."""

import torch

from gradwave.postscf.mae import MAEResult


def test_mae_result_roundtrip(tmp_path):
    eigs = [torch.randn(4, 6, dtype=torch.float64),
            torch.randn(3, 6, dtype=torch.float64)]
    f_band = torch.tensor([-1.5, -1.4997], dtype=torch.float64)
    src = MAEResult(directions=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                    band_free_energies=f_band, mae=f_band - f_band[0],
                    fermi=[0.31, 0.32], eigenvalues=eigs, nk=[4, 3],
                    meta={"kmesh": [2, 2, 2], "ecut_eV": 408.0})
    path = tmp_path / "mae.pt"
    src.save(path)

    back = MAEResult.load(path)
    assert back.directions == src.directions
    assert torch.equal(back.band_free_energies, src.band_free_energies)
    assert torch.equal(back.mae, src.mae)
    assert back.fermi == src.fermi
    assert back.nk == src.nk
    assert back.meta == src.meta
    for a, b in zip(back.eigenvalues, src.eigenvalues, strict=True):
        assert torch.equal(a, b)

    # save-time meta override replaces the stored dict
    src.save(path, meta={"note": "override"})
    assert MAEResult.load(path).meta == {"note": "override"}
