"""Shared KB/PAW projector assembly at an arbitrary k-point.

Several analysis modules diagonalize the Hamiltonian at off-mesh k-points
(band structures, irreps, dielectric response, USPP bands) and each rebuilt the
same projector scaffold: the per-species angular momenta and D-matrices, plus
the radial beta form factors re-evaluated at ``|k+G|``. This collects that into
one place so the copies cannot drift.

Setup only — none of this sits on an autograd path. The callers are all
``@torch.no_grad`` or return numpy, so the tensors here carry no gradient.
"""

from __future__ import annotations

import torch

from gradwave.core.hamiltonian import build_projector_data
from gradwave.dtypes import RDTYPE
from gradwave.pseudo.kb import beta_form_factors


def species_projector_tables(pseudos, device=None):
    """Per-species ``(beta_ls, dij)`` tables for projector assembly.

    ``beta_ls[s]`` lists the angular momentum ``l`` of each beta channel of
    species ``s``; ``dij[s]`` is its ``(nchan, nchan)`` D-matrix [eV]. Both are
    k-independent, so a caller hoists this once outside its per-k loop rather
    than rebuilding it at every k-point.

    ``pseudos`` is ``system.upfs`` (norm-conserving) or ``system.paws`` (USPP).
    """
    beta_ls = [[b.l for b in p.betas] for p in pseudos]
    dij = [torch.as_tensor(p.dij, dtype=RDTYPE, device=device) for p in pseudos]
    return beta_ls, dij


def projector_data_at_k(sphere, species_of_atom, pseudos, beta_ls, dij, volume,
                        device=None):
    """KB/PAW ``ProjectorData`` for one G-sphere at its own ``k+G``.

    ``q = |sphere.kpg|`` drives the radial beta form factors. ``sphere`` may be
    a lightweight shim carrying a shifted ``kpg`` (as the dielectric response
    does), in which case the form factors and Ylm are evaluated at the shifted
    vectors. ``beta_ls``/``dij`` come from :func:`species_projector_tables`.

    Returns the assembled ``ProjectorData``; no autograd path.
    """
    q = torch.linalg.norm(sphere.kpg, dim=1).cpu().numpy()
    beta_tables = [
        torch.as_tensor(beta_form_factors(p, q), dtype=RDTYPE, device=device)
        for p in pseudos
    ]
    return build_projector_data(
        sphere, species_of_atom, beta_tables, beta_ls, dij, volume
    )
