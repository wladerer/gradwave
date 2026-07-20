"""Facade for the USPP/PAW machinery (refactor stage 3).

System construction lives in scf/uspp_setup.py, the SCF driver and
per-k generalized Davidson in scf/uspp_loop.py. Everything imports
through this module — tests/conftest.py patches the setup_uspp
attribute HERE for device routing, so the name must stay stable.
"""

from gradwave.scf.results import USPPNCResult, USPPResult  # noqa: F401
from gradwave.scf.uspp_loop import (  # noqa: F401
    _HkS,
    davidson_gen,
    scf_uspp,
)
from gradwave.scf.uspp_setup import (  # noqa: F401
    _MINUS_I_POW_L,
    AugSpecies,
    USPPSystem,
    _aug_tables,
    _make_becsum_sym,
    _mexp_index_map,
    setup_uspp,
)
