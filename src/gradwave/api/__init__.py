"""Python API mirroring the YAML input (Layer C).

run() executes a task and writes three files into the output directory:
<task>.json (the machine-readable summary — the parsing target),
<task>.out (the human-readable report) and, for SCF tasks, checkpoint.pt
(restartable state, wavefunctions excluded unless requested). The same
summary dict feeds gradwave.io.analysis for pandas/matplotlib work.

Formerly a single ~3k-line module, now a package; the implementation lives
in the sibling modules (system, scf, relax, eos, elastic, phonons,
dispersion, summary, dispatch). This __init__ re-exports the public surface
plus the underscore helpers that tests and tooling import from the old
``gradwave.api`` path.
"""

from gradwave.api._common import SPIN_XC_REGISTRY as SPIN_XC_REGISTRY
from gradwave.api._common import XC_REGISTRY as XC_REGISTRY
from gradwave.api._common import _mixing_scheme as _mixing_scheme
from gradwave.api.converge import recommend_convergence as recommend_convergence
from gradwave.api.dispatch import run as run
from gradwave.api.dispatch import run_magnetism as run_magnetism
from gradwave.api.dispersion import _compute_dispersion as _compute_dispersion
from gradwave.api.elastic import run_elastic as run_elastic
from gradwave.api.eos import run_eos as run_eos
from gradwave.api.flapw import run_flapw as run_flapw
from gradwave.api.flapw import run_nmr as run_nmr
from gradwave.api.phonons import run_phonons as run_phonons
from gradwave.api.relax import _build_relax_calc as _build_relax_calc
from gradwave.api.relax import _joint_supported as _joint_supported
from gradwave.api.relax import _resolve_pulay_correction as _resolve_pulay_correction
from gradwave.api.relax import run_relax as run_relax
from gradwave.api.scf import _run_scf_noncollinear as _run_scf_noncollinear
from gradwave.api.scf import run_scf as run_scf
from gradwave.api.summary import _bands_extra as _bands_extra
from gradwave.api.summary import _write_volumetric as _write_volumetric
from gradwave.api.summary import build_summary as build_summary
from gradwave.api.system import _is_uspp as _is_uspp
from gradwave.api.system import _load_upf as _load_upf
from gradwave.api.system import build_system as build_system

__all__ = [
    "SPIN_XC_REGISTRY",
    "XC_REGISTRY",
    "build_summary",
    "build_system",
    "recommend_convergence",
    "run",
    "run_eos",
    "run_elastic",
    "run_flapw",
    "run_magnetism",
    "run_nmr",
    "run_phonons",
    "run_relax",
    "run_scf",
]
