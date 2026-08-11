"""YAML input schema → validated frozen dataclasses (Layer C).

Geometry goes through ASE, so `structure:` accepts any format ASE reads
(cif, POSCAR, xyz, ...) or an inline cell/positions/species block.
All energies in eV, lengths in Å (the package-wide convention).

Formerly a single module, now a package: ``models`` holds the dataclass
schema, ``parse`` the loading/validation. This __init__ re-exports the full
historical ``gradwave.inputs`` surface.
"""

from gradwave.inputs.models import BandsParams as BandsParams
from gradwave.inputs.models import CohpParams as CohpParams
from gradwave.inputs.models import DispersionParams as DispersionParams
from gradwave.inputs.models import ElasticParams as ElasticParams
from gradwave.inputs.models import EOSParams as EOSParams
from gradwave.inputs.models import HubbardManifoldSpec as HubbardManifoldSpec
from gradwave.inputs.models import HubbardParams as HubbardParams
from gradwave.inputs.models import HybridParams as HybridParams
from gradwave.inputs.models import Input as Input
from gradwave.inputs.models import InputError as InputError
from gradwave.inputs.models import KPointsParams as KPointsParams
from gradwave.inputs.models import MagneticParams as MagneticParams
from gradwave.inputs.models import MagnetismParams as MagnetismParams
from gradwave.inputs.models import MixingParams as MixingParams
from gradwave.inputs.models import PhononParams as PhononParams
from gradwave.inputs.models import ProjectionsParams as ProjectionsParams
from gradwave.inputs.models import RelaxParams as RelaxParams
from gradwave.inputs.models import SCFParams as SCFParams
from gradwave.inputs.models import SmearingParams as SmearingParams
from gradwave.inputs.models import VolumetricParams as VolumetricParams
from gradwave.inputs.parse import _load_structure as _load_structure
from gradwave.inputs.parse import _normalize_kerker as _normalize_kerker
from gradwave.inputs.parse import load_input as load_input

__all__ = [
    "BandsParams",
    "CohpParams",
    "DispersionParams",
    "EOSParams",
    "ElasticParams",
    "HubbardManifoldSpec",
    "HubbardParams",
    "HybridParams",
    "Input",
    "InputError",
    "KPointsParams",
    "MagneticParams",
    "MagnetismParams",
    "MixingParams",
    "PhononParams",
    "ProjectionsParams",
    "RelaxParams",
    "SCFParams",
    "SmearingParams",
    "VolumetricParams",
    "load_input",
]
