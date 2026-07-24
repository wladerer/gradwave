"""gradwave: differentiable plane-wave DFT in PyTorch.

Two entry points mirror one another:

- the YAML-driven pipeline: ``load_input`` parses an input file into an
  ``Input``, ``run`` executes the requested task and returns the summary dict;
- the ASE-style ``GradWave`` calculator for programmatic use.

``analysis.load`` reads a run's JSON summary back into plain dicts/frames for
plotting. Submodules (``core``, ``scf``, ``postscf``, ``pseudo``, ``solvers``)
hold the physics layers and are imported directly when needed.
"""

from gradwave._logging import _install_null_handler, configure_logging
from gradwave._threads import apply_default_threads, set_num_threads
from gradwave.api import run
from gradwave.calculator import GradWave
from gradwave.inputs import Input, InputError, load_input

# library convention: emit through gradwave.* loggers, silent until the caller
# opts in (configure_logging or their own logging config). See gradwave._logging.
_install_null_handler()

# Pick a sane intra-op CPU thread default (min(cores, 8)) instead of inheriting
# the caller's nproc: the latency-bound SCF is 2-3x slower fanned out across all
# cores of a many-core / hybrid CPU. Honours GRADWAVE_NUM_THREADS and never
# clobbers an explicit OMP/MKL/OpenBLAS choice. See gradwave._threads.
apply_default_threads()

__version__ = "0.1.0"

__all__ = ["GradWave", "Input", "InputError", "__version__", "configure_logging",
           "load_input", "run", "set_num_threads"]
