"""Logging seam for gradwave.

The library follows the standard convention for a library: every module emits
records through a module-level ``logging.getLogger(__name__)`` under the
``gradwave`` namespace, and the package attaches a ``NullHandler`` so nothing is
printed unless the *application* opts in. Default behaviour is therefore
unchanged — the existing ``verbose`` prints still drive user-facing output, and
the ``logger.debug`` / ``logger.warning`` calls added at the SCF, eigensolver,
mixer, and pseudopotential branch points stay silent.

To see the diagnostics, either configure the stdlib ``logging`` module yourself
or call :func:`configure_logging` (the CLI ``--log-level`` flag does this):

    >>> import gradwave
    >>> gradwave.configure_logging("DEBUG")   # stream gradwave.* to stderr

Records are namespaced by module (``gradwave.scf.loop``, ``gradwave.scf.mixing``,
``gradwave.solvers.davidson``, ``gradwave.pseudo`` …) so a caller can raise or
silence one subsystem independently.
"""

from __future__ import annotations

import logging

_ROOT = "gradwave"

# a handler this module installed, tracked so repeat calls replace rather than
# stack duplicates
_managed_handler: logging.Handler | None = None


def configure_logging(level: int | str = "INFO", *, stream=None) -> logging.Logger:
    """Route ``gradwave.*`` log records to a stream (stderr by default).

    Convenience for interactive/CLI use so the DEBUG diagnostics are reachable
    without hand-configuring the stdlib ``logging`` module. Idempotent: a handler
    installed by a previous call is removed first, so calling this repeatedly (or
    from both the CLI and a notebook) never duplicates output. Pass ``level=None``
    to detach the handler and return to the silent NullHandler-only default.

    Returns the ``gradwave`` root logger.
    """
    global _managed_handler
    logger = logging.getLogger(_ROOT)
    if _managed_handler is not None:
        logger.removeHandler(_managed_handler)
        _managed_handler = None
    if level is None:
        return logger
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    _managed_handler = handler
    return logger


def _install_null_handler() -> None:
    """Attach a NullHandler to the package root so records are silent by default
    and Python never emits the 'No handlers could be found' warning."""
    logging.getLogger(_ROOT).addHandler(logging.NullHandler())
