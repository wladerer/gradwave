"""The package version, as a leaf module.

Import __version__ from here inside the library (not from the gradwave root):
the root __init__ imports the api layer, so a root import from a low-level
module creates an indirect dependency on the whole stack — the import-linter
"api is the top of the stack" contract flags it.
"""

__version__ = "0.1.0"
