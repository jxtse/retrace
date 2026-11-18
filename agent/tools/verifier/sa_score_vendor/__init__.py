"""Vendored copy of RDKit Contrib SA_Score (Ertl & Schuffenhauer 2009).

Files copied verbatim from rdkit/Contrib/SA_Score/:
  - sascorer.py
  - fpscores.pkl.gz

Vendoring avoids RDKit Contrib path drift across conda / pip / system installs.
Use ``from .sascorer import calculateScore``.
"""
from . import sascorer  # noqa: F401
