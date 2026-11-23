"""Sphinx configuration for OpenImageGen."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as get_package_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "OpenImageGen"
author = "ParisNeo"
copyright = f"{datetime.now(timezone.utc).year}, {author}"

try:
    release = get_package_version("openimagegen")
except PackageNotFoundError:
    release = "0.3.0"

# Sphinx requires both values to be strings. Keep the imported metadata
# function under a different name so it cannot replace this config value.
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
language = "en"

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "accelerate",
    "diffusers",
    "gguf",
    "huggingface_hub",
    "optimum_quanto",
    "peft",
    "PIL",
    "sentencepiece",
    "torch",
    "transformers",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"OpenImageGen {release}"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "titles_only": False,
}

nitpicky = False


def _normalize_project_header_docstring(app, what, name, obj, options, lines):
    """Make legacy ``@description`` continuation lines valid reStructuredText.

    Project module headers use visual indentation intended for source readers,
    not definition lists. Normalize only that specific continuation pattern;
    ordinary API documentation, parameter blocks, and examples remain intact.
    """
    previous_was_description = False
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if previous_was_description and stripped and line.startswith("             "):
            lines[index] = stripped
        previous_was_description = stripped.startswith("@description")


def setup(app):
    app.connect("autodoc-process-docstring", _normalize_project_header_docstring)