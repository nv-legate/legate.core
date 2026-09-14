# noqa: INP001
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import inspect
from datetime import datetime
from os import getenv
from typing import Any

SWITCHER_PROD = "https://docs.nvidia.com/legate/switcher.json"
SWITCHER_DEV = "http://localhost:8000/legate/switcher.json"
JSON_URL = SWITCHER_DEV if getenv("SWITCHER_DEV") == "1" else SWITCHER_PROD

ANNOTATE = getenv("LEGATE_ANNOTATION_DOCS") == "1"

# This final documentation snapshot always displays the final release line.
BASE_VERSION = "26.06"

# make sure BASE VERSION is formatted as expected
_yy, _mm = BASE_VERSION.split(".")
assert _yy.isdigit()
assert _mm.isdigit()

# -- Project information -----------------------------------------------------

project = "NVIDIA legate"
copyright = f"2021-{datetime.now().year}, NVIDIA"  # noqa: A001
author = "NVIDIA Corporation"

version = release = BASE_VERSION

# -- General configuration ---------------------------------------------------

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "myst_parser",
    "legate._sphinxext.settings",
    "legate._sphinxext.releases",
]

suppress_warnings = ["ref.myst", "duplicate_declaration.cpp"]
exclude_patterns = [
    # Without this, Sphinx will emit warnings saying "dev.rst not included in
    # any toctree". But dev.rst is a symlink to the latest version, so we don't
    # care that it's not included anywhere, because whatever it links to is.
    "changes/dev.rst"
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

cpp_id_attributes = [
    "LEGATE_EXPORT",
    "LEGATE_PYTHON_EXPORT",
    "LEGATE_NO_EXPORT",
]

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "announcement": (
        "This project has reached end of life and is no longer maintained or "
        "supported. The final release is 26.06.01. This "
        "documentation is retained for historical reference."
    ),
    "switcher": {
        "json_url": JSON_URL,
        "navbar_start": ["navbar-logo", "version-switcher"],
        "version_match": BASE_VERSION,
    },
    "extra_footer": [
        '<script type="text/javascript">if (typeof _satellite !== "undefined"){ _satellite.pageBottom();}</script>'  # NOQA: E501
    ],
    "show_version_warning_banner": True,
}

# -- Options for extensions --------------------------------------------------

autosummary_generate = True
templates_path = ["_templates"]
napoleon_include_special_with_doc = True

breathe_default_project = "legate"
breathe_default_members = ("members", "protected-members")

copybutton_prompt_text = ">>> "

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

pygments_style = "sphinx"

myst_heading_anchors = 3


def _dont_skip_documented_dunders(  # noqa: PLR0913, PLR0911
    app: Any,  # noqa: ARG001
    what: str,
    name: str,
    obj: Any,
    skip: bool,  # noqa: ARG001, FBT001
    options: dict[str, Any],  # noqa: ARG001
) -> bool | None:
    SKIP = True  # definitely skip the value (does not show up in docs)
    KEEP = False  # definitely do NOT skip the value (will show up in docs)
    LET_AUTODOC_DECIDE = None  # fall back to autodoc (might show up in docs)

    if not name.startswith("_"):
        # Not a dunder, defer to default autodoc-skip-member
        return LET_AUTODOC_DECIDE
    if obj is getattr(object, name, None):
        # The function is actually object.__foo__, which, obviously we did not
        # write. Defer to autodoc.
        return LET_AUTODOC_DECIDE
    if inspect.isbuiltin(obj):
        # Some object.__foo__ methods aren't caught by the above, but they are
        # caught by this.
        return LET_AUTODOC_DECIDE
    if not hasattr(obj, "__doc__"):
        return LET_AUTODOC_DECIDE

    if "pyx_vtable" in name or "_cython_" in repr(obj):
        # Cython implementation details, we never want these
        return SKIP

    if any(
        item in name
        for item in (
            "array_interface",
            "legate_data_interface",
            "__init__",
            "__doc__",
        )
    ):
        # We always want these
        return KEEP

    match what:
        case "method":
            return KEEP
        case _:
            print(  # noqa: T201
                f"====== LET AUTODOC DECIDE ({what})", name, "->", obj
            )
            return LET_AUTODOC_DECIDE


def setup(app):  # noqa: D103
    if ANNOTATE:
        app.add_js_file("https://hypothes.is/embed.js", kind="hypothesis")
    app.add_css_file("params.css")
    app.connect("autodoc-skip-member", _dont_skip_documented_dunders)
