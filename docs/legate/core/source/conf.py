# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "legate.core"
copyright = "2021-2023, NVIDIA"
author = "NVIDIA"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "myst_parser",
    "legate._sphinxext.settings",
]

suppress_warnings = ["ref.myst"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "footer_start": ["copyright"],
    "github_url": "https://github.com/nv-legate/legate.core",
    # https://github.com/pydata/pydata-sphinx-theme/issues/1220
    "icon_links": [],
    "logo": {
        "text": project,
        "link": "https://nv-legate.github.io/legate.core/",
    },
    "navbar_align": "left",
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "primary_sidebar_end": ["indices.html"],
    "secondary_sidebar_items": ["page-toc"],
    "show_nav_level": 2,
    "show_toc_level": 2,
}

# -- Options for extensions --------------------------------------------------

autosummary_generate = True

copybutton_prompt_text = ">>> "

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

pygments_style = "sphinx"

myst_heading_anchors = 3


def setup(app):
    app.add_css_file("params.css")
