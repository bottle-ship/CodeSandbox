# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime
from pathlib import Path

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, str(Path(__file__).parents[2]))

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.insert(0, os.path.abspath("sphinxext"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "learning_kit"
copyright = f"2023 - {datetime.now().year}"
author = "Byungseon Choi"
# version = ""
release = "0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "add_toctree_functions",
    "sphinx_copybutton",
    "myst_parser"
]

# generate autosummary even if no references
autosummary_generate = True

numpydoc_show_class_members = False

# # this is needed for some reason...
# # see https://github.com/numpy/numpydoc/issues/69
# numpydoc_class_members_toctree = False

# The suffix of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Type aliases for common types
# Sphinx type aliases only works with Postponed Evaluation of Annotations
# (PEP 563) enabled (via `from __future__ import annotations`), which keeps the
# type annotations in string form instead of resolving them to actual types.
# However, PEP 563 does not work well with JIT, which uses the type information
# to generate the code. Therefore, the following dict does not have any effect
# until PEP 563 is supported by JIT and enabled in files.
autodoc_type_aliases = {}

# Specify how to identify the prompt when copying code snippets
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Options for internationalization ----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-internationalization

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "_static/img/pytorch-logo-dark-unstable.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
