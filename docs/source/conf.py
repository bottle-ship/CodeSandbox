# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import inspect
import os
import pkgutil
import shutil
import sys
from collections import OrderedDict
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


# -- Generate API Reference documentation ------------------------------------


def get_package_modules(package_name: str):
    package = __import__(package_name)

    module_name_list = [
        module.name
        for module in pkgutil.walk_packages(
            path=package.__path__, prefix=package.__name__ + "."
        )
    ]

    module_list = list()
    for module_name in module_name_list:
        if ".tests." in module_name or module_name.endswith(".tests"):
            continue

        module = importlib.import_module(module_name)

        if os.path.basename(module.__file__) != "__init__.py" and hasattr(
            module, "__all__"
        ):
            continue

        module_list.append(module)

    return module_list


def write_autosummary(module) -> str:
    members = OrderedDict(inspect.getmembers(module))

    module_name = members["__name__"]

    supported_section_lines = ["=", "-", "*"]
    depth = len(module_name.split("."))
    section_line = supported_section_lines[depth - 2]
    section_lines = section_line * len(module_name)

    autosummary = f"{module_name}\n{section_lines}\n"
    autosummary += "\n"
    autosummary += (
        f".. automodule:: {module_name}\n    :no-members:\n    :no-inherited-members:\n"
    )
    autosummary += "\n"

    package_name = module_name.split(".")[0]
    sub_package_name = ".".join(module_name.split(".")[1:])

    autosummary += f".. currentmodule:: {package_name}\n"
    autosummary += "\n"

    class_names = list()
    func_names = list()
    for key in members.keys():
        if key.startswith("__") and key.endswith("__"):
            continue

        if inspect.isclass(members[key]):
            class_names.append(members[key].__qualname__)

        if inspect.isfunction(members[key]):
            func_names.append(members[key].__qualname__)

    if len(class_names) > 0:
        autosummary += f".. autosummary::\n    :toctree: generated/\n"
        autosummary += "\n"
        for class_name in class_names:
            autosummary += f"    {sub_package_name}.{class_name}\n"
        autosummary += "\n"

    if len(func_names) > 0:
        autosummary += f".. autosummary::\n    :toctree: generated/\n"
        autosummary += "\n"
        for func_name in func_names:
            autosummary += f"    {sub_package_name}.{func_name}\n"
        autosummary += "\n"

    return autosummary


def generate_api_reference_documentation():
    section_title = "API Reference"
    section_lines = "=" * len(section_title)
    description = "This page contains the API reference for public objects and functions in learning-kit."

    api_doc = ".. _api_reference:\n\n"
    api_doc += f"{section_lines}\n{section_title}\n{section_lines}\n\n"
    api_doc += f"{description}\n\n"

    for module in get_package_modules("learning_kit"):
        api_doc += write_autosummary(module)

    with open(os.path.join(os.path.dirname(__file__), "api.rst"), "w") as f:
        f.write(api_doc)

    shutil.rmtree(os.path.join(os.path.dirname(__file__), "generated"))


generate_api_reference_documentation()


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
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx_copybutton",
    "myst_parser",
]

autodoc_default_options = {"members": True, "inherited-members": True}

# Make sure the autogenerated targets are unique
autosectionlabel_prefix_document = True

# generate autosummary even if no references
autosummary_generate = True

numpydoc_show_class_members = False

# Do not create toctree entries for each class/function
toc_object_entries = False

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
html_theme_options = {
    # "logo_only": True,
    # "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#343131",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "_static/img/pytorch-logo-dark-unstable.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def clear_build_dir(app):
    import shutil

    build_dir = os.path.join(Path(__file__).parents[1], "build")
    for sub_dir in os.listdir(build_dir):
        shutil.rmtree(os.path.join(build_dir, sub_dir))


def setup(app):
    app.connect("builder-inited", clear_build_dir)
