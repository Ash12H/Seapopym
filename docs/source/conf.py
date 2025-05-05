# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Seapopym"
copyright = "2024, Jules Lehodey"
author = "Jules Lehodey"

extensions = ["sphinx.ext.autodoc", "sphinx_rtd_theme", "nbsphinx"]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {"collapse_navigation": False, "navigation_depth": 4}

# __init__.py documentation -- Remove if not needed
autoclass_content = "both"

# Include custom CSS
html_css_files = ["custom.css"]
