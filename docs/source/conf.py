# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from importlib import metadata

project = 'Spright'
copyright = '2023, Hannu Parviainen'
author = 'Hannu Parviainen'
release = metadata.version('spright')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.mathjax',
              'matplotlib.sphinxext.plot_directive']

# -- Options for the matplotlib plot directive -------------------------------
# The figures in the documentation are created at build time from the scripts
# in the 'figures' directory and from the code examples in the text.

plot_include_source = False
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [('png', 150)]
plot_rcparams = {'font.size': 9,
                 'axes.spines.top': False,
                 'axes.spines.right': False,
                 'figure.facecolor': 'white',
                 'figure.figsize': (7, 3.4),
                 'figure.constrained_layout.use': True}
plot_apply_rcparams = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'furo'
html_static_path = ['_static']
