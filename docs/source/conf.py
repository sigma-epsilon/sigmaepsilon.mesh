# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# For ideas:

# https://github.com/pradyunsg/furo/blob/main/docs/conf.py
# https://github.com/sphinx-gallery/sphinx-gallery/blob/master/doc/conf.py

# --------------------------------------------------------------------------

import sys
import os
from datetime import date
import warnings

import sigmaepsilon.mesh as library

from sphinx.config import Config

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../src"))

from doc_utils import generate_examples_gallery_rst

generate_examples_gallery_rst(
    title="Examples", filename="examples_gallery", foldername="examples", reversed=True
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = library.__pkg_name__
copyright = "2014-%s, Bence Balogh" % date.today().year
author = "Bence Balogh"


def setup(app: Config):
    app.add_config_value("project_name", project, "html")


# The short X.Y version.
version = library.__version__
# The full version, including alpha/beta/rc tags.
release = "v" + library.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # allows to work with markdown files
    "myst_parser",  # pip install myst-parser for this
    # to plot summary about durations of file generations
    "sphinx.ext.duration",
    # to test code snippets in docstrings
    "sphinx.ext.doctest",
    # for automatic exploration of the source files
    "sphinx.ext.autodoc",
    # to enable cross referencing other documents on the internet
    "sphinx.ext.intersphinx",
    # Napoleon is a extension that enables Sphinx to parse both NumPy and Google style docstrings
    "sphinx.ext.napoleon",
    #'sphinx_gallery.gen_gallery',
    #'sphinx_gallery.load_style',  # load CSS for gallery (needs SG >= 0.6)
    "nbsphinx",  # to handle jupyter notebooks
    # "nbsphinx_link",  # for including notebook files from outside the sphinx source root
    "sphinx_copybutton",  # for "copy to clipboard" buttons
    "sphinx.ext.mathjax",  # for math equations
    "sphinxcontrib.bibtex",  # for bibliographic references
    "sphinxcontrib.rsvgconverter",  # for SVG->PDF conversion in LaTeX output
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_inline_tabs",
]

autosummary_generate = True

templates_path = ["_templates"]

exclude_patterns = ["_build"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

language = "EN"

# See warnings about bad links
nitpicky = True
nitpick_ignore = [
    ("", "Pygments lexer name 'ipython' is not known"),
    ("", "Pygments lexer name 'ipython3' is not known"),
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
pygments_dark_style = "github-dark"
highlight_language = "python3"

intersphinx_mapping = {
    "python": (r"https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": (r"https://numpy.org/doc/stable/", None),
    "scipy": (r"http://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": (r"https://matplotlib.org/stable", None),
    "sphinx": (r"https://www.sphinx-doc.org/en/master", None),
    "pandas": (r"https://pandas.pydata.org/pandas-docs/stable/", None),
    "sigmaepsilon.core": (r"https://sigmaepsiloncore.readthedocs.io/en/latest/", None),
    "sigmaepsilon.math": (r"https://sigmaepsilonmath.readthedocs.io/en/latest/", None),
    "linkeddeepdict": (r"https://linkeddeepdict.readthedocs.io/en/latest/", None),
}

# -- bibtex configuration -------------------------------------------------
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

# If no encoding is specified, utf-8-sig is assumed.
# bibtex_encoding = 'latin'

# -- MathJax Configuration -------------------------------------------------

mathjax3_config = {
    "tex": {"tags": "ams", "useLabelIds": True},
}

# -- Image scapers configuration -------------------------------------------------

image_scrapers = ("matplotlib",)

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = 'docs/source/' + env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/sigmaepsilon/{{ env.config.project_name }}/blob/{{ env.config.release|e }}/{{ docname|e }}">{{ docname|e }}</a>.
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""

# This is processed by Jinja2 and inserted after each notebook
nbsphinx_epilog = r"""
{% set docname = 'docs/source/' + env.doc2path(env.docname, base=None) %}
.. raw:: latex

    \nbsphinxstopnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{\dotfill\ \sphinxcode{\sphinxupquote{\strut
    {{ docname | escape_latex }}}} ends here.}}
"""
