# Copyright (c) 2017-2025 Cardiff University
#               2014-2017 Louisiana State University
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

import os.path
import re
import warnings
from datetime import datetime
from datetime.timezone import UTC
from pathlib import Path

import matplotlib
import sphinx_github_style
from numpydoc import docscrape_sphinx
from sphinx.util import logging

import gwpy
from gwpy.utils.sphinx import gallery as cli_gallery

TODAY = datetime.now(tz=UTC).date()

SPHINX_DIR = Path(__file__).parent.absolute()
STATIC_DIRNAME = "_static"

PROJECT_URL = "https://gitlab.com/gwpy/gwpy"

# use caching in GWpy calls
os.environ.update({
    "GWPY_CACHE": "true",
})

# -- versions ---------------

GWPY_VERSION = gwpy.__version__

# parse version number to get git reference
_setuptools_scm_version_regex = re.compile(
    r"\+g(\w+)(?:\Z|\.)",
)
if match := _setuptools_scm_version_regex.search(GWPY_VERSION):
    GWPY_GIT_REF, = match.groups()
else:
    GWPY_GIT_REF = f"v{GWPY_VERSION}"

# -- matplotlib -------------

matplotlib.use("agg")

# ignore warnings that aren't useful for documentation
if "CI" not in os.environ:
    for message, category in (
        (".*non-GUI backend.*", UserWarning),
        (".*gwpy.plot.*", DeprecationWarning),
        ("", matplotlib.MatplotlibDeprecationWarning),
    ):
        warnings.filterwarnings("ignore", message=message, category=category)

# -- general ----------------

needs_sphinx = "4.0"
project = "GWpy"
copyright = f"{TODAY.year} Cardiff University"
version = "dev" if ".dev" in GWPY_VERSION else GWPY_VERSION
release = GWPY_VERSION

# extension modules
# DEVNOTE: please make sure and add 3rd-party dependencies to
#          pyproject.toml's [project.optional-dependencies]/docs
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.imgmath",
    "sphinx.ext.autosummary",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.linkcode",
    "sphinx.ext.ifconfig",
    "sphinx_automodapi.automodapi",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.programoutput",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
]

# content management
default_role = "obj"
exclude_patterns = [
    "references.rst",
    "_build",
    "_generated",
    "cli/_examples*",
]
templates_path = [
    "_templates",
]

# epilog
rst_epilog = "\n.. include:: /references.rst"

# -- HTML formatting --------

html_theme = "furo"
html_title = f"{project} {version}"
html_static_path = [STATIC_DIRNAME]
html_favicon = str(Path(STATIC_DIRNAME) / "favicon.png")
html_logo = str(Path(STATIC_DIRNAME) / "favicon.png")
html_css_files = [
    # GWpy customisations
    "css/gwpy-sphinx.css",
    # add fontawesome for icons
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.2/css/brands.min.css",
]
html_theme_options = {
    # add gitlab link in footer
    "footer_icons": [
        {
            "name": "GitLab",
            "url": PROJECT_URL,
            "class": "fa-brands fa-gitlab",
        },
    ],
    # hide project name in sidebar, we have a logo
    "sidebar_hide_name": True,
    # colours
    "dark_css_variables": {
        "color-brand-content": (gwpy_dark_mode_colour := "#ededed"),
        "color-brand-primary": gwpy_dark_mode_colour,
        "color-brand-visited": gwpy_dark_mode_colour,
    },
    "light_css_variables": {
        "color-brand-content": (gwpy_light_mode_colour := "#29435f"),
        "color-brand-primary": gwpy_light_mode_colour,
        "color-brand-visited": gwpy_light_mode_colour,
    },
}

# -- extensions config ------

# -- autodoc

autoclass_content = "class"
autodoc_default_flags = ["show-inheritance", "members", "inherited-members"]

# -- autosummary

autosummary_generate = True

# -- copybutton

copybutton_prompt_text = " |".join((  # noqa: FLY002
    ">>>",
    r"\.\.\.",
    r"\$"
    r"In \[\d*\]:",
    r" {2,5}\.\.\.:",
    " {5,8}: ",
))
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"

# -- plot_directive

plot_rcparams = dict(matplotlib.rcParams)
plot_rcparams.update({
    "backend": "agg",
})
plot_apply_rcparams = True
plot_formats = ["png"]
plot_include_source = True
plot_html_show_source_link = False

# -- numpydoc

# fix numpydoc autosummary
numpydoc_show_class_members = False

# use blockquotes (numpydoc>=0.8 only)
numpydoc_use_blockquotes = True

# auto-insert plot directive in examples
numpydoc_use_plots = True

# update the plot detection to include .show() calls
parts = re.split(r"[\(\)|]", docscrape_sphinx.IMPORT_MATPLOTLIB_RE)[1:-1]
parts.extend(("fig.show()", "plot.show()"))
docscrape_sphinx.IMPORT_MATPLOTLIB_RE = r"\b({})\b".format("|".join(parts))

# -- inhertiance_diagram

# configure inheritance diagram
inheritance_graph_attrs = {"rankdir": "TB"}

# -- intersphinx

# Intersphinx
intersphinx_mapping = {key: (value, None) for key, value in {
    "astropy": "https://docs.astropy.org/en/stable/",
    "coloredlogs": "https://coloredlogs.readthedocs.io/en/latest/",
    "dateparser": "https://dateparser.readthedocs.io/en/stable/",
    "dateutil": "https://dateutil.readthedocs.io/en/stable/",
    "dqsegdb2": "https://dqsegdb2.readthedocs.io/en/stable/",
    "gssapi": "https://pythongssapi.github.io/python-gssapi/",
    "gwdatafind": "https://gwdatafind.readthedocs.io/en/stable/",
    "gwosc": "https://gwosc.readthedocs.io/en/stable/",
    "h5py": "https://docs.h5py.org/en/latest/",
    "humanfriendly": "https://humanfriendly.readthedocs.io/en/latest/",
    "igwn-auth-utils": "https://igwn-auth-utils.readthedocs.io/en/stable/",
    "igwn-ligolw": "https://igwn-ligolw.readthedocs.io/en/stable/",
    "igwn-segments": "https://igwn-segments.readthedocs.io/en/stable/",
    "ligo.skymap": "https://lscsoft.docs.ligo.org/ligo.skymap/",
    "lscsoft-glue": "https://lscsoft.docs.ligo.org/glue/",
    "matplotlib": "https://matplotlib.org/",
    "numpy": "https://numpy.org/doc/stable/",
    "pycbc": "https://pycbc.org/pycbc/latest/html/",
    "python": "https://docs.python.org/3/",
    "requests-pelican": "https://requests-pelican.readthedocs.io/en/stable/",
    "requests-scitokens": "https://requests-scitokens.readthedocs.io/en/stable/",
    "uproot": "https://uproot.readthedocs.io/en/stable/",
    "scipy": "https://docs.scipy.org/doc/scipy/reference/",
    "scitokens": "https://scitokens.readthedocs.io/en/stable/",
}.items()}

# -- linkcode

# linkcode
linkcode_url = sphinx_github_style.get_linkcode_url(
    blob=sphinx_github_style.get_linkcode_revision("head"),
    url=PROJECT_URL,
)
linkcode_resolve = sphinx_github_style.get_linkcode_resolve(linkcode_url)

# -- sphinx-gallery

sphinx_gallery_conf = {
    # where to find the examples
    "examples_dirs": [
        str(Path("..") / "examples"),
        str(Path() / "cli" / "_examples"),
    ],
    # where to render them
    "gallery_dirs": [
        str(Path() / "examples"),
        str(Path() / "cli" / "examples"),
    ],
    "download_all_examples": False,
    "filename_pattern": r"/.*\.py",  # execute all examples
    "ignore_pattern": r"test_.*\.py",  # ignore example tests
    "reset_modules": [],  # don't reset matplotlib (or seaborn)
    "write_computation_times": False,
    "within_subsection_order": "ExampleTitleSortKey",
    # allow some examples to fail
    "only_warn_on_example_error": True,
}

# -- build CLI examples

def render_cli_examples(app):
    """Render the CLI examples for sphinx-gallery."""
    clidir = Path(app.confdir) / "cli"
    return cli_gallery.render_entry_point_examples(
        clidir / "examples.ini",
        clidir / "_examples",
        logger=logging.getLogger("gwpy-plot-gallery"),
        filename_prefix="gwpy-plot-",
    )


# -- setup sphinx -----------

def setup(app):
    # render the CLI examples, with low priority number so that this occurs
    # before sphinx-gallery attempts to execute them
    app.connect("builder-inited", render_cli_examples, priority=10)
