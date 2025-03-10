# Copyright (C) Louisiana State University (2014-2017),
#               Cardiff University (2017-2025)
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
import shlex
import warnings
from configparser import ConfigParser
from datetime import date
from pathlib import Path
from string import Template

import matplotlib
import sphinx_github_style
from numpydoc import docscrape_sphinx
from sphinx.util import logging

import gwpy
from gwpy.utils.sphinx import (
    zenodo,
)

TODAY = date.today()

SPHINX_DIR = Path(__file__).parent.absolute()
STATIC_DIRNAME = "_static"

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
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.programoutput",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
    #'sphinxcontrib.doxylink',  # noqa: E265
]

# content management
default_role = "obj"
exclude_patterns = [
    "references.rst",
    "_build",
    "_generated",
    "cli/examples/examples.rst",
]
templates_path = [
    "_templates",
]

# epilog
rst_epilog = "\n.. include:: /references.rst"

# -- HTML formatting --------

extensions.append("sphinx_immaterial")
html_theme = "sphinx_immaterial"
html_theme_options = {
    # metadata
    "repo_name": "GWpy",
    "repo_type": "gitlab",
    "repo_url": "https://gitlab.com/gwpy/gwpy",
    "edit_uri": "blob/main/docs",
    "globaltoc_collapse": True,
    # features
    "features": [
        "navigation.sections",
    ],
    # colouring and light/dark mode
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "blue-grey",
            "accent": "deep-orange",
            "toggle": {
                "icon": "material/eye-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "amber",
            "accent": "deep-orange",
            "toggle": {
                "icon": "material/eye",
                "name": "Switch to light mode",
            },
        },
    ],
    # table of contents
    "toc_title_is_page_title": True,
    # version dropdown
    "version_dropdown": True,
    "version_json": "../versions.json",
}
html_static_path = [STATIC_DIRNAME]
html_favicon = str(Path(STATIC_DIRNAME) / "favicon.png")
html_logo = str(Path(STATIC_DIRNAME) / "favicon.png")
html_css_files = ["css/gwpy-sphinx.css"]

# -- extensions config ------

# -- autodoc

autoclass_content = "class"
autodoc_default_flags = ["show-inheritance", "members", "inherited-members"]

# -- autosummary

autosummary_generate = True

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
inheritance_graph_attrs = dict(rankdir="TB")

# -- doxylink

LALSUITE_DOCS = "https://lscsoft.docs.ligo.org/lalsuite"

doxylink = {
    "lal": ("lal.tag", f"{LALSUITE_DOCS}/lal/"),
    "lalframe": ("lalframe.tag", f"{LALSUITE_DOCS}/lalframe/"),
}

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
    "ligo.skymap": "https://lscsoft.docs.ligo.org/ligo.skymap/",
    "igwn-auth-utils": "https://igwn-auth-utils.readthedocs.io/en/stable/",
    "igwn-segments": "https://igwn-segments.readthedocs.io/en/stable/",
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
    url="https://gitlab.com/gwpy/gwpy",
)
linkcode_resolve = sphinx_github_style.get_linkcode_resolve(linkcode_url)

# -- sphinx-gallery

sphinx_gallery_conf = {
    # where to find the examples
    "examples_dirs": [
        str(SPHINX_DIR.parent / "examples"),
    ],
    # where to render them
    "gallery_dirs": [
        str(SPHINX_DIR / "examples"),
    ],
    "filename_pattern": r"/.*\.py",  # execute all examples
    "ignore_pattern": r"test_.*\.py",  # ignore example tests
    "write_computation_times": False,
}

# -- plugins ----------------

# -- build CLI examples

CLI_INDEX_TEMPLATE = Template("""
.. toctree::
   :numbered:

   ${examples}
""".strip())

CLI_TEMPLATE = Template("""
.. _gwpy-cli-example-${tag}:

${titleunderline}
${title}
${titleunderline}

${description}

.. code-block:: shell

   ${command}

.. plot::
   :align: center
   :alt: ${title}
   :context: reset
   :format: python
   :include-source: false

   ${code}
""".strip())


def _new_or_different(content, target):
    """Return `True` if a target file doesn't exist, or doesn't have the
    specified content
    """
    try:
        return Path(target).read_text() != content
    except FileNotFoundError:
        return True


def _render_cli_example(config, section, outdir, logger):
    """Render a :mod:`gwpy.cli` example as RST to be processed by Sphinx.
    """
    # read config values (allow for multi-line definition)
    raw = config.get(
        section,
        "command",
    ).strip().replace("\n", " ") + " --interactive"
    title = config.get(
        section,
        "title",
        fallback=" ".join(map(str.title, section.split("-"))),
    )
    desc = config.get(section, "description", fallback="")

    # build command-line string for display
    cmdstr = f"gwpy-plot {raw}".replace(  # split onto multiple lines
        " --",
        " \\\n       --",
    )

    # build code to generate the plot when sphinx runs
    args = ", ".join(map(repr, shlex.split(raw)))
    code = "\n   ".join([
        "from gwpy.cli.gwpy_plot import main as gwpy_plot",
        f"gwpy_plot([{args}])",
    ])

    rst = CLI_TEMPLATE.substitute(
        title=title,
        titleunderline="#" * len(title),
        description=desc,
        tag=section,
        command=cmdstr,
        code=code,
    )

    # only write RST if new or changed
    rstfile = outdir / f"{section}.rst"
    if _new_or_different(rst, rstfile):
        rstfile.write_text(rst)
        logger.info(f"[cli] wrote {rstfile}")
    return rstfile


def render_cli_examples(_):
    """Render all :mod:`gwpy.cli` examples as RST to be processed by Sphinx.
    """
    logger = logging.getLogger("cli-examples")

    # directories
    clidir = SPHINX_DIR / "cli"
    exini = clidir / "examples.ini"
    exdir = clidir / "examples"
    exdir.mkdir(exist_ok=True, parents=True)

    # read example config
    config = ConfigParser()
    config.read(exini)

    # render examples
    rsts = []
    for sect in config.sections():
        rst = _render_cli_example(config, sect, exdir, logger)
        # record the path relative to the /cli/ directory
        # because that's where the toctree is included
        rsts.append(rst.relative_to(clidir))

    rst = CLI_INDEX_TEMPLATE.substitute(
        examples="\n   ".join(str(rst.with_suffix("")) for rst in rsts),
    )
    (exdir / "examples.rst").write_text(rst)


# -- create citation file

def write_citing_rst(app):
    """Render the ``citing.rst`` file using the Zenodo API
    """
    logger = logging.getLogger("zenodo")
    citing = SPHINX_DIR / "citing.rst"
    content = citing.with_suffix(".rst.in").read_text()
    content += "\n" + zenodo.format_citations(597016)
    citing.write_text(content)
    logger.info(f"[zenodo] wrote {citing}")


# -- setup sphinx -----------

def setup(app):
    app.connect("builder-inited", write_citing_rst)
    app.connect("builder-inited", render_cli_examples)
