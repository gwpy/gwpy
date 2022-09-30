# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2014-2017),
#               Cardiff University (2017-2021)
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

import inspect
import os.path
import re
import shlex
import shutil
import sys
import warnings
from configparser import ConfigParser
from pathlib import Path
from string import Template

import matplotlib

from sphinx.util import logging

from numpydoc import docscrape_sphinx

import gwpy
from gwpy.utils.sphinx import (
    ex2rst,
    zenodo,
)

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
    GWPY_GIT_REF = 'v{}'.format(GWPY_VERSION)

# -- matplotlib -------------

matplotlib.use('agg')

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
project = 'GWpy'
copyright = ' and '.join((
    '2013, 2017-2021 Cardiff University',
    '2013-2017 Lousiana State University',
))
version = "dev" if ".dev" in GWPY_VERSION else GWPY_VERSION
release = GWPY_VERSION

# extension modules
# DEVNOTE: please make sure and add 3rd-party dependencies to
#          pyproject.toml's [project.optional-dependencies]/docs
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.autosummary',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.linkcode',
    'sphinx.ext.ifconfig',
    'sphinx_automodapi.automodapi',
    'sphinx_panels',
    'sphinxcontrib.programoutput',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
    #'sphinxcontrib.doxylink',  # noqa: E265
    'gwpy.utils.sphinx.epydoc',
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
    "repo_type": "github",
    "repo_url": "https://github.com/gwpy/gwpy",
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

autoclass_content = 'class'
autodoc_default_flags = ['show-inheritance', 'members', 'inherited-members']

# -- autosummary

autosummary_generate = True

# -- plot_directive

plot_rcparams = dict(matplotlib.rcParams)
plot_rcparams.update({
    'backend': 'agg',
})
plot_apply_rcparams = True
plot_formats = ['png']
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
parts = re.split(r'[\(\)|]', docscrape_sphinx.IMPORT_MATPLOTLIB_RE)[1:-1]
parts.extend(('fig.show()', 'plot.show()'))
docscrape_sphinx.IMPORT_MATPLOTLIB_RE = r'\b({})\b'.format('|'.join(parts))

# -- inhertiance_diagram

# configure inheritance diagram
inheritance_graph_attrs = dict(rankdir='TB')

# -- epydoc

# epydoc extension config for GLUE
epydoc_mapping = {
    'http://software.ligo.org/docs/glue/': [r'glue(\.|$)'],
}

# -- epydoc

LALSUITE_DOCS = 'http://software.ligo.org/docs/lalsuite'

doxylink = {
    'lal': ('lal.tag', '%s/lal/' % LALSUITE_DOCS),
    'lalframe': ('lalframe.tag', '%s/lalframe/' % LALSUITE_DOCS),
}

# -- intersphinx

# Intersphinx
intersphinx_mapping = {
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'dateutil': ('https://dateutil.readthedocs.io/en/stable/', None),
    'dqsegdb2': ('https://dqsegdb2.readthedocs.io/en/stable/', None),
    # 'glue': ('https://docs.ligo.org/lscsoft/glue/', None),
    'gwdatafind': ('https://gwdatafind.readthedocs.io/en/stable/', None),
    'gwosc': ('https://gwosc.readthedocs.io/en/stable/', None),
    'h5py': ('https://docs.h5py.org/en/latest/', None),
    'ligo.skymap': ('https://lscsoft.docs.ligo.org/ligo.skymap/', None),
    'ligo-segments': ('https://lscsoft.docs.ligo.org/ligo-segments/', None),
    'ligolw': ('https://docs.ligo.org/kipp.cannon/python-ligo-lw/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pycbc': ('https://pycbc.org/pycbc/latest/html/', None),
    'python': ('https://docs.python.org/3/', None),
    'uproot': ('https://uproot.readthedocs.io/en/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
}

# -- sphinx-panels

panels_css_variables = {
    "tabs-color-label-active": "#ff6e40;",
    "tabs-size-label": ".8rem",
}


# -- linkcode

def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object

    This code is stolen with thanks from the scipy team.
    """
    if domain != "py" or not info["module"]:
        return None

    def find_source(module, fullname):
        obj = sys.modules[module]
        for part in fullname.split("."):
            obj = getattr(obj, part)
        try:  # unwrap a decorator
            obj = obj.im_func.func_closure[0].cell_contents
        except (AttributeError, TypeError):
            pass
        # get filename
        filename = Path(inspect.getsourcefile(obj)).relative_to(
            Path(gwpy.__file__).parent,
        ).as_posix()
        # get line numbers of this object
        source, lineno = inspect.getsourcelines(obj)
        if lineno:
            return "{}#L{:d}-L{:d}".format(
                filename,
                lineno,
                lineno + len(source) - 1,
            )
        return filename

    try:
        fileref = find_source(info["module"], info["fullname"])
    except (
        AttributeError,  # object not found
        OSError,  # file not found
        TypeError,  # source for object not found
        ValueError,  # file not from GWpy
    ):
        return None

    return "https://github.com/gwpy/gwpy/tree/{}/gwpy/{}".format(
        GWPY_GIT_REF,
        fileref,
    )


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

.. code:: sh

   $$ ${command}

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
    raw = config.get(section, 'command') + " --interactive"
    title = config.get(
        section,
        'title',
        fallback=' '.join(map(str.title, section.split('-'))),
    )
    desc = config.get(section, 'description', fallback='')

    # build command-line string for display
    cmdstr = f"gwpy-plot {raw}".replace(  # split onto multiple lines
        ' --',
        ' \\\n       --',
    )

    # build code to generate the plot when sphinx runs
    args = ", ".join(map(repr, shlex.split(raw)))
    code = "\n   ".join([
        "from gwpy.cli.gwpy_plot import main as gwpy_plot",
        f"gwpy_plot([{args}])",
    ])

    rst = CLI_TEMPLATE.substitute(
        title=title,
        titleunderline='#' * len(title),
        description=desc,
        tag=section,
        command=cmdstr,
        code=code,
    )

    # only write RST if new or changed
    rstfile = outdir / f"{section}.rst"
    if _new_or_different(rst, rstfile):
        rstfile.write_text(rst)
        logger.info("[cli] wrote {}".format(rstfile))
    return rstfile


def render_cli_examples(_):
    """Render all :mod:`gwpy.cli` examples as RST to be processed by Sphinx.
    """
    logger = logging.getLogger('cli-examples')

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
        examples="\n   ".join((str(rst.with_suffix("")) for rst in rsts)),
    )
    (exdir / "examples.rst").write_text(rst)


# -- examples

def _render_example(example, outdir, logger):
    # render the example
    rst = ex2rst.ex2rst(example)

    # if it has changed, write it (prevents sphinx from
    # unnecessarily reprocessing)
    target = outdir / example.with_suffix(".rst").name
    if _new_or_different(rst, target):
        target.write_text(rst)
        logger.debug('[examples] wrote {0}'.format(target))


def render_examples(_):
    """Render all examples as RST to be processed by Sphinx.
    """
    logger = logging.getLogger("examples")
    logger.info("[examples] converting examples to RST...")

    srcdir = SPHINX_DIR.parent / "examples"
    outdir = SPHINX_DIR / "examples"
    outdir.mkdir(exist_ok=True)

    # find all examples
    for exdir in next(os.walk(srcdir))[1]:
        if exdir in {"__pycache__"}:  # ignore
            continue
        subdir = outdir / exdir
        subdir.mkdir(exist_ok=True)
        # copy index
        index = subdir / "index.rst"
        shutil.copyfile(srcdir / exdir / index.name, index)
        logger.debug('[examples] copied {0}'.format(index))
        # render python script as RST
        for expy in (srcdir / exdir).glob("*.py"):
            _render_example(expy, subdir, logger)
        logger.info('[examples] converted all in examples/{0}'.format(exdir))


# -- create citation file

def write_citing_rst(app):
    """Render the ``citing.rst`` file using the Zenodo API
    """
    logger = logging.getLogger('zenodo')
    citing = SPHINX_DIR / "citing.rst"
    content = citing.with_suffix(".rst.in").read_text()
    content += '\n' + zenodo.format_citations(597016)
    citing.write_text(content)
    logger.info('[zenodo] wrote {0}'.format(citing))


# -- setup sphinx -----------

def setup(app):
    app.connect('builder-inited', write_citing_rst)
    app.connect('builder-inited', render_examples)
    app.connect('builder-inited', render_cli_examples)
