# Copyright (c) 2025 Cardiff University
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

"""Render command-line examples for `sphinx-gallery`.

The command-line examples are specified in a INI-format configuration file.
Each section has the following options:

"title"
    The name of the example

"description"
    The summary of what is being shown

"command"
    The argv options to pass to the function

"entry_point"
    The entry-point to call with ``command``.
    Default is `gwpy-plot`.

See `/docs/cli/examples.ini` in the GWpy project for an example.
"""

from __future__ import annotations

import importlib.metadata
import logging
import shlex
from configparser import ConfigParser
from string import Template
from textwrap import indent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path

    from sphinx.util.logging import SphinxLoggerAdapter

logger = logging.getLogger(__name__)

CLI_TEMPLATE = Template("""
\"\"\"
${titleunderline}
${title}
${titleunderline}

${description}

.. code-block:: ${language}
    ${caption}
    ${command}
\"\"\"

# %%
# The same command can be executed directly in Python as:

${pyimport}
${pyfunction}([
    ${args},
])
""".strip())


def _render_entry_point_example(
    config: ConfigParser,
    section: str,
    outdir: Path,
    logger: Logger | SphinxLoggerAdapter = logger,
    entry_point: str = "gwpy-plot",
    filename_prefix: str = "cli_",
) -> Path:
    """Render an entry point example as RST to be processed by Sphinx."""
    # read config values (allow for multi-line definition)
    argv = config.get(section, "argv").strip().replace("\n", " ")
    title = config.get(
        section,
        "title",
        fallback=" ".join(map(str.title, section.split("-"))),
    )
    desc = config.get(section, "description", fallback="")
    caption = config.get(section, "caption", fallback="")
    if caption:  # sphinx doesn't support an empty caption
        caption = f":caption: {caption}\n"
    language = config.get(section, "language", fallback="shell")
    ep_name = config.get(section, "entry_point", fallback=entry_point)

    # format the python function
    try:
        entrypoint, = importlib.metadata.entry_points(name=ep_name)
    except ValueError as exc:
        msg = "ambiguous entry point"
        raise ValueError(msg) from exc
    pymod, pyfunc = entrypoint.value.rsplit(":", 1)
    pyimport = f"from {pymod} import {pyfunc}"

    # split argv onto multiple lines
    indentstr = " " * 4
    argstr = argv.replace(" -", f" \\\\\n{indentstr}-")
    # build command-line string for display
    cmdstr = indent(f"{entrypoint.name} {argstr}", indentstr).strip()

    # build code to generate the plot when sphinx runs
    pyargs = indent(
        # join the arguments into a list, but drop a
        # new line for each new option flag, and indent
        ", ".join(map(repr, shlex.split(argv))).replace(
            ", '-",
            ",\n'-",
        ),
        indentstr,
    ).strip()

    # create Python script
    pyfile = outdir / f"{filename_prefix}{section}.py"
    code = CLI_TEMPLATE.substitute(
        title=title,
        titleunderline="#" * len(title),
        description=desc,
        language=language,
        caption=caption,
        command=cmdstr,
        args=pyargs,
        pyimport=pyimport,
        pyfunction=pyfunc,
    )
    pyfile.write_text(code)
    logger.info("[cli] wrote %s", str(pyfile))
    return pyfile


def render_entry_point_examples(
    inifile: Path,
    outdir: Path,
    logger: Logger | SphinxLoggerAdapter = logger,
    entry_point: str = "gwpy-plot",
    filename_prefix: str = "cli_",
) -> None:
    """Render entry-point examples as RST to be processed by Sphinx."""
    outdir.mkdir(exist_ok=True, parents=True)

    # read example config
    config = ConfigParser()
    config.read(inifile)

    logger.info("rendering entry point examples")

    # render examples
    for sect in config.sections():
        _render_entry_point_example(
            config,
            sect,
            outdir,
            logger,
            entry_point=entry_point,
            filename_prefix=filename_prefix,
        )

    # write a gallery header (blank)
    (outdir / "GALLERY_HEADER.rst").write_text("""
Click on a thumbnail image below to see the command used to generate
the example, and the full-resolution output image.
""".strip())
