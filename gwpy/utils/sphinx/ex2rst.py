# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-)
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

"""Convert GWpy example python files into rst files for sphinx documentation."""

from __future__ import annotations

import argparse
from pathlib import Path

__author__ = "Duncan Macleod <duncan.macleod@ligo.org"

METADATA: dict[str, str | None] = {
    # replacement for __metadata__ variables, use None to ignore
    "author": "sectionauthor",
    "credits": None,
}


def postprocess_code(
    code: list[str],
    context: str,
):
    if any("plot.show()" in line for line in code):
        ctx = "close-figs"
    else:
        code.insert(2, "   :nofigs:")
        ctx = ""

    code.insert(2, f"   :context: {context}".rstrip())
    code.append("")
    return code, ctx


def ex2rst(infile: str | Path) -> str:
    """Convert a Python example script into RST.

    Returns
    -------
    rst : `str`
        The fully rendered RST text block.
    """
    infile = Path(infile)
    lines = infile.read_text().splitlines()
    ref = "-".join((
        infile.parent.name,
        infile.with_suffix("").name,
    ))

    output: list[str] = []
    header = [f".. _gwpy-example-{ref}:\n"]

    indoc = False
    incode = False
    code: list[str] = []
    context = "reset"

    for i, line in enumerate(lines):
        # skip file header
        if len(output) == 0 and line.startswith("#"):
            continue

        # hide lines
        if line.endswith("# hide"):
            continue

        # find block docs
        if '"""' in line:
            indoc = not indoc
            line = line.strip('"')
            if not line:  # start/end of block quote
                continue

        # skip empty lines not in a block quote
        if not line and not indoc:
            if output:
                output.append("")
            continue

        # finish code block
        if incode and line.startswith(('"', "#", "__")):
            incode = False
            code, context = postprocess_code(code, context)
            output.extend(code)

        # comments
        if line.startswith("#"):
            output.append(line[2:])
        # metadata
        elif line.startswith("__"):
            key, value = map(
                lambda x: x.strip(" _=\"'").rstrip(" _=\"'"),
                line.split("=", 1),
            )
            try:
                metakey = METADATA[key]
            except KeyError:
                header.append(f".. {key}:: {value}\n")
            else:
                if metakey is not None:
                    header.append(f".. {METADATA[key]}:: {value}\n")
        # block quote
        elif indoc:
            output.append(line.strip('"').rstrip('"'))
        # code
        else:
            if not incode:  # restart code block
                code = [
                    "",
                    ".. plot::",
                    "   :include-source:",
                    "",
                ]
            code.append(f"   {line}")
            incode = True

        # end block quote
        if line == '"""' and indoc:
            indoc = False
        elif line.endswith('"""') and indoc:
            output.append("")
            indoc = False

        if len(output) == 1:
            output.append("#" * len(output[0]))

    if incode:
        output.extend(postprocess_code(code, context)[0])

    output = header + output
    return "\n".join(output).replace("\n\n\n", "\n\n")


def create_parser() -> argparse.ArgumentParser:
    """Create an `~argparse.ArgumentParser` for this tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "infile",
        type=Path,
        metavar="example.py",
        help="python file to convert",
    )
    parser.add_argument(
        "outfile",
        type=Path,
        metavar="example.rst",
        nargs="?",
        help="rst file to write, default: print to screen",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    """Run this tool."""
    # parse command line
    parser = create_parser()
    opts = parser.parse_args(args)

    # convert python to RST
    rst = ex2rst(opts.infile)

    # write output
    if opts.outfile:
        with open(opts.outfile, "w") as f:
            print(rst, file=f)
    else:
        print(rst)


if __name__ == "__main__":
    main()
