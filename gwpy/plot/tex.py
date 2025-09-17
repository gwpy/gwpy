# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Handle TeX formatting for matplotlib output."""

import functools
import re
from shutil import which

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# -- tex configuration ---------------

MACROS = [
    r"\def\rtHz{\ensuremath{\sqrt{\mathrm{Hz}}}}",  # \sqrt{Hz} label
]


def _test_usetex() -> None:
    """Draw (but don't show) a test image using matplotlib and LaTeX."""
    from matplotlib import (
        pyplot,
        rc_context,
    )
    with rc_context({"text.usetex": True}):
        fig = pyplot.figure()
        try:
            ax = fig.gca()
            ax.set_xlabel(r"\LaTeX")
            fig.canvas.draw()
        finally:
            pyplot.close(fig)


@functools.cache
def has_tex() -> bool:
    """Return `True` if LaTeX is usable on this system.

    Checks for ``latex``, ``pdflatex``, and ``dvipng`` on the path, and
    then attemps to draw an image using LaTeX syntax.

    Returns
    -------
    hastex : `bool`
        `True` if the test image is drawn correctly, otherwise `False`.
    """
    # run basic sanity checks
    for exe in ("latex", "pdflatex", "dvipng"):
        if which(exe) is None:
            return False

    # attempt to render an image with latex
    try:
        _test_usetex()
    except Exception:  # noqa: BLE001
        # failed for any reason
        return False

    return True


# -- tex formatting ------------------

LATEX_CONTROL_CHARS = [
    "%",
    "\\",
    "_",
    "~",
    "&",
    "#",
]
re_latex_control = re.compile(
    rf"(?<!\\)[{''.join(LATEX_CONTROL_CHARS)}](?!.*{{)",
)


def float_to_latex(
    x: float,
    format: str = "%.2g",  # noqa: A002
) -> str:
    r"""Convert a floating point number to a latex representation.

    In particular, scientific notation is handled gracefully: e -> 10^

    Parameters
    ----------
    x : `float`
        the number to represent

    format : `str`, optional
        the output string format

    Returns
    -------
    tex : `str`
        a TeX representation of the input

    Examples
    --------
    >>> from gwpy.plot.tex import float_to_latex
    >>> float_to_latex(1)
    '1'
    >>> float_to_latex(2000)
    '2\times 10^{3}'
    >>> float_to_latex(100)
    '10^{2}'
    >>> float_to_latex(-500)
    r'-5\!\!\times\!\!10^{2}'
    """
    if x == 0.:
        return "0"

    base_str = format % x
    if "e" not in base_str:
        return base_str

    mantissa, exponent = base_str.split("e")

    exponent = exponent.lstrip("0+")
    if exponent.startswith("-0"):
        exponent = "-" + exponent[2:]

    if (fmantissa := float(mantissa)) == 1.0:
        return fr"10^{{{exponent}}}"
    if fmantissa.is_integer():
        mantissa = str(int(fmantissa))

    return fr"{mantissa}\!\!\times\!\!10^{{{exponent}}}"


def label_to_latex(text: str) -> str:
    r"""Convert text into a latex-passable representation.

    This method just escapes the following reserved LaTeX characters:
    % \ _ ~ &, whilst trying to avoid doubly-escaping already escaped
    characters

    Parameters
    ----------
    text : `str`
        input text to convert

    Returns
    -------
    tex : `str`
        a modified version of the input text with all unescaped reserved
        latex characters escaped

    Examples
    --------
    >>> from gwpy.plot.tex import label_to_latex
    >>> label_to_latex('normal text')
    'normal text'
    >>> label_to_latex('$1 + 2 = 3$')
    '$1 + 2 = 3$'
    >>> label_to_latex('H1:ABC-DEF_GHI')
    'H1:ABC-DEF\\_GHI'
    >>> label_to_latex('H1:ABC-DEF\_GHI')
    'H1:ABC-DEF\\_GHI'
    """
    if text is None:
        return ""

    out: list[str] = []

    # loop over matches in reverse order and replace
    x = None
    for m in re_latex_control.finditer(text):
        a, b = m.span()
        char = m.group()[0]
        out.extend((
            text[x:a],
            fr"\{char}",
        ))
        x = b

    if not x:  # no match
        return text

    # append prefix and return joined components
    out.append(text[b:])
    return "".join(out)
