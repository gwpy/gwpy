# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Handle TeX formatting for matplotlib output
"""

from __future__ import division

import re

from ..utils.shell import which

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# -- tex configuration --------------------------------------------------------

MACROS = [
    r'\def\rtHz{\ensuremath{\sqrt{\mathrm{Hz}}}}',  # \sqrt{Hz} label
]


def has_tex():
    """Returns whether tex is installed on this system

    Returns
    -------
    True
        if ``pdflatex`` and ``dvipng`` executables are found on the path
    False
        otherwise
    """
    try:
        which('pdflatex')
    except ValueError:
        return False
    try:
        which('dvipng')
    except ValueError:
        return False
    return True


HAS_TEX = has_tex()

# -- tex formatting -----------------------------------------------------------

LATEX_CONTROL_CHARS = ["%", "\\", "_", "~", "&"]
re_latex_control = re.compile(r'(?<!\\)[%s](?!.*{)'
                              % ''.join(LATEX_CONTROL_CHARS))


def float_to_latex(x, format="%.2g"):  # pylint: disable=redefined-builtin
    # pylint: disable=anomalous-backslash-in-string
    """Convert a floating point number to a latex representation.

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
    >>> from gwpy.plotter.tex import float_to_latex
    >>> float_to_latex(1)
    '1'
    >>> float_to_latex(2000)
    '2\times 10^{3}'
    >>> float_to_latex(100)
    '10^{2}'
    >>> float_to_latex(-500)
    r'-5\!\!\times\!\!10^{2}'
    """
    base_str = format % x
    if "e" not in base_str:
        return base_str
    mantissa, exponent = base_str.split("e")
    if float(mantissa).is_integer():
        mantissa = int(float(mantissa))
    exponent = exponent.lstrip("0+")
    if exponent.startswith('-0'):
        exponent = '-' + exponent[2:]
    if float(mantissa) == 1.0:
        return r"10^{%s}" % exponent
    return r"%s\!\!\times\!\!10^{%s}" % (mantissa, exponent)


def label_to_latex(text):
    # pylint: disable=anomalous-backslash-in-string
    """Convert text into a latex-passable representation.

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
    >>> from gwpy.plotter.tex import label_to_latex
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
        return ''
    out = []
    x = None
    # loop over matches in reverse order and replace
    for m in re_latex_control.finditer(text):
        a, b = m.span()
        char = m.group()[0]
        out.append(text[x:a])
        out.append(r'\%s' % char)
        x = b
    if not x:  # no match
        return text
    # append prefix and return joined components
    out.append(text[b:])
    return ''.join(out)


def unit_to_latex(unit):
    # pylint: disable=anomalous-backslash-in-string
    """Convert a `~astropy.units.Unit` to a latex string

    Parameters
    ----------
    unit : `~astropy.units.Unit`
        input unit to represent

    Returns
    -------
    tex : `str`
        a tex-formatted version of the unit

    Examples
    --------
    >>> from gwpy.plotter.tex import unit_to_latex
    >>> unit_to_latex(units.Hertz)
    '$\mathrm{Hz}$'
    >>> unit_to_latex(units.Volt.decompose())
    '$\mathrm{m^{2}\,kg\,A^{-1}\,s^{-3}}$'
    """
    from astropy import units
    from astropy.units.format import utils as unit_utils

    if unit is None:
        return ''
    if isinstance(unit, units.NamedUnit):
        ustr = label_to_latex(unit.name)
    elif isinstance(unit, units.CompositeUnit):
        if unit.scale != 1:
            ustr = float_to_latex(unit.scale) + r'\ '
        else:
            ustr = ''
        if unit.bases:
            positives, negatives = unit_utils.get_grouped_by_powers(
                unit.bases, unit.powers)
            if negatives == 1:
                negatives = _format_unit_list(negatives)
                positives = _format_unit_list(positives) if positives else 1
                ustr += r'{0}/{1}'.format(positives, negatives)
            elif negatives:
                positives = _format_unit_list(positives) if positives else ''
                negatives = _format_unit_list(negatives, negative=True)
                ustr += r'{0}\,{1}'.format(positives, negatives)
            else:
                positives = _format_unit_list(positives)
                ustr += positives
    elif isinstance(unit, units.UnitBase):
        return unit.to_string('latex_inline')
    else:
        ustr = str(unit)
    if ustr:
        return r'$\mathrm{{{0}}}$'.format(ustr)
    return ''


def _format_unit_list(unitlist, negative=False):
    from astropy.units.format import latex

    out = []
    texformatter = latex.Latex()
    for base, power in unitlist:
        if power == 1 and not negative:
            out.append(texformatter._get_unit_name(base))
        elif power == 0.5 and not negative:
            out.append('\sqrt{{{0}}}'.format(label_to_latex(base.name)))
        elif power != 1 and 1/power == int(1/power):
            out.append('{0}^{{1/{1}}}'.format(
                label_to_latex(base.name), int(1/power)))
        elif negative:
            out.append('{0}^{{-{1}}}'.format(
                label_to_latex(base.name), power))
        else:
            out.append('{0}^{{{1}}}'.format(
                label_to_latex(base.name), power))
    return r'\,'.join(out)
