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

import os

from astropy import units
from astropy.units.format import (latex as ulatex, utils as uutils)

from .. import version

__author__ = "Duncan M. Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

USE_TEX = os.system('which pdflatex > %s 2>&1' % os.devnull) == 0

LATEX_CONTROL_CHARS = ["%", "\\", "_", "~"]


def float_to_latex(x, format="%.2g"):
    """Convert a floating point number to a latex representation. 

    In particular, scientific notation is handled gracefully: e -> 10^

    Example:
    @code
    >>> float_to_latex(2000)
    '2\times 10^{3}'
    @endcode

    @returns a string in latex mathmode
    """
    base_str = format % x
    if "e" not in base_str:
        return base_str
    mantissa, exponent = base_str.split("e")
    exponent = exponent.lstrip("0+")
    if exponent.startswith('-0'):
        exponent = '-' + exponent[2:]
    if float(mantissa) == 1.0:
        return r"10^{%s}" % exponent
    else:
        return r"%s\!\!\times\!\!10^{%s}" % (mantissa, exponent)


def label_to_latex(text):
    """Convert an abitrary string of text into a latex-passable
    representation.
    """
    if text is None:
        return ''
    for ch in LATEX_CONTROL_CHARS:
        text = text.replace(ch, "\\%s" % ch)
    return text


def unit_to_latex(unit):
    if isinstance(unit, units.NamedUnit):
        s = label_to_latex(unit.name)
    elif isinstance(unit, units.CompositeUnit):
        if unit.scale != 1:
            s = float_to_latex(unit.scale) + r'\ '
        else:
            s = ''
        if len(unit.bases):
            positives, negatives = uutils.get_grouped_by_powers(
                unit.bases, unit.powers)
            if len(negatives) == 1:
                negatives = format_unit_list(negatives)
                positives = positives and format_unit_list(positives) or 1
                s += r'{0}/{1}'.format(positives, negatives)
            elif len(negatives):
                if len(positives):
                    positives = format_unit_list(positives)
                else:
                    positives = ''
                negatives = format_unit_list(negatives, negative=True)
                s += r'{0}\,{1}'.format(positives, negatives)
            else:
                positives = format_unit_list(positives)
                s += positives
    return r'$\mathrm{{{0}}}$'.format(s)


def format_unit_list(unitlist, negative=False):
    out = []
    texformatter = ulatex.Latex()
    for base, power in unitlist:
        if power == 1 and not negative:
            out.append(texformatter._get_unit_name(base))
        elif power == 0.5 and not negative:
            out.append('\sqrt{{{0}}}'.format(label_to_latex(base.name)))
        elif 1/power == int(1/power):
            out.append('{0}^{{1/{1}}}'.format(
                label_to_latex(base.name), int(1/power)))
        elif negative:
            out.append('{0}^{-{{1}}}'.format(
                label_to_latex(base.name), power))
        else:
            out.append('{0}^{{{1}}}'.format(
                label_to_latex(base.name), power))
    return r'\,'.join(out)
