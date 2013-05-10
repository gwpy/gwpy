#!/usr/bin/env python

# Copyright (C) 2012 Duncan M. Macleod
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Helper functions for plotting data with matplotlib and LAL
"""

import numpy

from lal import git_version
from lal.gpstime import gps_to_str

__author__ = "Duncan M. Macleod <duncan.macleod@ligo.org>"
__version__ = git_version.id
__date__ = git_version.date


def float_to_latex(x, format="%.2g"):
    """Convert a floating point number into a TeX representation.

    In particular, scientific notation is handled gracefully: e -> 10^

    @code
    >>> float_to_latex(10)
    '$10$'
    >>> float_to_latex(1000)
    r'$10^{3}$'
    >>> float_to_latex(123456789)
    r'$1.2\times 10^{8}$'

    @returns a TeX format string (with math-mode dollars ($))
    """
    base_str = format % x
    if "e" not in base_str:
        return "$%s$" % base_str
    mantissa, exponent = base_str.split("e")
    exponent = exponent.lstrip("0+")
    if mantissa == "1":
        return r"$10^{%s}$" % exponent
    else:
        return r"$%s\times 10^{%s}$" % (mantissa, exponent)


def str_to_latex(input):
    """Format a string for display in TeX

    @param text
       any string of text to be converted into TeX format OR
       a LIGOTimeGPS to be converted to UTC for display

    @code
    >>> latex("bank_chisq_dof")
    'Bank $\\chi^2$ degrees of freedom'
    >>> latex(LIGOTimeGPS(1000000000))
    'September 14 2011, 01:46:25 UTC'
    @endcode

    @returns the input text formatted for latex
    """

    # if given a GPS time, convert it to UTC
    if isinstance(text, Real) or isinstance(text, LIGOTimeGPS):
       return gpstime.gps_to_str(text)
    # otherwise parse word by word into latex format


def log_transform(lin_range):
    """Return the logarithmic ticks and labels corresponding to the
    input lin_range.
    """
    log_range = numpy.log10(lin_range)
    slope = (lin_range[1] - lin_range[0]) / (log_range[1] - log_range[0])
    inter = lin_range[0] - slope * log_range[0]
    tick_range = [tick for tick in range(int(log_range[0] - 1.0),\
                                         int(log_range[1] + 2.0))\
                  if tick >= log_range[0] and tick<=log_range[1]]
    ticks = [inter + slope * tick for tick in tick_range]
    labels = ["${10^{%d}}$" % tick for tick in tick_range]
    minorticks = []
    for i in range(len(ticks[:-1])):
        minorticks.extend(numpy.logspace(numpy.log10(ticks[i]),\
                                         numpy.log10(ticks[i+1]), num=10)[1:-1])
    return ticks, labels, minorticks
