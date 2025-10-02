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

"""Helper functions for plotting data with matplotlib and LAL."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from matplotlib import rcParams

if TYPE_CHECKING:
    from collections.abs import (
        Iterable,
        Iterator,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# groups of input parameters (for passing to Plot())
FIGURE_PARAMS: list[str] = [
    "dpi",
    "figsize",
]
AXES_PARAMS: list[str] = [
    # standard options
    "projection",
    "title",
    # X-axis params
    "sharex",
    "xlabel",
    "xlim",
    "xscale",
    # Y-axis params
    "sharey",
    "ylabel",
    "ylim",
    "yscale",
    # special GWpy extras
    "epoch",
    "insetlabels",
]


def color_cycle(colors: Iterable[str] | None = None) -> Iterator[str]:
    """Return an infinite iterator of the given (or default) colors.

    Parameters
    ----------
    colors : `list` of `str`, optional
        The colours to iterate.
        Default is `None` to use Matplotlib's default set of colours.

    Returns
    -------
    colors : iterator of `str`
        The new iterator that yields colour strings infinitely.
    """
    if colors:
        return itertools.cycle(colors)
    return itertools.cycle(p["color"] for p in rcParams["axes.prop_cycle"])


def marker_cycle(markers: Iterable[str] | None = None) -> Iterator[str]:
    """Return an infinite iterator of the given (or default) markers.

    Parameters
    ----------
    markers : `list` of `str`, optional
        The markers to iterate.
        Default is `None` to use a standard set of markers.

    Returns
    -------
    markers : iterator of `str`
        The new iterator that yields marker strings infinitely.
    """
    if markers:
        return itertools.cycle(markers)
    return itertools.cycle(("o", "x", "+", "^", "D", "H", "1"))
