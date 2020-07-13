# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Custom default figure configuration
"""

from matplotlib import (rcParams, rc_params, RcParams)

from . import tex
from ..utils.env import bool_env

# record matplotlib's original rcParams
MPL_RCPARAMS = rc_params()

# record the LaTeX preamble
try:
    PREAMBLE = rcParams.get('text.latex.preamble', []) + tex.MACROS
except TypeError:  # matplotlib < 3.1.0
    PREAMBLE = rcParams.get('text.latex.preamble', '') + '\n'.join(tex.MACROS)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# -- custom rc ----------------------------------------------------------------

# set default params
GWPY_RCPARAMS = RcParams(**{
    # axes boundary colours
    'axes.edgecolor': 'gray',
    # grid
    'axes.grid': True,
    'axes.axisbelow': False,
    'grid.linewidth': .5,
    # ticks
    'axes.formatter.limits': (-3, 4),
    'axes.formatter.use_mathtext': True,
    'axes.labelpad': 5,
    # fonts
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',
    'font.family': ['sans-serif'],
    'font.sans-serif': [
        'FreeSans',
        'Helvetica Neue',
        'Helvetica',
        'Arial',
    ] + rcParams['font.sans-serif'],
    'font.size': 12,
    # legend (revert to mpl 1.5 formatting in parts)
    'legend.edgecolor': 'inherit',
    'legend.numpoints': 2,
    'legend.handlelength': 1,
    'legend.fancybox': False,
})

# set latex options
GWPY_TEX_RCPARAMS = RcParams(**{
    # use latex styling
    'text.usetex': True,
    'text.latex.preamble': PREAMBLE,
    # use bigger font for labels (since the font is good)
    'font.family': ['serif'],
    'font.size': 16,
    # don't use mathtext for offset
    'axes.formatter.use_mathtext': False,
})


def rc_params(usetex=None):
    """Returns a new `matplotlib.RcParams` with updated GWpy parameters

    The updated parameters are globally stored as
    `gwpy.plot.rc.GWPY_RCPARAMS`, with the updated TeX parameters as
    `gwpy.plot.rc.GWPY_TEX_RCPARAMS`.

    .. note::

       This function doesn't apply the new `RcParams` in any way, just
       creates something that can be used to set `matplotlib.rcParams`.

    Parameters
    ----------
    usetex : `bool`, `None`
        value to set for `text.usetex`; if `None` determine automatically
        using the ``GWPY_USETEX`` environment variable, and whether `tex`
        is available on the system. If `True` is given (or determined)
        a number of other parameters are updated to improve TeX formatting.

    Examples
    --------
    >>> import matplotlib
    >>> from gwpy.plot.rc import rc_params as gwpy_rc_params()
    >>> matplotlib.rcParams.update(gwpy_rc_params(usetex=False))
    """
    # if user didn't specify to use tex or not, guess based on
    # the `GWPY_USETEX` environment variable, or whether tex is
    # installed at all.
    if usetex is None:
        usetex = bool_env(
            'GWPY_USETEX',
            default=rcParams['text.usetex'] or tex.has_tex())

    # build RcParams from matplotlib.rcParams with GWpy extras
    rcp = GWPY_RCPARAMS.copy()
    if usetex:
        rcp.update(GWPY_TEX_RCPARAMS)
    return rcp


# -- dynamic subplot positioning ----------------------------------------------

SUBPLOT_WIDTH = {
    6.4: (.1875, .87),
    8.: (.15, .85),
    12.: (.1, .90),
}
SUBPLOT_HEIGHT = {
    3.: (.25, .83),
    4.: (.2, .85),
    4.8: (.16, .88),
    5: (.15, .89),
    6.: (.13, .9),
    8.: (.1, .93),
}


def get_subplot_params(figsize):
    """Return sensible default `SubplotParams` for a figure of the given size

    Parameters
    ----------
    figsize : `tuple` of `float`
         the ``(width, height)`` figure size (inches)

    Returns
    -------
    params : `~matplotlib.figure.SubplotParams`
        formatted set of subplot parameters
    """
    from matplotlib.figure import SubplotParams

    width, height, = figsize
    try:
        left, right = SUBPLOT_WIDTH[width]
    except KeyError:
        left = right = None
    try:
        bottom, top = SUBPLOT_HEIGHT[height]
    except KeyError:
        bottom = top = None
    return SubplotParams(left=left, bottom=bottom, right=right, top=top)
