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

"""Docstring
"""

import numpy

from matplotlib import (colors, colorbar, pyplot as mpl, ticker)
from mpl_toolkits.axes_grid import make_axes_locatable

from .decorators import auto_refresh
from .tex import float_to_latex

from lalapps import git_version as version
__author__ = "Duncan M. Macleod <duncan.macleod@ligo.org>"
__version__ = version.id
__date__ = version.date


class Colorbar(object):
    """An extended version of the default `matplotlib` Colorbar
    """
    def __init__(self, parent, layer):
        """Create a new `Colorbar`
        """
        self._figure = parent._figure
        self._cax = None
        self._parent = parent
        self._mappable = self._parent._layers[layer]
        self._auto_refresh = auto_refresh
        self._log = False

    def show(self, location='right', width=0.2, pad=0.1, log=False,
             label="", clim=None, visible=True, **kwargs):
        divider = make_axes_locatable(self._parent._ax)
        if location == 'right':
            self._cax = divider.new_horizontal(size=width, pad=pad)
        elif location == 'top':
            self._cax = divider.new_vertical(size=width, pad=pad)
        else:
            raise ValueError("'left' and 'bottom' colorbars have not "
                             "been implemented")
        if visible:
            divider._fig.add_axes(self._cax)
        else:
            return

        if not clim:
            cmin = self._mappable.get_array().min()
            cmax = self._mappable.get_array().max()
            clim = [cmin, cmax]

        if mpl.rcParams["text.usetex"]:
            kwargs.setdefault('format',
                              ticker.FuncFormatter(lambda x,pos:
                                                   '$%s$' % float_to_latex(x)))
        kwargs.setdefault("norm", self._mappable.norm)
        self._colorbar = self._figure.colorbar(self._mappable, cax=self._cax,
                                               **kwargs)
        self._mappable.set_colorbar(self._colorbar, self._cax)
        self._mappable.set_clim(clim)
        if log:
            self.set_scale('log')
        if label:
            self._colorbar.set_label(label)
        self._colorbar.draw_all()

    @auto_refresh
    def set_scale(self, scale):
        if scale == "linear" and self._log:
            self._mappable.set_norm(None)
            clim = self._colorbar.get_clim()
            nticks = len(self._colorbar.get_ticks())
            self._colorbar.set_ticks(numpy.linspace(clim[0], clim=[1], 
                                                    num=nticks, endpoint=True))
        elif scale == "log" and not self._log:
            self._mappable.set_norm(colors.LogNorm(*self._mappable.get_clim()))
            self._colorbar.set_norm(self._mappable.norm)
            #clim = self._colorbar.get_clim()
            #nticks = len(self._colorbar.ax.get_yticks())
            #self._colorbar.update_ticks()
            #self._colorbar.set_ticks(numpy.logspace(numpy.log10(clim[0]),
            #                                        numpy.log10(clim[1]),
            #                                        num=nticks,
            #                                        endpoint=True))

