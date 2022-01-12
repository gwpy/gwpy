# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2019-2022)
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

"""Extensions of `~matplotlib.legend` for gwpy
"""

from matplotlib import legend_handler

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class HandlerLine2D(legend_handler.HandlerLine2D):
    """Custom Line2D legend handler that draws lines with custom linewidth

    Parameters
    ----------
    linewidth : `float`, optional
        the linewidth to use when drawing lines on the legend

    **kw
        all keywords are passed to `matplotlib.legend_handler.HandlerLine2D`

    See also
    --------
    matplotlib.legend_handler.HanderLine2D
        for all information
    """
    def __init__(self, linewidth=6., **kw):
        super().__init__(**kw)
        self._linewidth = linewidth

    def create_artists(self, *args, **kwargs):
        artists = super().create_artists(
            *args,
            **kwargs,
        )
        artists[0].set_linewidth(self._linewidth)
        return artists
