# Copyright (c) 2019-2025 Cardiff University
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

"""Extensions of `~matplotlib.legend` for gwpy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import legend_handler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.legend import Legend
    from matplotlib.transforms import Transform

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class HandlerLine2D(legend_handler.HandlerLine2D):
    """Custom Line2D legend handler that draws lines with custom linewidth.

    Parameters
    ----------
    linewidth : `float`, optional
        the linewidth to use when drawing lines on the legend

    kwargs
        All keyword arguments are passed to
        :class`matplotlib.legend_handler.HandlerLine2D`.

    See Also
    --------
    matplotlib.legend_handler.HanderLine2D
        For all information.
    """

    def __init__(self, linewidth: float = 6., **kwargs) -> None:
        """Initialise a new `HandlerLine2D`."""
        super().__init__(**kwargs)
        self._linewidth = linewidth

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]:
        """Create artists for this legend handler."""
        artists = super().create_artists(
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        )
        artists[0].set_linewidth(self._linewidth)
        return artists
