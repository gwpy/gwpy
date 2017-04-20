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

"""An extension of the Plot class for handling Spectrograms
"""


from .timeseries import TimeSeriesPlot

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['SpectrogramPlot']


class SpectrogramPlot(TimeSeriesPlot):
    """`Figure` for displaying a `~gwpy.spectrogram.Spectrogram`.
    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('sep', True)
        super(SpectrogramPlot, self).__init__(*args, **kwargs)
