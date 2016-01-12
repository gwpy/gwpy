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

"""Custom filtering utilities for the `TimeSeries`
"""

from __future__ import division

from scipy import signal

from astropy.units import Quantity

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


def create_notch(frequency, sample_rate, type='iir', **kwargs):
    """Design a ZPK notch filter for the given frequency and sampling rate
    """
    frequency = Quantity(frequency, 'Hz').value
    sample_rate = Quantity(sample_rate, 'Hz').value
    nyq = 0.5 * sample_rate
    df = 1.0
    df2 = 0.1
    low1 = (frequency - df)/nyq
    high1 = (frequency + df)/nyq
    low2 = (frequency - df2)/nyq
    high2 = (frequency + df2)/nyq
    if type == 'iir':
        kwargs.setdefault('gpass', 1)
        kwargs.setdefault('gstop', 10)
        kwargs.setdefault('ftype', 'ellip')
        return signal.iirdesign([low1, high1], [low2, high2], output='zpk',
                                **kwargs)
    else:
        raise NotImplementedError("Generating %r notch filters has not been "
                                  "implemented yet" % type)
