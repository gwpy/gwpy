#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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

"""Plotting observing segments for O1

The data from the full
`Observing Run 1 (O1) <https://www.gw-openscience.org/O1/>`__
have been released by |GWOSC|_.

This example demonstrates how to download segment information into a
:class:`~gwpy.segments.DataQualityFlag`, and then plot them.
"""

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__currentmodule__ = 'gwpy.segments'

# All we need to do is import the `DataQualityFlag` object, and then call
# the :meth:`DataQualityFlag.fetch_open_data` method to query for, and download
# the segments for all of O1:

from gwpy.segments import DataQualityFlag
h1segs = DataQualityFlag.fetch_open_data('H1_DATA', 'Sep 12 2015',
                                         'Jan 19 2016')

# We can then generate a plot of the times when LIGO-Hanford was operating:

plot = h1segs.plot(color='gwpy:ligo-hanford')
plot.show()

# That's a lot of segments. We can pare-down the list a little to display
# only the segments from the first month of the run:

h1month1 = DataQualityFlag.fetch_open_data('H1_DATA', 'Sep 12 2015',
                                           'Oct 12 2015')

# We can also download the LIGO-Livingston segments from the same period
# and display them alongside, as well as those segments during which both
# interferometers were operating at the same time
# (see :ref:`gwpy-segments-intersection` for more details on this use of the
# ``&`` operator):

l1month1 = DataQualityFlag.fetch_open_data('L1_DATA', 'Sep 12 2015',
                                           'Oct 12 2015')
bothon = h1month1 & l1month1
plot = h1month1.plot()
ax = plot.gca()
ax.plot(l1month1)
ax.plot(bothon, label='Both')

plot.show()
