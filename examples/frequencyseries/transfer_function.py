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

"""
.. sectionauthor:: Duncan Macleod <duncan.macleod@ligo.org>
.. currentmodule:: gwpy.timeseries

Transfer function
#################

In this example we demonstrate how to calculate the transfer function
between two `TimeSeries` signals.

All ground-based gravitational wave observatories would be unable to
operate if they did not employ sophisticated ground-motion suppression
technology to prevent vibrations from the local (or remote) environment
from transferring through to optical components.

The impact of the seismic isolation system can be seen by calculating
the transfer function between the ground motion at the laboratory
and that of the optical suspension points.
"""

from gwpy.time import tconvert
from gwpy.timeseries import TimeSeriesDict
from gwpy.plot import BodePlot

# %%
# For this example we will use data from the |GWOSC_AUX_GW170817|_:

start = tconvert("August 14 2017 10:25")
end = start + 1800
gndchannel = "L1:ISI-GND_STS_ITMY_Y_DQ"
suschannel = "L1:ISI-ITMY_SUSPOINT_ITMY_EUL_L_DQ"

# %%
# We can call the :meth:`~TimeSeriesDict.get` method of the `TimeSeriesDict`
# to retrieve all data in a single operation:
data = TimeSeriesDict.get(
    [gndchannel, suschannel],
    start,
    end,
    host="nds.gwosc.org",
)
gnd = data[gndchannel]
sus = data[suschannel]

# %%
# The transfer function between time series is easily computed with the
# :meth:`~TimeSeries.transfer_function` method:
tf = gnd.transfer_function(sus, fftlength=128, overlap=64)

# %%
# The `~gwpy.plot.BodePlot` knows how to separate a complex-valued
# `~gwpy.frequencyseries.FrequencySeries` into magnitude and phase:
plot = BodePlot(tf)
plot.maxes.set_title(
    r"L1 ITMY ground $\rightarrow$ SUS transfer function",
)
plot.maxes.set_xlim(5e-2, 30)
plot.show()

# This example demonstrates the impressive noise suppression of the LIGO
# seismic isolation system. For more details, please see
# https://www.ligo.caltech.edu/page/vibration-isolation.
