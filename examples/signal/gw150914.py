#!/usr/bin/env python
# Copyright (C) Duncan Macleod (2016)
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

"""Filtering a `TimeSeries` to detect gravitational waves

The raw 'strain' output of the LIGO detectors is recorded as a `TimeSeries`
with contributions from a large number of known and unknown noise sources,
as well as possible gravitational wave signals.

In order to uncover a real signal we need to filter out noises that otherwise
hide the signal in the data. We can do this by using the :mod:`gwpy.signal`
module to design a digital filter to cut out low and high frequency noise, as
well as notch out fixed frequencies polluted by known artefacts.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First we download the raw strain data from the LOSC public archive:

from gwpy.timeseries import TimeSeries
data = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)

# Next we can design a zero-pole-gain filter to remove the extranious noise.

# First we import the `gwpy.signal.filter_design` module and create a
# :meth:`~gwpy.signal.filter_design.bandpass` filter to remove both low and
# high frequency content

from gwpy.signal import filter_design
bp = filter_design.bandpass(50, 250, data.sample_rate)

# Now we want to combine the bandpass with a series of
# :meth:`~gwpy.signal.filter_design.notch` filters, so we create those
# for the first three harmonics of the 60 Hz AC mains power:

notches = [filter_design.notch(line, data.sample_rate) for
           line in (60, 120, 180)]

# and concatenate each of our filters together to create a single ZPK:

zpk = filter_design.concatenate_zpks(bp, *notches)

# Now, we can apply our combined filter to the data, using `filtfilt=True`
# to filter both backwards and forwards to preserve the correct phase
# at all frequencies

b = data.filter(zpk, filtfilt=True)

# .. note::
#
#    The :mod:`~gwpy.signal.filter_design` methods return digital filters
#    by default, so we apply them using `TimeSeries.filter`. If we had
#    analogue filters (perhaps by passing `analog=True` to the filter design
#    method), the easiest application would be `TimeSeries.zpk`

# Finally, we can :meth:`~TimeSeries.plot` the original and filtered data,
# adding some code to prettify the figure:

from gwpy.plotter import TimeSeriesPlot
plot = TimeSeriesPlot(
    data.crop(*data.span.contract(1)),
    b.crop(*b.span.contract(1)),
    figsize=[12, 8], sep=True, sharex=True, color='gwpy:ligo-hanford')
plot.axes[0].set_title('LIGO-Hanford strain data around GW150914')
plot.axes[0].text(
    1.0, 1.0, 'Unfiltered data',
    transform=plot.axes[0].transAxes, ha='right')
plot.axes[0].set_ylabel('Amplitude [strain]', y=-0.2)
plot.axes[1].text(
    1.0, 1.0, '50-250\,Hz bandpass, notches at 60, 120, 180 Hz',
    transform=plot.axes[1].transAxes, ha='right')
plot.show()

# We see now a spike around 16 seconds into the data, so let's zoom into
# that time by using :meth:`~TimeSeries.crop` and :meth:`~TimeSeries.plot`:

plot = b.crop(1126259462, 1126259462.6).plot(
    figsize=[12, 4], color='gwpy:ligo-hanford')
plot.set_title('LIGO-Hanford strain data around GW150914')
plot.set_ylabel('Amplitude [strain]')
plot.set_epoch(1126259462.427)
plot.show()

# Congratulations, you have succesfully filtered LIGO data to uncover the
# first ever directly-detected gravitational wave signal, GW150914!
# The above filtering is (almost) the same as what was applied to LIGO data
# to produce Figure 1 in
# `Abbott et al. (2016) <https://doi.org/10.1103/PhysRevLett.116.061102>`_
# (the joint LSC-Virgo publication announcing this detection).
