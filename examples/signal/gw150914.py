#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2016-2020)
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

# First we download the raw strain data from the GWOSC public archive:

from gwpy.timeseries import TimeSeries
hdata = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)

# Next we can design a zero-pole-gain filter to remove the extranious noise.

# First we import the `gwpy.signal.filter_design` module and create a
# :meth:`~gwpy.signal.filter_design.bandpass` filter to remove both low and
# high frequency content

from gwpy.signal import filter_design
bp = filter_design.bandpass(50, 250, hdata.sample_rate)

# Now we want to combine the bandpass with a series of
# :meth:`~gwpy.signal.filter_design.notch` filters, so we create those
# for the first three harmonics of the 60 Hz AC mains power:

notches = [filter_design.notch(line, hdata.sample_rate) for
           line in (60, 120, 180)]

# and concatenate each of our filters together to create a single ZPK:

zpk = filter_design.concatenate_zpks(bp, *notches)

# Now, we can apply our combined filter to the data, using `filtfilt=True`
# to filter both backwards and forwards to preserve the correct phase
# at all frequencies

hfilt = hdata.filter(zpk, filtfilt=True)

# .. note::
#
#    The :mod:`~gwpy.signal.filter_design` methods return digital filters
#    by default, so we apply them using `TimeSeries.filter`. If we had
#    analogue filters (perhaps by passing `analog=True` to the filter design
#    method), the easiest application would be `TimeSeries.zpk`

# The :mod:`~gwpy.signal.filter_design` methods return infinite impulse
# response filters by default, which, when applied, corrupt a small amount of
# data at the beginning and the end of our original `TimeSeries`.
# We can discard those data using the :meth:`~TimeSeries.crop` method
# (for consistency we apply this to both data series):

hdata = hdata.crop(*hdata.span.contract(1))
hfilt = hfilt.crop(*hfilt.span.contract(1))

# Finally, we can :meth:`~TimeSeries.plot` the original and filtered data,
# adding some code to prettify the figure:

from gwpy.plot import Plot
plot = Plot(hdata, hfilt, figsize=[12, 6], separate=True, sharex=True,
            color='gwpy:ligo-hanford')
ax1, ax2 = plot.axes
ax1.set_title('LIGO-Hanford strain data around GW150914')
ax1.text(1.0, 1.01, 'Unfiltered data', transform=ax1.transAxes, ha='right')
ax1.set_ylabel('Amplitude [strain]', y=-0.2)
ax2.set_ylabel('')
ax2.text(1.0, 1.01, r'50-250\,Hz bandpass, notches at 60, 120, 180 Hz',
         transform=ax2.transAxes, ha='right')
plot.show()
plot.close()  # hide

# We see now a spike around 16 seconds into the data, so let's zoom into
# that time (and prettify):

plot = hfilt.plot(color='gwpy:ligo-hanford')
ax = plot.gca()
ax.set_title('LIGO-Hanford strain data around GW150914')
ax.set_ylabel('Amplitude [strain]')
ax.set_xlim(1126259462, 1126259462.6)
ax.set_xscale('seconds', epoch=1126259462)
plot.show()
plot.close()  # hide

# Congratulations, you have succesfully filtered LIGO data to uncover the
# first ever directly-detected gravitational wave signal, GW150914!
# But wait, what about LIGO-Livingston?
# We can easily add that to our figure by following the same procedure.
#
# First, we load the new data

ldata = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
lfilt = ldata.filter(zpk, filtfilt=True)

# The article announcing the detector told us that the signals were
# separated by 6.9 ms between detectors, and that the relative orientation
# of those detectors means that we need to invert the data from one
# before comparing them, so we apply those corrections:

lfilt.shift('6.9ms')
lfilt *= -1

# and finally make a new plot with both detectors:

plot = Plot(figsize=[12, 4])
ax = plot.gca()
ax.plot(hfilt, label='LIGO-Hanford', color='gwpy:ligo-hanford')
ax.plot(lfilt, label='LIGO-Livingston', color='gwpy:ligo-livingston')
ax.set_title('LIGO strain data around GW150914')
ax.set_xlim(1126259462, 1126259462.6)
ax.set_xscale('seconds', epoch=1126259462)
ax.set_ylabel('Amplitude [strain]')
ax.set_ylim(-1e-21, 1e-21)
ax.legend()
plot.show()

# The above filtering is (almost) the same as what was applied to LIGO data
# to produce Figure 1 in
# `Abbott et al. (2016) <https://doi.org/10.1103/PhysRevLett.116.061102>`_
# (the joint LSC-Virgo publication announcing this detection).
