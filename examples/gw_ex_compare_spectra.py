#!/usr/bin/env python

"""GWpy Example: comparing a the Spectrum of a channel at different times

Problem
-------

I'm interested in comparing the amplitude spectrum of a channel between a
known 'good' time - where the spectrum is what we expect it to be - and a
known 'bad' time - where some excess noise appeared and the spectrum
changed appreciably.

"""

from gwpy.time import Time, TimeDelta
from gwpy.timeseries import TimeSeries

from gwpy import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

# set the times
goodtime = Time(1061800700, format='gps')
badtime = Time(1061524816, format='gps')
duration = TimeDelta(120, format='sec')

# read the data over the network
gooddata = TimeSeries.fetch('L1:PSL-ISS_PDB_OUT_DQ', goodtime,
                            goodtime+duration)
baddata = TimeSeries.fetch('L1:PSL-ISS_PDB_OUT_DQ', badtime,
                           badtime+duration)

# calculate spectrum with 1.8 Hz resolution
goodasd = gooddata.asd(8, 4, 'welch')
badasd = baddata.asd(8, 4, 'welch')

# plot
plot = badasd.plot()
plot.add_spectrum(goodasd)
plot.xlim = [10, 8000]
plot.ylim = [1e-6, 5e-4]

from gwpy.plotter import IS_INTERACTIVE
if not IS_INTERACTIVE and not '__IPYTHON__' in globals():
    outfile = __file__.replace('.py', '.png')
    plot.save(outfile)
    print("Example output saved as\n%s" % outfile)
