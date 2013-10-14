#!/usr/bin/env python

"""GWpy Example: plotting a spectrum

Problem
-------

I'm interested in the level of ground motion surrounding a particular time
during commissioning of the Advanced LIGO Livingston Observatory. I don't
have access to the frame files on disk, so I'll need to use NDS.

"""

from gwpy.time import Time
from gwpy.timeseries import TimeSeries

from gwpy import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


# set the times
start = Time('2010-08-01 00:00:00', format='iso', scale='utc')
end = Time('2010-08-01 00:02:00', format='iso', scale='utc')

# read the data over the network
lho = TimeSeries.fetch('H1:LDAS-STRAIN', start.gps, end.gps, verbose=True)
lho.unit = 'strain'
llo = TimeSeries.fetch('L1:LDAS-STRAIN', start.gps, end.gps, verbose=True)
llo.unit = 'strain'

# calculate spectrum with 0.5Hz resolution
lhoasd = lho.asd(2, 1)
lloasd = llo.asd(2, 1)

# plot
plot = lhoasd.plot(color='b', label='LHO')
plot.add_spectrum(lloasd, color='g', label='LLO')
plot.xlim = [40, 4096]
plot.ylim = [1e-23, 7.5e-21]

# hide some axes
plot.axes[0].spines['right'].set_visible(False)
plot.axes[0].spines['top'].set_visible(False)
plot.axes[0].get_xaxis().tick_bottom()
plot.axes[0].get_yaxis().tick_right()

if __name__ == '__main__':
    try:
        outfile = __file__.replace('.py', '.png')
    except NameError:
        pass
    else:
        plot.save(outfile)
        print("Example output saved as\n%s" % outfile)
