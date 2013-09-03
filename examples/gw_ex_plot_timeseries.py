#!/usr/bin/env python

"""GWpy Example: plotting a time-series

Problem
-------

I would like to study the gravitational wave strain time-series around the time of an interesting simulated signal during the last science run (S6). I have access to the frame files on the LIGO Data Grid machine `ldas-pcdev2.ligo-wa.caltech.edu` and so can read them directly.
"""

from glue import datafind

from gwpy.time import Time
from gwpy.data import TimeSeries

from gwpy import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


# set the times
start = Time('2010-09-16 06:42:00', format='iso', scale='utc')
end = Time('2010-09-16 06:43:00', format='iso', scale='utc')

# find the data: uses glue.datafind service
connection = datafind.GWDataFindHTTPConnection()
framecache = connection.find_frame_urls('H', 'H1_LDAS_C02_L2',
                                        start.gps, end.gps, urltype='file')

# make timeseries
data = TimeSeries.read(framecache, 'H1:LDAS-STRAIN', epoch=start.gps, duration=60)

# plot
plot = data.plot()
