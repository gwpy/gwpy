
"""Wrapper to the nds2-client package, providing network access
to LIGO data.
"""

import nds2

from .. import (version, detectors)
from ..time import Time

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


DEFAULT_HOSTS = {detectors.LHO_4k.prefix: ('nds.ligo-wa.caltech.edu', 31200),
                 detectors.LLO_4k.prefix: ('nds.ligo-la.caltech.edu', 31200)}


class NDSConnection(object):

    def __init__(self, host, port):
        self._connection = nds2.connection(host, port)

    def fetch(self, start, end, channels):
        if isinstance(channel, basestring):
            channels = [channels]
        channels = map(str, channels)
        gpsstart = isinstance(start, Time) and start.gps or start
        gpsend = isinstance(end, Time) and end.gps or end
        out = self._connection.fetch(gpsstart, gpsend, channels)
        series = []
        for i,data in enumerate(out):
            epoch = Time(data.gps_seconds, data.gps_nanoseconds, format='gps')
            name = data.channel.name
            rate = 1/float(data.channel.sample_rate)
            series.append(TimeSeries(data, name=name, epoch=epoch, dt=rate,
                                     channel=Channel.from_nds(data.channel)))
        if len(series) == 1:
            return series[0]
        else:
            return series

    def close(self):
        del self._connection

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etrace):
        if etype is None:
            self.close()
        else:
            raise
