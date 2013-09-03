
"""Wrapper to the nds2-client package, providing network access
to LIGO data.
"""


from math import (floor, ceil)

import nds2

from .. import (version, detector)
from ..detector import Channel
from ..time import Time
from ..data import TimeSeries

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


DEFAULT_HOSTS = {detector.LHO_4k.prefix: ('nds.ligo-wa.caltech.edu', 31200),
                 detector.LLO_4k.prefix: ('nds.ligo-la.caltech.edu', 31200),
                 detector.CIT_40.prefix: ('nds40.ligo.caltech.edu', 31200)}


class NDSConnection(object):

    def __init__(self, host, port):
        self._connection = nds2.connection(host, port)

    def fetch(self, start, end, channels):
        if isinstance(channels, basestring) or isinstance(channels, Channel):
            channels = [channels]
        channels = map(str, channels)
        gpsstart = int(floor(isinstance(start, Time) and start.gps or start))
        gpsend = int(ceil(isinstance(end, Time) and end.gps or end))
        print(gpsstart, gpsend, channels)
        out = self._connection.fetch(gpsstart, gpsend, channels)
        series = []
        for i,data in enumerate(out):
            epoch = Time(data.gps_seconds, data.gps_nanoseconds, format='gps')
            channel = Channel.query(data.channel.name)
            series.append(TimeSeries(data.data, channel=channel, epoch=epoch))
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
