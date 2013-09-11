
"""Wrapper to the nds2-client package, providing network access
to LIGO data.
"""

import nds2
import os
import sys
from math import (floor, ceil)

from ... import (version, detector)
from ...detector import Channel
from ...time import Time
from ...timeseries import TimeSeries

from .kerberos import *

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


try:
    from collections import OrderedDict
except ImportError:
    from astropy.utils import OrderedDict
finally:
    DEFAULT_HOSTS = OrderedDict([
                    (None,('nds.ligo.caltech.edu', 31200)),
                    (detector.LHO_4k.prefix,('nds.ligo-wa.caltech.edu', 31200)),
                    (detector.LLO_4k.prefix,('nds.ligo-la.caltech.edu', 31200)),
                    (detector.CIT_40.prefix,('nds40.ligo.caltech.edu', 31200))])


class NDSRedirectStdStreams(object):
    def __init__(self, stdout=sys.stdout, stderr=sys.stderr):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class NDSWarning(UserWarning):
    pass


class NDSConnection(object):

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._connection = nds2.connection(host, port)

    def fetch(self, start, end, channels, silent=False):
        if isinstance(channels, basestring) or isinstance(channels, Channel):
            channels = [channels]
        for i,c in enumerate(channels):
            try:
                channels[i] = self._connection.find_channels('%s,*' % c)[0].name
            except IndexError:
                try:
                    channels[i] = self._connection.find_channels(c.name)[0].name
                except IndexError:
                    raise RuntimeError("Cannot find channel '%s' in NDS "
                                       "database on %s" % (c, self.host))
        gpsstart = int(floor(isinstance(start, Time) and start.gps or start))
        gpsend = int(ceil(isinstance(end, Time) and end.gps or end))
        if silent:
            outputcontext = NDSRedirectStdStreams(open(os.devnull, 'w'),
                                                  open(os.devnull, 'w'))
        else:
            outputcontext = NDSRedirectStdStreams()
        with outputcontext:
            out = self._connection.fetch(gpsstart, gpsend, channels)
        series = []
        for i,data in enumerate(out):
            epoch = Time(data.gps_seconds, data.gps_nanoseconds, format='gps')
            channel = Channel.from_nds2(data.channel)
            try:
                cisch = Channel.query(channel.name)
            except ValueError:
                pass
            else:
                channel.model = cisch.model
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


_HOST_RESOLTION_ORDER = ['nds.ligo.caltech.edu'] + DEFAULT_HOSTS.values()
def host_resolution_order(ifo):
    hosts = []
    if ifo in DEFAULT_HOSTS:
        hosts.append(DEFAULT_HOSTS[ifo])
    for difo,hp in DEFAULT_HOSTS.iteritems():
        if difo != ifo:
            hosts.append(hp)
    return hosts
