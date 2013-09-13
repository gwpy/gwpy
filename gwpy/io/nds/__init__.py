
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

    def fetch(self, start, end, channels, ndschanneltype=None, silent=False):
        # find channels
        if isinstance(channels, basestring) or isinstance(channels, Channel):
            channels = [channels]
        channels = map(lambda c: c.name, self.find(channels, ndschanneltype))
        # format times
        gpsstart = int(floor(isinstance(start, Time) and start.gps or start))
        gpsend = int(ceil(isinstance(end, Time) and end.gps or end))
        # set verbose context
        if silent:
            outputcontext = NDSRedirectStdStreams(open(os.devnull, 'w'),
                                                  open(os.devnull, 'w'))
        else:
            outputcontext = NDSRedirectStdStreams()
        # fetch data
        with outputcontext:
            out = self._connection.fetch(gpsstart, gpsend, channels)
        # convert to TimeSeries
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
        # return
        if len(series) == 1:
            return series[0]
        else:
            return series

    def find(self, channels, nds2channeltype=None, nds2datatype=None,
             minsamp=0, maxsamp=None):
        """Search for the given channels in the NDS2 database for this host
        """
        # format args for nds module call
        args = [arg for arg in (nds2channeltype, nds2datatype) if
                arg is not None]
        if maxsamp:
            args.extend([minsamp, maxsamp])
        if isinstance(channels, basestring):
            channels = [channels]
        # loop over channels, returning all found
        out = []
        for channel in channels:
            if isinstance(channel, nds2.channel):
                out.append(channel)
            else:
                channel = str(channel)
                try:
                    out.append(self._connection.find_channels(
                                   "%s,*" % channel, *args)[0])
                except IndexError:
                    out.extend(self._connection.find_channels(channel, *args))
        return out

    def iterate(channels, start=None, stop=None, stride=None):
        """Retreive data over NDS in pieces
        """
        # format args for nds module call
        args = [arg for arg in (start, stop, stride, channel) if
                arg is not None]
        # return iterator
        return self._connection.iterate(*args)

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
