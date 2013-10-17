
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
from ...timeseries import (TimeSeries, TimeSeriesList)

from .kerberos import *

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


try:
    from collections import OrderedDict
except ImportError:
    from astropy.utils import OrderedDict
finally:
    DEFAULT_HOSTS = OrderedDict([
                    (None,('ldas-pcdev4.ligo.caltech.edu', 31200)),
                    (None,('nds.ligo.caltech.edu', 31200)),
                    (detector.LHO_4k.prefix,('nds.ligo-wa.caltech.edu', 31200)),
                    (detector.LLO_4k.prefix,('nds.ligo-la.caltech.edu', 31200)),
                    (detector.CIT_40.prefix,('nds40.ligo.caltech.edu', 31200))])


class NDSOutputContext(object):
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


class NDS2Connection(nds2.connection):
    """Thin wrapper of the NDS2 client library `connection` object

    Provides some trivial niceties to clean up user-code

    Parameters
    ----------
    host : `str`
        URL of NDS(2) connection host
    port : `int`, optional, default: ``31200``
        port number for NDS(2) connection on host

    Returns
    -------
    connection
        a new (open) `NDS2Connection`
    """
    def __init__(self, host, port=31200):
        """Set up a new connection to a given NDS(2) host
        """
        super(NDS2Connection, self).__init__(host, port)

    @property
    def host(self):
        """The host URL for this NDS(2) connection
        """
        return self.get_host()

    @property
    def port(self):
        """The host port number for this NDS(2) connection
        """
        return self.get_port()

    def _find(self, channels, nds2channeltype=None, nds2datatype=None,
              minsamp=0, maxsamp=None):
        """Search for the given channels in the NDS2 database for this host
        """
        # format args for nds module call
        args = [arg for arg in (nds2channeltype, nds2datatype) if
                arg is not None]
        if maxsamp:
            args.extend([minsamp, maxsamp])
        if isinstance(channels, basestring) or isinstance(channels, Channel):
            channels = [channels]
        # loop over channels, returning all found
        out = []
        for channel in channels:
            if isinstance(channel, nds2.channel):
                out.append(channel)
            else:
                channel = str(channel)
                try:
                    out.append(self.find_channels(
                                   "%s,*" % channel, *args)[0])
                except IndexError:
                    out.extend(self.find_channels(channel, *args))
        return out

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
