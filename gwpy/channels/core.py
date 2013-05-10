# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Provides a LIGO data channel class
"""

import re

from astropy import units as aunits

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


class Channel(object):
    """Representation of a LaserInterferometer data channel
    """
    def __init__(self, name=None, sample_rate=None, unit=None):
        self.name = name
        self.ifo, self.system, self.subsystem, self.signal = (
            parse_channel_name(name))
        self.sample_rate = sample_rate
        self.unit = unit and aunits.Unit(unit) or None

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Channel("%s")' % str(self)

    @property
    def tex_name(self):
        return str(self).replace("_", r"\_")

    def fetch(self, start, end, host=None, port=None):
        from ..io import nds
        if not host or port:
            dhost,dport = nds.DEFAULT_HOSTS[self.ifo]
            host = host or dhost
            port = port or dport
        with nds.NDSConnection(host, port) as connection:
            return connection.fetch(start, end, self)

    @classmethod
    def from_nds(cls, ndschannel):
        return cls(ndschannel.name, sample_rate=ndschannel.sample_rate,
                   type=ndschannel.type)

    @classmethod
    def from_cis(cls, cisjson):
        raise NotImplementedError("Conversion from CIS JSON out to Channel "
                                  "has not been implemented yet")


_re_ifo = re.compile("[A-Z]\d:")
_re_cchar = re.compile("[-_]")

def parse_channel_name(name):
    """Decompose a channel name string into its components
    """
    if not name:
        return None, None, None, None
    # parse ifo
    if _re_ifo.match(name):
        ifo,name = name.split(":",1)
    else:
        ifo = None
    # parse systems
    tags = _re_cchar.split(name, maxsplit=3)
    system = tags[0]
    if len(tags) > 1:
        subsystem = tags[1]
    else:
        subsystem = None
    if len(tags) > 2:
        signal = tags[2]
    else:
        signal = None
    return ifo, system, subsystem, signal
