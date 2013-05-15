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
    def __init__(self, ch, sample_rate=None, unit=None, type=None):
        self.name = None
        self.sample_rate = None
        self.unit = None
        self.type = None
        if isinstance(ch, Channel):
            sample_rate = sample_rate or ch.sample_rate
            unit = unit or ch.unit
            ch = ch.name
        self.name = ch
        if sample_rate:
            self.sample_rate = sample_rate
        if unit:
            self.unit = unit
        if type:
            self.type = type

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, n):
        self._name = str(n)
        self.ifo, self.system, self.subsystem, self.signal = (
            parse_channel_name(self.name))

    @property
    def sample_rate(self):
        return self._sample_rate
    @sample_rate.setter
    def sample_rate(self, rate):
        if rate is None:
            self._sample_rate = None
        else:
            self._sample_rate = aunits.Quantity(float(rate), unit=aunits.Hertz)

    @property
    def unit(self):
        return self._unit
    @unit.setter
    def unit(self, u):
        if u is None:
            self._unit = None
        else:
            self._unit = aunits.Unit(u)

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
        try:
            type_ = ndschannel.type
        except AttributeError:
            type_ = None
        return cls(ndschannel.name, sample_rate=ndschannel.sample_rate,
                   type=type_)

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
