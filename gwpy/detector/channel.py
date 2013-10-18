# Copyright (C) Duncan Macleod (2013)
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""Provides a LIGO data channel class
"""

import re
import numpy


from astropy import units as aunits

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


class Channel(object):
    """Representation of a LaserInterferometer data channel.

    Parameters
    ----------
    ch : `str`, `Channel`
        name of this Channel (or another  Channel itself).
        If a `Channel` is given, all other parameters not set explicitly
        will be copied over.
    sample_rate : `float`, |Quantity|, optional
        number of samples per second
    unit : :class:`~astropy.units.core.Unit`, `str`, optional
        name of the unit for the data of this channel
    dtype : `numpy.dtype`, optional
        numeric type of data for this channel
    model : `str`, optional
        name of the SIMULINK front-end model that produces this `Channel`

    Notes
    -----
    The `Channel` structure implemented here is designed to match the
    data recorded in the LIGO Channel Information System
    (https://cis.ligo.org) for which a query interface is provided.

    Attributes
    ----------
    name
    ifo
    system
    subsystem
    signal
    sample_rate
    unit
    dtype
    model

    Methods
    -------
    query
    """
    def __init__(self, ch, sample_rate=None, unit=None, dtype=None,
                 type=None, model=None):
        # test for Channel input
        if isinstance(ch, Channel):
            sample_rate = sample_rate or ch.sample_rate
            unit = unit or ch.unit
            type = type or ch.type
            dtype = dtype or ch.dtype
            model = model or ch.model
            ch = ch.name
        # set attributes
        self.name = ch
        self.sample_rate = sample_rate
        self.unit = unit
        self.type = type
        self.dtype = dtype
        self.model = model

    @property
    def name(self):
        """Name of this `Channel`. This should follow the naming
        convention, with the following format: 'IFO:SYSTEM-SUBSYSTEM_SIGNAL'

        :type: `str`
        """
        return self._name

    @name.setter
    def name(self, n):
        self._name = str(n)
        self._ifo, self._system, self._subsystem, self._signal = (
            parse_channel_name(self.name))

    @property
    def ifo(self):
        """Interferometer prefix for this `Channel`, e.g `H1`.

        :type: `str`
        """
        return self._ifo

    @property
    def system(self):
        """Instrumental system for this `Channel`, e.g `PSL`
        (pre-stabilised laser).

        :type: `str`
        """
        return self._system

    @property
    def subsystem(self):
        """Instrumental sub-system for this `Channel`, e.g `ISS`
        (pre-stabiliser laser intensity stabilisation servo).

        :type: `str`
        """
        return self._subsystem

    @property
    def signal(self):
        """Instrumental signal for this `Channel`, relative to the
        system and sub-system, e.g `FIXME`.

        :type: `str`
        """
        return self._signal

    @property
    def sample_rate(self):
        """Rate of samples (Hertz) for this `Channel`

        :type: |Quantity|
        """
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate):
        if isinstance(rate, aunits.Unit):
            self._sample_rate = rate
        elif rate is None:
            self._sample_rate = None
        elif isinstance(rate, aunits.Quantity):
            self._sample_rate = rate
        else:
            self._sample_rate = aunits.Quantity(float(rate), unit=aunits.Hertz)

    @property
    def unit(self):
        """Data unit for this `Channel`

        :type: |Unit|
        """
        return self._unit

    @unit.setter
    def unit(self, u):
        if u is None:
            self._unit = None
        else:
            self._unit = aunits.Unit(u)

    @property
    def model(self):
        """Name of the SIMULINK front-end model that defines this `Channel`

        :type: `str`
        """
        return self._model

    @model.setter
    def model(self, mdl):
        self._model = mdl and mdl.lower() or mdl

    @property
    def dtype(self):
        """Numeric type for data in this `Channel`

        :type: :class:`~numpy.dtype`
        """
        return self._dtype

    @dtype.setter
    def dtype(self, type_):
        self._dtype = numpy.dtype(type_)

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Channel("%s")' % str(self)

    @property
    def tex_name(self):
        """Name of this `Channel` in LaTeX printable format
        """
        return str(self).replace("_", r"\_")

    @classmethod
    def query(cls, name, debug=False):
        """Query the LIGO Channel Information System for the `Channel`
        matching the given name

        Parameters
        ----------
        name : `str`
            name of channel
        debug : `bool`, optional
            print verbose HTTP connection status for debugging,
            default: `False`

        Returns
        -------
        Channel
             a new `Channel` containing all of the attributes set from
             its entry in the CIS
        """
        channellist = ChannelList.query(name, debug=debug)
        if len(channellist) == 0:
            raise ValueError("No channels found matching '%s'." % name)
        if len(channellist) > 1:
            raise ValueError("%d channels found matching '%s', please refine "
                             "search, or use `ChannelList.query` to return "
                             "all results." % (len(channellist), name))
        return channellist[0]

    @classmethod
    def from_nds2(cls, nds2channel):
        """Generate a new channel using an existing nds2.channel object
        """
        name = nds2channel.name
        sample_rate = nds2channel.sample_rate
        unit = nds2channel.signal_units
        if not unit:
            unit = None
        ctype = nds2channel.channel_type_to_string(nds2channel.channel_type)
        dtypestr = nds2channel.data_type_to_string(nds2channel.data_type)
        try:
            from lal import utils as lalutils
        except ImportError:
            dtype=None
        else:
            laltype = lalutils.LAL_TYPE_FROM_STR[
                          dtypestr.replace('_', '').upper()]
            dtype = lalutils.NUMPY_TYPE_FROM_LAL[laltype]
        return cls(name, sample_rate=sample_rate, unit=unit, dtype=dtype,
                   type=ctype)

_re_ifo = re.compile("[A-Z]\d:")
_re_cchar = re.compile("[-_]")


def parse_channel_name(name):
    """Decompose a channel name string into its components
    """
    if not name:
        return None, None, None, None
    # parse ifo
    if _re_ifo.match(name):
        ifo, name = name.split(":", 1)
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

from .channel import Channel


class ChannelList(list):
    """A list of Channels, with parsing/sieveing utilities.
    """
    def find(self, name):
        """Find the channel with the given name in this ChannelList.

        Parameters
        ----------
        name : `str`
            name of the `Channel` to find

        Returns
        -------
        idx : `int`
            returns the position of the first `Channel` in self
            whose name matches the input

        Raises
        ------
        ValueError if no such element exists.
        """
        for i, chan in enumerate(self):
            if name == chan.name:
                return i
        raise ValueError(name)

    def sieve(self, name=None, sample_rate=None, sample_range=None,
              exact_match=False):
        """Find all Channels in this list that match the specified
        criteria.

        Parameters
        ----------
        name : `str`, or regular expression
            any part of the channel name against which to match
            (or full name if `exact_match=False` is given)
        sample_rate : `float`
            rate (number of samples per second) to match exactly
        sample_range : 2-`tuple`
            `[low, high]` closed interval or rates to match within
        exact_match : `bool`
            return channels matching `name` exactly, default: `False`

        Returns
        -------
        new : `ChannelList`
            a new `ChannelList` containing the matching channels
        """
        # format name regex
        if isinstance(name, re._pattern_type):
            flags = name.flags
            name = name.pattern
        else:
            flags = 0
        if exact_match:
            name = name.startswith('\\A') and name or r"\A%s" % name
            name = name.endwith('\\Z') and name or r"%s\Z" % name
        name_regexp = re.compile(name, flags=flags)
        c = list(self)
        if name is not None:
            c = [entry for entry in c if
                 name_regexp.search(entry.name) is not None]
        if sample_rate is not None:
            c = [entry for entry in c if
                 float(entry.sample_rate) == sample_rate]
        if sample_range is not None:
            c = [entry for entry in c if
                 sample_range[0] <= float(entry.sample_rate) <=
                 sample_range[1]]

        return self.__class__(c)

    @classmethod
    def query(cls, name, debug=False):
        """Query the LIGO Channel Information System a `ChannelList`
        of entries matching the given name regular expression.

        Parameters
        ----------
        name : `str`
            name of channel, or part of it.
        debug : `bool`, optional
            print verbose HTTP connection status for debugging,
            default: `False`

        Returns
        -------
        `ChannelList`
        """
        from ..io import cis
        return cis.query(name, debug=debug)

    @property
    def ifos(self):
        return set([c.ifo for c in self.channels])
