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

"""Boolean array representing the state of some data

Such states are typically the comparison of a `TimeSeries` against some
threshold, where sub-threshold is good and sup-threshold is bad,
for example.

Single `StateTimeSeries` can be bundled together to form `StateVector`
arrays, representing a bit mask of states that combine to make a detailed
statement of instrumental operation
"""

import numpy
import sys
from itertools import izip

if sys.version_info[0] < 3:
    range = xrange

from astropy.units import (Unit, Quantity)

from .core import *
from ..detector import Channel
from ..time import Time
from ..segments import *
from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = ['StateTimeSeries', 'StateVector', 'BitMask']


class StateTimeSeries(TimeSeries):
    """Boolean array representing a good/bad state determination
    of some data.

    Parameters
    ----------
    data : `numpy.ndarray`, `list`
        Data values to initialise TimeSeries
    times : `numpy.ndarray`, optional
        array of time values to accompany data, these are required for
        StateTimeSeries with un-even sampling
    epoch : `float` GPS time, or :class:`~gwpy.time.Time`, optional
        StateTimeSeries start time
    channel : :class:`~gwpy.detector.Channel`, or `str`, optional
        Data channel for this TimeSeries
    unit : :class:`~astropy.units.Unit`, optional
        The units of the data
    sample_rate : `float`, optional
        number of samples per second for this StateTimeSeries
    name : `str`, optional
        descriptive title for this StateTimeSeries

    Returns
    -------
    statebit : `StateTimeSeries`
        A new `StateTimeSeries`

    Attributes
    ----------
    name
    epoch
    channel
    sample_rate

    Methods
    -------
    to_dqflag
    """
    def __new__(cls, data, times=None, epoch=None, channel=None,
                sample_rate=None, name=None, **kwargs):
        """Generate a new StateTimeSeries
        """
        if isinstance(data, (list, tuple)):
           data = numpy.asarray(data)
        if not isinstance(data, cls):
            data = data.astype(bool)
        return super(StateTimeSeries, cls).__new__(cls, data, name=name,
                                                   epoch=epoch, channel=channel,
                                                   sample_rate=sample_rate,
                                                   times=times)

    def to_dqflag(self, name=None, minlen=1, dtype=float, round=False,
                  comment=None):
        """Convert this `StateTimeSeries` into a `DataQualityFlag`.

        Each contiguous set of `True` values are grouped as a `Segment`
        running from the start of the first found `True`, to the end of
        the last.

        Parameters
        ----------
        minlen : `int`, optional, default: 1
            minimum number of consecutive `True` values to identify as a
            `Segment`. This is useful to ignore single bit flips,
            for example.
        dtype : `type`, `callable`, default: `float`
            output segment entry type, can pass either a type for simple
            casting, or a callable function that accepts a float and returns
            another numeric type
        round : `bool`, optional, default: False
            choose to round each `Segment` to its inclusive integer
            boundaries

        Returns
        -------
        dqflag : :class:`~gwpy.segments.flag.DataQualityFlag`
            a segment representation of this `StateTimeSeries`, the span
            defines the `valid` segments, while the contiguouse `True`
            sets defined each of the `active` segments
        """
        active = SegmentList()
        start = self.x0.value
        dt = self.dx.value
        bitstream = iter(self.data)
        i = 0
        while True:
            try:
                if bitstream.next():
                    # found start of True block; find the end
                    j = i + 1
                    try:
                        while bitstream.next():
                            j += 1
                    except StopIteration:
                        pass
                    finally:  # make sure StopIteration doesn't kill final segment
                        if j - i >= minlen:
                           active.append(Segment(start + i * dt,
                                                 start + j * dt))
                    i = j  # advance to end of block
                i += 1
            except StopIteration:
                break
        if dtype is not float:
            active = active.__class__([Segment(dtype(s[0]), dtype(s[1])) for
                                       s in active])
        valid = SegmentList([self.span])
        out = DataQualityFlag(name=name or self.name, active=active,
                              valid=valid, comment=comment or self.name)
        if round:
            out = out.round()
        return out.coalesce()

    def to_lal(self, *args, **kwargs):
        """Bogus function inherited from superclass, do not use.
        """
        raise NotImplementedError("The to_lal method, inherited from the "
                                  "TimeSeries, cannot be used with the "
                                  "StateTimeSeries because LAL has no "
                                  "BooleanTimeSeries structure")


class BitMask(list):
    """Definition of the bits in a `StateVector`.

    Parameters
    ----------
    bits : `list`
        list of bit names
    channel : `Channel`, `str`, optional
        data channel associated with this BitMask
    epoch : `float`, optional
        defining GPS epoch for this `BitMask`
    description : `dict`, optional
        (bit, desc) `dict` of longer descriptions for each bit

    Attributes
    ----------
    channel
    epoch
    description
    """
    def __init__(self, bits, channel=None, epoch=None, description={}):
        list.__init__(self, bits)
        if channel is not None:
            self.channel = channel
        if epoch is not None:
            self.epoch = epoch
        self.description = description
        for i,bit in enumerate(bits):
            if bit is None or bit in self.description:
                continue
            elif channel:
                self.description[bit] = '%s bit %d' % (self.channel, i)
            else:
                self.description[bit] = None

    @property
    def epoch(self):
        """Starting GPS time epoch for this `Array`.

        This attribute is recorded as a `~gwpy.time.Time` object in the
        GPS format, allowing native conversion into other formats.

        See `~astropy.time` for details on the `Time` object.
        """
        try:
            return Time(self._epoch, format='gps')
        except AttributeError:
            return None

    @epoch.setter
    def epoch(self, epoch):
        if isinstance(epoch, Time):
            self._epoch = epoch.gps
        elif isinstance(epoch, Quantity):
            self._epoch = epoch.value
        else:
            self._epoch = float(epoch)

    @property
    def channel(self):
        """Data channel associated with this `TimeSeries`.
        """
        try:
            return self._channel
        except AttributeError:
            return None

    @channel.setter
    def channel(self, ch):
        self._channel = Channel(ch)

    @property
    def description(self):
        """(key, value) dictionary of long bit descriptions.
        """
        return self._description

    @description.setter
    def description(self, desc):
        self._description = desc

    def __repr__(self):
        indent = " " * len('<%s(' % self.__class__.__name__)
        mask = ('\n%s' % indent).join(['%d: %s' % (idx, repr(bit)) for
                                       idx,bit in enumerate(self)
                                       if bit])
        return ("<{1}({2},\n{0}channel={3},\n{0}epoch={4})>".format(
                    indent, self.__class__.__name__,
                    mask, repr(self.channel), repr(self.epoch)))

    def __str__(self):
        indent = " " * len('%s(' % self.__class__.__name__)
        mask = ('\n%s' % indent).join(['%d: %s' % (idx, str(bit)) for
                                       idx,bit in enumerate(self)
                                       if bit])
        return ("{1}({2},\n{0}channel={3},\n{0}epoch={4})".format(
                    indent, self.__class__.__name__,
                    mask, str(self.channel), str(self.epoch)))


class StateVector(TimeSeries):
    """Binary array representing a set of good/bad state determinations
    of some data.

    Each binary bit represents a single boolean condition, with the
    definitins of all the bits stored in the `StateVector.bitmask`
    attribute

    Parameters
    ----------
    data : `numpy.ndarray`, `list`
        Binary data values to initialise `StateVector`
    bitmask : `BitMask`, `list`, optional
        list of bits defining this `StateVector`
    times : `numpy.ndarray`, optional
        array of time values to accompany data, these are required for
        `StateVector` with un-even sampling
    name : `str`, optional
        descriptive title for this StateTimeSeries
    epoch : `float` GPS time, or :class:`~gwpy.time.Time`, optional
        starting GPS epoch for this `StateVector`
    channel : :class:`~gwpy.detector.Channel`, or `str`, optional
        data channel associated with this `StateVector`
    sample_rate : `float`, optional
        data rate for this `StateVector` in samples per second (Hertz).

    Attributes
    ----------
    name
    epoch
    channel
    sample_rate
    duration
    span
    bitmask
    boolean
    bits

    Methods
    -------
    to_dqflags
    read
    fetch
    """
    _metadata_slots = TimeSeries._metadata_slots + ['bitmask']
    def __new__(cls, data, bitmask=[], times=None, epoch=None, channel=None,
                sample_rate=None, name=None, **kwargs):
        """Generate a new StateTimeSeries
        """
        if not isinstance(data, cls):
            data = data.astype(numpy.uint64)
        return super(StateVector, cls).__new__(cls, data, name=name,
                                               epoch=epoch, channel=channel,
                                               sample_rate=sample_rate,
                                               times=times, bitmask=bitmask)

    # -------------------------------------------
    # StateVector properties

    @property
    def bitmask(self):
        """The list of bit names for this `StateVector`.
        """
        try:
            return self.metadata['bitmask']
        except:
            self.bitmask = BitMask([])
            return self.bitmask

    @bitmask.setter
    def bitmask(self, mask):
        if not isinstance(mask, BitMask):
            mask = BitMask(mask, channel=self.channel, epoch=self.epoch)
        self.metadata['bitmask'] = mask

    @property
    def boolean(self):
        """A mapping of this `StateVector` to a 2-D array containing all
        binary bits as booleans, for each time point.
        """
        try:
            return self._boolean
        except AttributeError:
            nbits = len(self.bitmask)
            boolean = numpy.zeros((self.size, nbits), dtype=bool)
            for i,d in enumerate(self.data):
                boolean[i,:] = [int(d)>>j & 1 for
                                j in xrange(nbits)]
            self._boolean = ArrayTimeSeries(boolean, name=self.name,
                                            epoch=self.epoch,
                                            sample_rate=self.sample_rate, 
                                            y0=0, dy=1)
            return self.boolean

    @property
    def bits(self):
        """A list of `StateTimeSeries` for each of the individual
        bits in this `StateVector`.
        """
        try:
            return self._bits
        except AttributeError:
            self._bits = [StateTimeSeries(
                              self.boolean.data[:,i], name=bit,
                              epoch=self.x0.value, channel=self.channel,
                              sample_rate=self.sample_rate) for
                          i,bit in enumerate(self.bitmask)]
            return self.bits

    # -------------------------------------------
    # StateVector methods

    def to_dqflags(self, minlen=1, dtype=float, round=False):
        """Convert this `StateVector` into a `SegmentListDict`.

        The `StateTimeSeries` for each bit is converted into a `SegmentList`
        with the bits combined into a dict.

        Parameters
        ----------
        minlen : `int`, optional, default: 1
           minimum number of consecutive `True` values to identify as a
           `Segment`. This is useful to ignore single bit flips,
           for example.

        Returns
        -------
        DataQualityFlag list : `list`
            a list of :class:`~gwpy.segments.flag.DataQualityFlag`
            reprensentations for each bit in this `StateVector`

        See Also
        --------
        :meth:`StateTimeSeries.to_dqflag`
            for details on the segment representation method for
            `StateVector` bits
        """
        return [self.bits[i].to_dqflag(
                    name=bit, minlen=minlen, round=round, dtype=dtype, 
                    comment=self.bitmask.description[bit])
                for i,bit in enumerate(self.bitmask)]

    @classmethod
    def read(cls, source, channel, bitmask=[], start=None, end=None,
             datatype=numpy.uint64, verbose=False):
        """Read a `StateVector` channel from a given source.

        Parameters
        ----------
        source : `str`, :class:`glue.lal.Cache`, :lalsuite:`LALCache`
            source for data, one of:

            - a filepath for a GWF-format frame file,
            - a filepath for a LAL-format Cache file
            - a Cache object from GLUE or LAL
        channel : `str`, :class:`~gwpy.detector.channel.Channel`
            channel (name or object) to read
        bitmask : `BitMask`, `list`, optional
            definition of bits for this `StateVector`
        start : :class:`~gwpy.time.Time`, `float`, optional
            start GPS time of desired data
        end : :class:`~gwpy.time.Time`, `float`, optional
            end GPS time of desired data
        datatype : `type`, `numpy.dtype`, `str`, optional
            identifier for desired output data type, default: `uint64`
        verbose : `bool`, optional
            print verbose output

        Returns
        -------
        state : `StateVector`
            a new `StateVector` read from the given source
        """
        new = super(StateVector, cls).read(source, channel, start=start,
                                           end=end, datatype=datatype,
                                           verbose=verbose)
        new.bitmask = bitmask
        return new

    @classmethod
    def fetch(cls, channel, start, end, bitmask=[], host=None,
              port=None, verbose=False):
        """Fetch data from NDS into a `StateVector`.

        Parameters
        ----------
        channel : :class:`~gwpy.detector.channel.Channel`, or `str`
            required data channel
        start : `~gwpy.time.Time`, or float
            GPS start time of data span
        end : `~gwpy.time.Time`, or float
            GPS end time of data span
        bitmask : `BitMask`, `list`, optional
            definition of bits for this `StateVector`
        host : `str`, optional
            URL of NDS server to use, defaults to observatory site host
        port : `int`, optional
            port number for NDS server query, must be given with `host`
        verbose : `bool`, optional
            print verbose output about NDS progress

        Returns
        -------
        StateVector
            a new `StateVector` containing the data read from NDS
        """
        new = super(StateVector, cls).fetch(channel, start, end,
                                            host=host, port=port,
                                            verbose=verbose)
        new.bitmask = bitmask
        return new

    def to_lal(self, *args, **kwargs):
        """Bogus function inherited from superclass, do not use.
        """
        raise NotImplementedError("The to_lal method, inherited from the "
                                  "TimeSeries, cannot be used with the "
                                  "StateTimeSeries because LAL has no "
                                  "BooleanTimeSeries structure")

    def plot(self, format='segments', **kwargs):
        """Plot the data for this `StateVector`

        Parameters
        ----------
        format : `str`, optional, default: ``'segments'``
            type of plot to make, either 'segments' to plot the
            SegmentList for each bit, or 'timeseries' to plot the raw
            data for this `StateVector`.
        **kwargs
            other keyword arguments to be passed to either
            :class:`~gwpy.plotter.segments.SegmentPlot` or
            :class:`~gwpy.plotter.timeseries.TimeSeriesPlot`, depending
            on ``format``.

        Returns
        -------
        plot : :class:`~gwpy.plotter.segments.SegmentPlot`, or
               :class:`~gwpy.plotter.timeseries.TimeSeriesPlot`
            output plot object, subclass of :class:`~gwpy.plotter.core.Plot`
        """
        if format == 'timeseries':
            return super(StateVector, self).plot(**kwargs)
        elif format == 'segments':
            from ..plotter import SegmentPlot
            return SegmentPlot(*self.to_dqflags(), **kwargs)
        raise ValueError("'format' argument must be one of: 'timeseries' or "
                         "'segments'")

