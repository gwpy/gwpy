# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2021)
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

"""This module defines the Boolean array representing the state of some data

Such states are typically the comparison of a `TimeSeries` against some
threshold, where sub-threshold is good and sup-threshold is bad,
for example.

Single `StateTimeSeries` can be bundled together to form `StateVector`
arrays, representing a bit mask of states that combine to make a detailed
statement of instrumental operation
"""

from functools import wraps
from math import (ceil, log)

import numpy

from astropy import units

from .core import (TimeSeriesBase, TimeSeriesBaseDict, TimeSeriesBaseList,
                   as_series_dict_class)
from ..types import Array2D
from ..detector import Channel
from ..time import Time
from ..io.nds2 import Nds2ChannelType

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = ['StateTimeSeries', 'StateTimeSeriesDict',
           'StateVector', 'StateVectorDict', 'StateVectorList', 'Bits']


# -- utilities ----------------------------------------------------------------

def _bool_segments(array, start=0, delta=1, minlen=1):
    """Yield segments of consecutive `True` values in a boolean array

    Parameters
    ----------
    array : `iterable`
        An iterable of boolean-castable values.

    start : `float`
        The value of the first sample on the indexed axis
        (e.g.the GPS start time of the array).

    delta : `float`
        The step size on the indexed axis (e.g. sample duration).

    minlen : `int`, optional
        The minimum number of consecutive `True` values for a segment.

    Yields
    ------
    segment : `tuple`
        ``(start + i * delta, start + (i + n) * delta)`` for a sequence
        of ``n`` consecutive True values starting at position ``i``.

    Notes
    -----
    This method is adapted from original code written by Kipp Cannon and
    distributed under GPLv3.

    The datatype of the values returned will be the larger of the types
    of ``start`` and ``delta``.

    Examples
    --------
    >>> print(list(_bool_segments([0, 1, 0, 0, 0, 1, 1, 1, 0, 1]))
    [(1, 2), (5, 8), (9, 10)]
    >>> print(list(_bool_segments([0, 1, 0, 0, 0, 1, 1, 1, 0, 1]
    ...                           start=100., delta=0.1))
    [(100.1, 100.2), (100.5, 100.8), (100.9, 101.0)]
    """
    array = iter(array)
    i = 0
    while True:
        try:  # get next value
            val = next(array)
        except StopIteration:  # end of array
            return

        if val:  # start of new segment
            n = 1  # count consecutive True
            try:
                while next(array):  # run until segment will end
                    n += 1
            except StopIteration:  # have reached the end
                return  # stop
            finally:  # yield segment (including at StopIteration)
                if n >= minlen:  # ... if long enough
                    yield (start + i * delta, start + (i + n) * delta)
            i += n
        i += 1


# -- StateTimeSeries ----------------------------------------------------------

class StateTimeSeries(TimeSeriesBase):
    """Boolean array representing a good/bad state determination

    Parameters
    ----------
    value : array-like
        input data array

    t0 : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine

    dt : `float`, `~astropy.units.Quantity`, optional, default: `1`
        time between successive samples (seconds), can also be given inversely
        via `sample_rate`

    sample_rate : `float`, `~astropy.units.Quantity`, optional, default: `1`
        the rate of samples per second (Hertz), can also be given inversely
        via `dt`

    times : `array-like`
        the complete array of GPS times accompanying the data for this series.
        This argument takes precedence over `t0` and `dt` so should be given
        in place of these if relevant, not alongside

    name : `str`, optional
        descriptive title for this array

    channel : `~gwpy.detector.Channel`, `str`, optional
        source data stream for these data

    dtype : `~numpy.dtype`, optional
        input data type

    copy : `bool`, optional, default: `False`
        choose to copy the input data to new memory

    subok : `bool`, optional, default: `True`
        allow passing of sub-classes by the array generator

    Notes
    -----
    Key methods

    .. autosummary::

       ~StateTimeSeries.to_dqflag
    """

    def __new__(cls, data, t0=None, dt=None, sample_rate=None, times=None,
                channel=None, name=None, **kwargs):
        """Generate a new StateTimeSeries
        """
        if kwargs.pop('unit', None) is not None:
            raise TypeError("%s does not accept keyword argument 'unit'"
                            % cls.__name__)
        if isinstance(data, (list, tuple)):
            data = numpy.asarray(data)
        if not isinstance(data, cls):
            data = data.astype(bool)
        return super().__new__(
            cls, data, t0=t0, dt=dt, sample_rate=sample_rate, times=times,
            name=name, channel=channel, **kwargs)

    # -- unit handling (always dimensionless) ---

    @property
    def unit(self):
        return units.dimensionless_unscaled

    def override_unit(self, unit, parse_strict='raise'):
        return NotImplemented

    def _to_own_unit(self, value, check_precision=True):
        if isinstance(value, units.Quantity) and value.unit != self.unit:
            raise ValueError("Cannot store %s with units %r"
                             % (type(self).__name__, value.unit))
        if not isinstance(value, units.Quantity):
            value *= self.unit
        return value

    # -- math handling (always boolean) ---------

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if out.ndim:
            return out.view(bool)
        return out

    def diff(self, n=1, axis=-1):
        slice1 = (slice(1, None),)
        slice2 = (slice(None, -1),)
        new = (self.value[slice1] ^ self.value[slice2]).view(type(self))
        new.__metadata_finalize__(self)
        try:  # shift x0 to the right by one place
            new.x0 = self._xindex[1]
        except AttributeError:
            new.x0 = self.x0 + self.dx
        if n > 1:
            return new.diff(n-1, axis=axis)
        return new
    diff.__doc__ = TimeSeriesBase.diff.__doc__

    @wraps(numpy.ndarray.all, assigned=('__doc__',))
    def all(self, axis=None, out=None):
        return numpy.all(self.value, axis=axis, out=out)

    # -- useful methods -------------------------

    def to_dqflag(self, name=None, minlen=1, dtype=None, round=False,
                  label=None, description=None):
        """Convert this series into a `~gwpy.segments.DataQualityFlag`.

        Each contiguous set of `True` values are grouped as a
        `~gwpy.segments.Segment` running from the GPS time the first
        found `True`, to the GPS time of the next `False` (or the end
        of the series)

        Parameters
        ----------
        minlen : `int`, optional
            minimum number of consecutive `True` values to identify as a
            `~gwpy.segments.Segment`. This is useful to ignore single
            bit flips, for example.

        dtype : `type`, `callable`
            output segment entry type, can pass either a type for simple
            casting, or a callable function that accepts a float and returns
            another numeric type, defaults to the `dtype` of the time index

        round : `bool`, optional
            choose to round each `~gwpy.segments.Segment` to its
            inclusive integer boundaries

        label : `str`, optional
            the :attr:`~gwpy.segments.DataQualityFlag.label` for the
            output flag.

        description : `str`, optional
            the :attr:`~gwpy.segments.DataQualityFlag.description` for the
            output flag.

        Returns
        -------
        dqflag : `~gwpy.segments.DataQualityFlag`
            a segment representation of this `StateTimeSeries`, the span
            defines the `known` segments, while the contiguous `True`
            sets defined each of the `active` segments
        """
        from ..segments import DataQualityFlag

        # format dtype
        if dtype is None:
            dtype = self.t0.dtype
        if isinstance(dtype, numpy.dtype):  # use callable dtype
            dtype = dtype.type
        start = dtype(self.t0.value)
        dt = dtype(self.dt.value)

        # build segmentlists (can use simple objects since DQFlag converts)
        active = _bool_segments(self.value, start, dt, minlen=int(minlen))
        known = [tuple(map(dtype, self.span))]

        # build flag and return
        out = DataQualityFlag(name=name or self.name, active=active,
                              known=known, label=label or self.name,
                              description=description)
        if round:
            return out.round()
        return out

    def to_lal(self, *args, **kwargs):
        """Bogus function inherited from superclass, do not use.
        """
        raise NotImplementedError("The to_lal method, inherited from the "
                                  "TimeSeries, cannot be used with the "
                                  "StateTimeSeries because LAL has no "
                                  "BooleanTimeSeries structure")

    @classmethod
    @wraps(TimeSeriesBase.from_nds2_buffer)
    def from_nds2_buffer(cls, buffer, **metadata):
        metadata.setdefault('unit', None)
        return super().from_nds2_buffer(buffer, **metadata)

    def __getitem__(self, item):
        if isinstance(item, (float, int)):
            return numpy.ndarray.__getitem__(self, item)
        else:
            return super().__getitem__(item)

    def tolist(self):
        return self.value.tolist()

    tolist.__doc__ = numpy.ndarray.tolist.__doc__


# -- Bits ---------------------------------------------------------------------

class Bits(list):
    """Definition of the bits in a `StateVector`.

    Parameters
    ----------
    bits : `list`
        list of bit names

    channel : `Channel`, `str`, optional
        data channel associated with this Bits

    epoch : `float`, optional
        defining GPS epoch for this `Bits`

    description : `dict`, optional
        (bit, desc) `dict` of longer descriptions for each bit
    """
    def __init__(self, bits, channel=None, epoch=None, description=None):
        # handle dict of (index, bitname) pairs
        if isinstance(bits, dict):
            n = max(map(int, bits.keys())) + 1
            list.__init__(self, [None] * n)
            for key, val in bits.items():
                self[int(key)] = val
        # otherwise just parse a list of bitnames
        else:
            list.__init__(self, [b or None for b in bits])

        # populate metadata
        if channel is not None:
            self.channel = channel
        if epoch is not None:
            self.epoch = epoch
        self.description = description

        # rebuild descriptions
        for i, bit in enumerate(self):
            if bit is None or bit in self.description:
                continue
            elif channel:
                self.description[bit] = '%s bit %d' % (self.channel, i)
            else:
                self.description[bit] = None

    @property
    def epoch(self):
        """Starting GPS time epoch for these `Bits`.

        This attribute is recorded as a `~astropy.time.Time` object in the
        GPS format, allowing native conversion into other formats.

        See :mod:`~astropy.time` for details on the `Time` object.
        """
        try:
            return Time(self._epoch, format='gps')
        except AttributeError:
            return None

    @epoch.setter
    def epoch(self, epoch):
        if isinstance(epoch, Time):
            self._epoch = epoch.gps
        elif isinstance(epoch, units.Quantity):
            self._epoch = epoch.value
        else:
            self._epoch = float(epoch)

    @property
    def channel(self):
        """Data channel associated with these `Bits`.
        """
        try:
            return self._channel
        except AttributeError:
            return None

    @channel.setter
    def channel(self, chan):
        if isinstance(chan, Channel):
            self._channel = chan
        else:
            self._channel = Channel(chan)

    @property
    def description(self):
        """(key, value) dictionary of long bit descriptions.
        """
        return self._description

    @description.setter
    def description(self, desc):
        if desc is None:
            self._description = {}
        else:
            self._description = desc

    def __repr__(self):
        indent = " " * len('<%s(' % self.__class__.__name__)
        mask = ('\n%s' % indent).join(['%d: %r' % (idx, bit) for
                                       idx, bit in enumerate(self)
                                       if bit])
        return ("<{1}({2},\n{0}channel={3},\n{0}epoch={4})>".format(
            indent, self.__class__.__name__,
            mask, repr(self.channel), repr(self.epoch)))

    def __str__(self):
        indent = " " * len('%s(' % self.__class__.__name__)
        mask = ('\n%s' % indent).join(['%d: %s' % (idx, bit) for
                                       idx, bit in enumerate(self)
                                       if bit])
        return ("{1}({2},\n{0}channel={3},\n{0}epoch={4})".format(
            indent, self.__class__.__name__,
            mask, str(self.channel), str(self.epoch)))

    def __array__(self):
        return numpy.array([b or '' for b in self])


# -- StateVector --------------------------------------------------------------

class StateVector(TimeSeriesBase):
    """Binary array representing good/bad state determinations of some data.

    Each binary bit represents a single boolean condition, with the
    definitions of all the bits stored in the `StateVector.bits`
    attribute.

    Parameters
    ----------
    value : array-like
        input data array

    bits : `Bits`, `list`, optional
        list of bits defining this `StateVector`

    t0 : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine

    dt : `float`, `~astropy.units.Quantity`, optional, default: `1`
        time between successive samples (seconds), can also be given inversely
        via `sample_rate`

    sample_rate : `float`, `~astropy.units.Quantity`, optional, default: `1`
        the rate of samples per second (Hertz), can also be given inversely
        via `dt`

    times : `array-like`
        the complete array of GPS times accompanying the data for this series.
        This argument takes precedence over `t0` and `dt` so should be given
        in place of these if relevant, not alongside

    name : `str`, optional
        descriptive title for this array

    channel : `~gwpy.detector.Channel`, `str`, optional
        source data stream for these data

    dtype : `~numpy.dtype`, optional
        input data type

    copy : `bool`, optional, default: `False`
        choose to copy the input data to new memory

    subok : `bool`, optional, default: `True`
        allow passing of sub-classes by the array generator

    Notes
    -----
    Key methods:

    .. autosummary::

        ~StateVector.fetch
        ~StateVector.read
        ~StateVector.write
        ~StateVector.to_dqflags
        ~StateVector.plot

    """
    _metadata_slots = TimeSeriesBase._metadata_slots + ('bits',)
    _print_slots = TimeSeriesBase._print_slots + ('_bits',)

    def __new__(cls, data, bits=None, t0=None, dt=None, sample_rate=None,
                times=None, channel=None, name=None, **kwargs):
        """Generate a new `StateVector`.
        """
        new = super().__new__(cls, data, t0=t0, dt=dt,
                              sample_rate=sample_rate,
                              times=times, channel=channel,
                              name=name, **kwargs)
        new.bits = bits
        return new

    # -- StateVector properties -----------------

    # -- bits
    @property
    def bits(self):
        """list of `Bits` for this `StateVector`

        :type: `Bits`
        """
        try:
            return self._bits
        except AttributeError:
            if self.dtype.name.startswith(('uint', 'int')):
                nbits = self.itemsize * 8
                self.bits = Bits(['Bit %d' % b for b in range(nbits)],
                                 channel=self.channel, epoch=self.epoch)
                return self.bits
            elif hasattr(self.channel, 'bits'):
                self.bits = self.channel.bits
                return self.bits
            return None

    @bits.setter
    def bits(self, mask):
        if mask is None:
            del self.bits
            return
        if not isinstance(mask, Bits):
            mask = Bits(mask, channel=self.channel,
                        epoch=self.epoch)
        self._bits = mask

    @bits.deleter
    def bits(self):
        try:
            del self._bits
        except AttributeError:
            pass

    # -- boolean
    @property
    def boolean(self):
        """A mapping of this `StateVector` to a 2-D array containing all
        binary bits as booleans, for each time point.
        """
        try:
            return self._boolean
        except AttributeError:
            nbits = len(self.bits)
            boolean = numpy.zeros((self.size, nbits), dtype=bool)
            for i, sample in enumerate(self.value):
                boolean[i, :] = [int(sample) >> j & 1 for j in range(nbits)]
            self._boolean = Array2D(boolean, name=self.name,
                                    x0=self.x0, dx=self.dx, y0=0, dy=1)
            return self.boolean

    # -- data type handling ---------------------

    def _to_own_unit(self, value, check_precision=True):
        if isinstance(value, units.Quantity) and value.unit != self.unit:
            raise ValueError("Cannot store %s with units %r"
                             % (type(self).__name__, value.unit))
        if not isinstance(value, units.Quantity):
            return value * self.unit
        return value

    # -- StateVector methods --------------------

    def get_bit_series(self, bits=None):
        """Get the `StateTimeSeries` for each bit of this `StateVector`.

        Parameters
        ----------
        bits : `list`, optional
            a list of bit indices or bit names, defaults to all bits

        Returns
        -------
        bitseries : `StateTimeSeriesDict`
            a `dict` of `StateTimeSeries`, one for each given bit
        """
        if bits is None:
            bits = [b for b in self.bits if b not in {None, ''}]
        bindex = []
        for bit in bits:
            try:
                bindex.append((self.bits.index(bit), bit))
            except (IndexError, ValueError) as exc:
                exc.args = ('Bit %r not found in StateVector' % bit,)
                raise
        self._bitseries = StateTimeSeriesDict()
        for i, bit in bindex:
            self._bitseries[bit] = StateTimeSeries(
                self.value >> i & 1, name=bit, epoch=self.x0.value,
                channel=self.channel, sample_rate=self.sample_rate)
        return self._bitseries

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read data into a `StateVector`

        Parameters
        ----------
        source : `str`, `list`
            Source of data, any of the following:

            - `str` path of single data file,
            - `str` path of LAL-format cache file,
            - `list` of paths.

        channel : `str`, `~gwpy.detector.Channel`
            the name of the channel to read, or a `Channel` object.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        bits : `list`, optional
            list of bits names for this `StateVector`, give `None` at
            any point in the list to mask that bit

        format : `str`, optional
            source format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats.

        nproc : `int`, optional, default: `1`
            number of parallel processes to use, serial process by
            default.

        gap : `str`, optional
            how to handle gaps in the cache, one of

            - 'ignore': do nothing, let the underlying reader method handle it
            - 'warn': do nothing except print a warning to the screen
            - 'raise': raise an exception upon finding a gap (default)
            - 'pad': insert a value to fill the gaps

        pad : `float`, optional
            value with which to fill gaps in the source data, only used if
            gap is not given, or `gap='pad'` is given

        Raises
        ------
        IndexError
            if ``source`` is an empty list

        Examples
        --------
        To read the S6 state vector, with names for all the bits::

            >>> sv = StateVector.read(
                'H-H1_LDAS_C02_L2-968654592-128.gwf', 'H1:IFO-SV_STATE_VECTOR',
                bits=['Science mode', 'Conlog OK', 'Locked',
                      'No injections', 'No Excitations'],
                dtype='uint32')

        then you can convert these to segments

            >>> segments = sv.to_dqflags()

        or to read just the interferometer operations bits::

            >>> sv = StateVector.read(
                'H-H1_LDAS_C02_L2-968654592-128.gwf', 'H1:IFO-SV_STATE_VECTOR',
                bits=['Science mode', None, 'Locked'], dtype='uint32')

        Running `to_dqflags` on this example would only give 2 flags, rather
        than all five.

        Alternatively the `bits` attribute can be reset after reading, but
        before any further operations.

        Notes
        -----"""
        return super().read(source, *args, **kwargs)

    def to_dqflags(self, bits=None, minlen=1, dtype=float, round=False):
        """Convert this `StateVector` into a `~gwpy.segments.DataQualityDict`

        The `StateTimeSeries` for each bit is converted into a
        `~gwpy.segments.DataQualityFlag` with the bits combined into a dict.

        Parameters
        ----------
        minlen : `int`, optional, default: 1
           minimum number of consecutive `True` values to identify as a
           `Segment`. This is useful to ignore single bit flips,
           for example.

        bits : `list`, optional
            a list of bit indices or bit names to select, defaults to
            `~StateVector.bits`

        Returns
        -------
        DataQualityFlag list : `list`
            a list of `~gwpy.segments.flag.DataQualityFlag`
            reprensentations for each bit in this `StateVector`

        See also
        --------
        StateTimeSeries.to_dqflag
            for details on the segment representation method for
            `StateVector` bits
        """
        from ..segments import DataQualityDict
        out = DataQualityDict()
        bitseries = self.get_bit_series(bits=bits)
        for bit, sts in bitseries.items():
            out[bit] = sts.to_dqflag(name=bit, minlen=minlen, round=round,
                                     dtype=dtype,
                                     description=self.bits.description[bit])
        return out

    @classmethod
    def fetch(cls, channel, start, end, bits=None, host=None, port=None,
              verbose=False, connection=None, type=Nds2ChannelType.any()):
        """Fetch data from NDS into a `StateVector`.

        Parameters
        ----------
        channel : `str`, `~gwpy.detector.Channel`
            the name of the channel to read, or a `Channel` object.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS end time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        bits : `Bits`, `list`, optional
            definition of bits for this `StateVector`

        host : `str`, optional
            URL of NDS server to use, defaults to observatory site host

        port : `int`, optional
            port number for NDS server query, must be given with `host`

        verify : `bool`, optional, default: `True`
            check channels exist in database before asking for data

        connection : `nds2.connection`
            open NDS connection to use

        verbose : `bool`, optional
            print verbose output about NDS progress

        type : `int`, optional
            NDS2 channel type integer

        dtype : `type`, `numpy.dtype`, `str`, optional
            identifier for desired output data type
        """
        new = cls.DictClass.fetch(
            [channel], start, end, host=host, port=port,
            verbose=verbose, connection=connection)[channel]
        if bits:
            new.bits = bits
        return new

    @classmethod
    def get(cls, channel, start, end, bits=None, **kwargs):
        """Get data for this channel from frames or NDS

        Parameters
        ----------
        channel : `str`, `~gwpy.detector.Channel`
            the name of the channel to read, or a `Channel` object.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS end time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        bits : `Bits`, `list`, optional
            definition of bits for this `StateVector`

        pad : `float`, optional
            value with which to fill gaps in the source data, only used if
            gap is not given, or ``gap='pad'`` is given

        dtype : `numpy.dtype`, `str`, `type`, or `dict`
            numeric data type for returned data, e.g. `numpy.float`, or
            `dict` of (`channel`, `dtype`) pairs

        nproc : `int`, optional, default: `1`
            number of parallel processes to use, serial process by
            default.

        verbose : `bool`, optional
            print verbose output about NDS progress.

        **kwargs
            other keyword arguments to pass to either
            :meth:`.find` (for direct GWF file access) or
            :meth:`.fetch` for remote NDS2 access

        See also
        --------
        StateVector.fetch
            for grabbing data from a remote NDS2 server
        StateVector.find
            for discovering and reading data from local GWF files
        """
        new = cls.DictClass.get([channel], start, end, **kwargs)[channel]
        if bits:
            new.bits = bits
        return new

    def plot(self, format='segments', bits=None, **kwargs):
        """Plot the data for this `StateVector`

        Parameters
        ----------
        format : `str`, optional, default: ``'segments'``
            The type of plot to make, either 'segments' to plot the
            SegmentList for each bit, or 'timeseries' to plot the raw
            data for this `StateVector`

        bits : `list`, optional
            A list of bit indices or bit names, defaults to
            `~StateVector.bits`. This argument is ignored if ``format`` is
            not ``'segments'``

        **kwargs
            Other keyword arguments to be passed to either
            `~gwpy.plot.SegmentAxes.plot` or
            `~gwpy.plot.Axes.plot`, depending
            on ``format``.

        Returns
        -------
        plot : `~gwpy.plot.Plot`
            output plot object

        See also
        --------
        matplotlib.pyplot.figure
            for documentation of keyword arguments used to create the
            figure
        matplotlib.figure.Figure.add_subplot
            for documentation of keyword arguments used to create the
            axes
        gwpy.plot.SegmentAxes.plot_flag
            for documentation of keyword arguments used in rendering each
            statevector flag.
        """
        if format == 'timeseries':
            return super().plot(**kwargs)
        if format == 'segments':
            from ..plot import Plot
            kwargs.setdefault('xscale', 'auto-gps')
            return Plot(*self.to_dqflags(bits=bits).values(),
                        projection='segments', **kwargs)
        raise ValueError("'format' argument must be one of: 'timeseries' or "
                         "'segments'")

    def resample(self, rate):
        """Resample this `StateVector` to a new rate

        Because of the nature of a state-vector, downsampling is done
        by taking the logical 'and' of all original samples in each new
        sampling interval, while upsampling is achieved by repeating
        samples.

        Parameters
        ----------
        rate : `float`
            rate to which to resample this `StateVector`, must be a
            divisor of the original sample rate (when downsampling)
            or a multiple of the original (when upsampling).

        Returns
        -------
        vector : `StateVector`
            resampled version of the input `StateVector`
        """
        rate1 = self.sample_rate.value
        if isinstance(rate, units.Quantity):
            rate2 = rate.value
        else:
            rate2 = float(rate)
        # upsample
        if (rate2 / rate1).is_integer():
            raise NotImplementedError("StateVector upsampling has not "
                                      "been implemented yet, sorry.")
        # downsample
        elif (rate1 / rate2).is_integer():
            factor = int(rate1 / rate2)
            # reshape incoming data to one column per new sample
            newsize = int(self.size / factor)
            old = self.value.reshape((newsize, self.size // newsize))
            # work out number of bits
            if self.bits:
                nbits = len(self.bits)
            else:
                max_ = self.value.max()
                nbits = int(ceil(log(max_, 2))) if max_ else 1
            bits = range(nbits)
            # construct an iterator over the columns of the old array
            itr = numpy.nditer(
                [old, None],
                flags=['external_loop', 'reduce_ok'],
                op_axes=[None, [0, -1]],
                op_flags=[['readonly'], ['readwrite', 'allocate']])
            dtype = self.dtype
            type_ = self.dtype.type
            # for each new sample, each bit is logical AND of old samples
            # bit is ON,
            for x, y in itr:
                y[...] = numpy.sum([type_((x >> bit & 1).all() * (2 ** bit))
                                    for bit in bits], dtype=self.dtype)
            new = StateVector(itr.operands[1], dtype=dtype)
            new.__metadata_finalize__(self)
            new._unit = self.unit
            new.sample_rate = rate2
            return new
        # error for non-integer resampling factors
        elif rate1 < rate2:
            raise ValueError("New sample rate must be multiple of input "
                             "series rate if upsampling a StateVector")
        else:
            raise ValueError("New sample rate must be divisor of input "
                             "series rate if downsampling a StateVector")


@as_series_dict_class(StateTimeSeries)
class StateTimeSeriesDict(TimeSeriesBaseDict):
    __doc__ = TimeSeriesBaseDict.__doc__.replace('TimeSeriesBase',
                                                 'StateTimeSeries')
    EntryClass = StateTimeSeries


@as_series_dict_class(StateVector)
class StateVectorDict(TimeSeriesBaseDict):
    __doc__ = TimeSeriesBaseDict.__doc__.replace('TimeSeriesBase',
                                                 'StateVector')
    EntryClass = StateVector

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read data for multiple bit vector channels into a `StateVectorDict`

        Parameters
        ----------
        source : `str`, `list`
            Source of data, any of the following:

            - `str` path of single data file,
            - `str` path of LAL-format cache file,
            - `list` of paths.

        channels : `~gwpy.detector.channel.ChannelList`, `list`
            a list of channels to read from the source.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str` optional
            GPS start time of required data, anything parseable by
            :meth:`~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data, anything parseable by
            :meth:`~gwpy.time.to_gps` is fine

        bits : `list` of `lists`, `dict`, optional
            the ordered list of interesting bit lists for each channel,
            or a `dict` of (`channel`, `list`) pairs

        format : `str`, optional
            source format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats.

        nproc : `int`, optional, default: ``1``
            number of parallel processes to use, serial process by
            default.

        gap : `str`, optional
            how to handle gaps in the cache, one of

            - 'ignore': do nothing, let the underlying reader method handle it
            - 'warn': do nothing except print a warning to the screen
            - 'raise': raise an exception upon finding a gap (default)
            - 'pad': insert a value to fill the gaps

        pad : `float`, optional
            value with which to fill gaps in the source data, only used if
            gap is not given, or `gap='pad'` is given

        Returns
        -------
        statevectordict : `StateVectorDict`
            a `StateVectorDict` of (`channel`, `StateVector`) pairs. The keys
            are guaranteed to be the ordered list `channels` as given.

        Notes
        -----"""
        return super().read(source, *args, **kwargs)


class StateVectorList(TimeSeriesBaseList):
    __doc__ = TimeSeriesBaseList.__doc__.replace('TimeSeriesBase',
                                                 'StateVector')
    EntryClass = StateVector
