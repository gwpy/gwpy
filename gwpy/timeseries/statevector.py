# -*- coding: utf-8 -*-
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

"""This module defines the Boolean array representing the state of some data

Such states are typically the comparison of a `TimeSeries` against some
threshold, where sub-threshold is good and sup-threshold is bad,
for example.

Single `StateTimeSeries` can be bundled together to form `StateVector`
arrays, representing a bit mask of states that combine to make a detailed
statement of instrumental operation
"""

from math import (ceil, log)
import sys

import numpy

from glue.segmentsUtils import from_bitstream
from astropy.units import Quantity

from .core import (TimeSeriesBase, TimeSeriesBaseDict, TimeSeriesBaseList,
                   NDS2_FETCH_TYPE_MASK, as_series_dict_class)
from ..data import Array2D
from ..detector import Channel
from ..time import Time
from ..io import (reader, writer)
from ..utils.docstring import interpolate_docstring

if sys.version_info[0] < 3:
    range = xrange

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = ['StateTimeSeries',
           'StateVector', 'StateVectorDict', 'StateVectorList', 'Bits']


@interpolate_docstring
class StateTimeSeries(TimeSeriesBase):
    """Boolean array representing a good/bad state determination
    of some data.

    Parameters
    ----------
    %(Array1)s

    %(time-axis)s

    %(Array2)s

    Notes
    -----
    The input data array is cast to the `bool` data type upon creation of
    this series.

    .. rubric:: Key methods

       ~StateTimeSeries.to_dqflag

    """

    def __new__(cls, data, times=None, epoch=None, channel=None,
                unit='dimensionless', sample_rate=None, name=None, **kwargs):
        """Generate a new StateTimeSeries
        """
        if isinstance(data, (list, tuple)):
            data = numpy.asarray(data)
        if not isinstance(data, cls):
            data = data.astype(bool)
        return super(StateTimeSeries, cls).__new__(cls, data, name=name,
                                                   epoch=epoch,
                                                   channel=channel,
                                                   sample_rate=sample_rate,
                                                   times=times, **kwargs)

    def to_dqflag(self, name=None, minlen=1, dtype=float, round=False,
                  label=None, description=None):
        """Convert this `StateTimeSeries` into a
        `~gwpy.segments.DataQualityFlag`

        Each contiguous set of `True` values are grouped as a
        `~gwpy.segments.Segment` running from the GPS time the first
        found `True`, to the GPS time of the next `False` (or the end
        of the series)

        Parameters
        ----------
        minlen : `int`, optional, default: 1
            minimum number of consecutive `True` values to identify as a
            `~gwpy.segments.Segment`. This is useful to ignore single
            bit flips, for example.

        dtype : `type`, `callable`, default: `float`
            output segment entry type, can pass either a type for simple
            casting, or a callable function that accepts a float and returns
            another numeric type

        round : `bool`, optional, default: False
            choose to round each `~gwpy.segments.Segment` to its
            inclusive integer boundaries

        Returns
        -------
        dqflag : `~gwpy.segments.DataQualityFlag`
            a segment representation of this `StateTimeSeries`, the span
            defines the `known` segments, while the contiguous `True`
            sets defined each of the `active` segments
        """
        from ..segments import (Segment, SegmentList, DataQualityFlag)
        start = self.x0.value
        dt = self.dx.value
        active = from_bitstream(self.value, start, dt, minlen=int(minlen))
        if dtype is not float:
            active = active.__class__([Segment(dtype(s[0]), dtype(s[1])) for
                                       s in active])
        known = SegmentList([self.span])
        out = DataQualityFlag(name=name or self.name, active=active,
                              known=known, label=label or self.name,
                              description=description)
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

    def __getitem__(self, item):
        if isinstance(item, (float, int)):
            return numpy.ndarray.__getitem__(self, item)
        else:
            return super(StateTimeSeries, self).__getitem__(item)


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
        list.__init__(self, [b or None for b in bits])
        if channel is not None:
            self.channel = channel
        if epoch is not None:
            self.epoch = epoch
        self.description = description
        for i, bit in enumerate(bits):
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
        elif isinstance(epoch, Quantity):
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
    def channel(self, ch):
        if isinstance(ch, Channel):
            self._channel = ch
        else:
            self._channel = Channel(ch)

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


@interpolate_docstring
class StateVector(TimeSeriesBase):
    """Binary array representing good/bad state determinations of some data.

    Each binary bit represents a single boolean condition, with the
    definitions of all the bits stored in the `StateVector.bits`
    attribute.

    Parameters
    ----------
    %(Array1)s

    bits : `Bits`, `list`, optional
        list of bits defining this `StateVector`

    %(time-axis)s

    %(Array2)s

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
    _metadata_slots = TimeSeriesBase._metadata_slots + ['bits']

    def __new__(cls, data, bits=None, times=None, epoch=None, sample_rate=None,
                channel=None, name=None, **kwargs):
        """Generate a new `StateVector`.
        """
        new = super(StateVector, cls).__new__(cls, data, name=name,
                                              epoch=epoch, channel=channel,
                                              sample_rate=sample_rate,
                                              times=times,
                                              **kwargs)
        new.bits = bits
        return new

    # -------------------------------------------
    # StateVector properties

    @property
    def bits(self):
        """The list of bit names for this `StateVector`.

        :type: `Bits`
        """
        try:
            return self._bits
        except AttributeError as e:
            if self.dtype.name.startswith(('uint', 'int')):
                nbits = self.itemsize * 8
                self.bits = Bits(['Bit %d' % b for b in range(nbits)],
                                 channel=self.channel, epoch=self.epoch)
                return self.bits
            elif hasattr(self.channel, 'bits'):
                self.bits = self.channel.bits
                return self.bits
            else:
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
            for i, d in enumerate(self.value):
                boolean[i, :] = [int(d) >> j & 1 for
                                 j in range(nbits)]
            self._boolean = Array2D(boolean, name=self.name,
                                    x0=self.x0, dx=self.dx, y0=0, dy=1)
            return self.boolean

    def get_bit_series(self, bits=None):
        """Get the `StateTimeSeries` for each bit of this `StateVector`.

        Parameters
        ----------
        bits : `list`, optional
            a list of bit indices or bit names, defaults to
            `~StateVector.bits`

        Returns
        -------
        bitseries : `TimeSeriesDict`
            a `TimeSeriesDict` of `StateTimeSeries`, one for each given
            bit
        """
        if bits is None:
            bits = [b for b in self.bits if b is not None and b is not '']
        bindex = []
        for b in bits:
            try:
                bindex.append((self.bits.index(b), b))
            except IndexError as e:
                e.args = ('Bit %r not found in StateVector' % b)
                raise e
        self._bitseries = StateTimeSeriesDict()
        for i, bit in bindex:
            self._bitseries[bit] = StateTimeSeries(
                self.value >> i & 1, name=bit, epoch=self.x0.value,
                channel=self.channel, sample_rate=self.sample_rate)
        return self._bitseries

    # -------------------------------------------
    # StateVector methods

    # use input/output registry to allow multi-format reading
    read = classmethod(interpolate_docstring(
        reader(doc="""Read data into a `StateVector`

        Parameters
        ----------
        %(timeseries-read1)s

        bits : `list`, optional
            list of bits names for this `StateVector`, give `None` at
            any point in the list to mask that bit

        %(timeseries-read2)s

        Example
        -------
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
        -----""")))

    write = writer(
        doc="""Write this `StateVector` to a file

        Parameters
        ----------
        outfile : `str`
            path of output file

        Notes
        -----
        """)

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
            a list of :class:`~gwpy.segments.flag.DataQualityFlag`
            reprensentations for each bit in this `StateVector`

        See Also
        --------
        :meth:`StateTimeSeries.to_dqflag`
            for details on the segment representation method for
            `StateVector` bits
        """
        from ..segments import DataQualityDict
        out = DataQualityDict()
        bitseries = self.get_bit_series(bits=bits)
        for bit, sts in bitseries.iteritems():
            out[bit] = sts.to_dqflag(name=bit, minlen=minlen, round=round,
                                     dtype=dtype,
                                     description=self.bits.description[bit])
        return out

    @classmethod
    @interpolate_docstring
    def fetch(cls, channel, start, end, bits=None, host=None, port=None,
              verbose=False, connection=None, type=NDS2_FETCH_TYPE_MASK):
        """Fetch data from NDS into a `StateVector`.

        Parameters
        ----------
        %(timeseries-fetch1)s

        bits : `Bits`, `list`, optional
            definition of bits for this `StateVector`

        %(timeseries-fetch2)s
        """
        new = cls.DictClass.fetch(
            [channel], start, end, host=host, port=port,
            verbose=verbose, connection=connection)[channel]
        if bits:
            new.bits = bits
        return new

    @classmethod
    def fetch_open_data(cls, ifo, start, end, name='quality/simple',
                        host='https://losc.ligo.org'):
        """Fetch open-access data from the LIGO Open Science Center

        Parameters
        ----------
        ifo : `str`
            the two-character prefix of the IFO in which you are interested,
            e.g. `'L1'`

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS start time of required data, defaults to start of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        name : `str`, optional
            the full name of HDF5 dataset that represents the data you want,
            e.g. `'strain/Strain'` for _h(t)_ data, or `'quality/simple'`
            for basic data-quality information

        host : `str`, optional
            HTTP host name of LOSC server to access
        """
        from .io.losc import fetch_losc_data
        return fetch_losc_data(ifo, start, end, channel=name, cls=cls,
                               host=host)

    @classmethod
    @interpolate_docstring
    def get(cls, channel, start, end, bits=None, **kwargs):
        """Get data for this channel from frames or NDS

        Parameters
        ----------
        %(timeseries-fetch1)s

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

        **kwargs            other keyword arguments to pass to either
            :meth:`.find` (for direct GWF file access) or
            :meth:`.fetch` for remote NDS2 access

        See Also
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

    def to_lal(self, *args, **kwargs):
        """Bogus function inherited from superclass, do not use.
        """
        raise NotImplementedError("The to_lal method, inherited from the "
                                  "TimeSeries, cannot be used with the "
                                  "StateTimeSeries because LAL has no "
                                  "BooleanTimeSeries structure")

    def plot(self, format='segments', bits=None, **kwargs):
        """Plot the data for this `StateVector`

        Parameters
        ----------
        format : `str`, optional, default: ``'segments'``
            type of plot to make, either 'segments' to plot the
            SegmentList for each bit, or 'timeseries' to plot the raw
            data for this `StateVector`
        bits : `list`, optional
            a list of bit indices or bit names, defaults to
            `~StateVector.bits`. This argument is ignored if ``format`` is
            not ``'segments'``
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
            kwargs.setdefault('facecolor', 'green')
            kwargs.setdefault('edgecolor', 'black')
            kwargs.setdefault('known', {'facecolor': 'red',
                                        'edgecolor': 'black'})
            from ..plotter import SegmentPlot
            return SegmentPlot(*self.to_dqflags(bits=bits).values(), **kwargs)
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
        vector : :class:`~gwpy.timeseries.statevector.StateVector`
            resampled version of the input `StateVector`
        """
        rate1 = self.sample_rate.value
        if isinstance(rate, Quantity):
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
            newsize = self.size / factor
            old = self.value.reshape((newsize, self.size // newsize))
            # work out number of bits
            if self.bits is not None and len(self.bits):
                nbits = len(self.bits)
            else:
                max = self.value.max()
                nbits = max != 0 and int(ceil(log(self.value.max(), 2))) or 1
            bits = range(nbits)
            # construct an iterator over the columns of the old array
            it = numpy.nditer([old, None],
                              flags=['external_loop', 'reduce_ok'],
                              op_axes=[None, [0, -1]],
                              op_flags=[['readonly'],
                                        ['readwrite', 'allocate']])
            dtype = self.dtype
            type_ = self.dtype.type
            # for each new sample, each bit is logical AND of old samples
            # bit is ON,
            for x, y in it:
                y[...] = numpy.sum([type_((x >> bit & 1).all() * (2 ** bit))
                                    for bit in bits], dtype=self.dtype)
            new = StateVector(it.operands[1], dtype=dtype)
            new.__dict__ = self.copy_metadata()
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
    EntryClass = StateTimeSeries
    read = classmethod(reader(doc=TimeSeriesBaseDict.read.__doc__))


@as_series_dict_class(StateVector)
class StateVectorDict(TimeSeriesBaseDict):
    EntryClass = StateVector
    read = classmethod(reader(doc="""
        Read data for multiple bit vector channels into a `StateVectorDict`

        Parameters
        ----------
        source : `str`, `~glue.lal.Cache`
            a single file path `str`, or a `~glue.lal.Cache` containing
            a contiguous list of files.

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

            .. note::

               Parallel frame reading, via the ``nproc`` keyword argument,
               is only available when giving a :class:`~glue.lal.Cache` of
               frames, or using the ``format='cache'`` keyword argument.

        gap : `str`, optional
            how to handle gaps in the cache, one of

            - 'ignore': do nothing, let the undelying reader method handle it
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
        -----
        """))


class StateVectorList(TimeSeriesBaseList):
    EntryClass = StateVector
