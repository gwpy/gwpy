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

"""Array with metadata
"""

from __future__ import (division, print_function)

import os
import sys
import warnings
import re
from math import ceil

import numpy

from astropy import units

try:
    import nds2
except ImportError:
    NDS2_FETCH_TYPE_MASK = None
else:
    NDS2_FETCH_TYPE_MASK = (nds2.channel.CHANNEL_TYPE_RAW |
                            nds2.channel.CHANNEL_TYPE_RDS |
                            nds2.channel.CHANNEL_TYPE_TEST_POINT |
                            nds2.channel.CHANNEL_TYPE_STATIC)

from ..data import (Array2D, Series)
from ..detector import (Channel, ChannelList)
from ..io import (reader, writer, datafind)
from ..time import (Time, to_gps)
from ..utils import (gprint, with_import)
from ..utils.docstring import interpolate_docstring
from ..utils.compat import OrderedDict

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

__all__ = ['TimeSeriesBase', 'ArrayTimeSeries', 'TimeSeriesBaseDict']

_UFUNC_STRING = {'less': '<',
                 'less_equal': '<=',
                 'equal': '==',
                 'greater_equal': '>=',
                 'greater': '>',
                 }

interpolate_docstring.update({
    'time-axis': (
        """sample_rate : `float`, `~astropy.units.Quantity`, optional, default: `1`
        the rate of samples per second (Hertz)
    times : `array-like`
        the complete array of GPS times accompanying the data for this series.
        This argument takes precedence over `epoch` and `sample_rate` so should
        be given in place of these if relevant, not alongside"""),
})

@interpolate_docstring
class TimeSeriesBase(Series):
    """An `Array` with time-domain metadata.
    """
    _default_xunit = units.second
    _metadata_slots = ['name', 'channel', 'x0', 'dx']
    DictClass = None

    def __new__(cls, data, unit=None, times=None, epoch=None, channel=None,
                sample_rate=None, name=None, **kwargs):
        """Generate a new `TimeSeriesBase`.
        """
        # parse Channel input
        if isinstance(channel, Channel):
            name = name or channel.name
            unit = unit or channel.unit
            if sample_rate is None and times is None and 'dx' not in kwargs:
                sample_rate = channel.sample_rate
        if times is None and 'xindex' in kwargs:
            times = kwargs.pop('xindex')
        if sample_rate is None and times is None and 'dx' not in kwargs:
            sample_rate = 1
        if epoch is None and times is None and 'x0' not in kwargs:
            epoch = 0
        # generate TimeSeries
        new = super(TimeSeriesBase, cls).__new__(cls, data, name=name,
                                                 unit=unit, xindex=times,
                                                 channel=channel, **kwargs)
        if epoch is not None:
            new.epoch = epoch
        if sample_rate is not None:
            new.sample_rate = sample_rate
        return new

    # -------------------------------------------
    # TimeSeries properties

    dt = Series.dx
    span = Series.xspan

    @property
    def epoch(self):
        """GPS epoch for these data.

        This attribute is stored internally by the `x0` attribute

        :type: `~astropy.time.Time`
        """
        if self.x0 is None:
            return None
        else:
            return Time(self.x0, format='gps', scale='utc')

    @epoch.setter
    def epoch(self, epoch):
        if epoch is None:
            self.x0 = None
        elif isinstance(epoch, Time):
            self.x0 = epoch.gps
        else:
            try:
                self.x0 = to_gps(epoch)
            except TypeError:
                self.x0 = epoch

    @property
    def sample_rate(self):
        """Data rate for this `TimeSeries` in samples per second (Hertz).

        This attribute is stored internally by the `dx` attribute

        :type: `~astropy.units.Quantity` scalar
        """
        return (1 / self.dx).to('Hertz')

    @sample_rate.setter
    def sample_rate(self, val):
        if val is None:
            del self.dx
            return
        self.dx = (1 / units.Quantity(val, units.Hertz)).to(self.xunit)
        if numpy.isclose(self.dx.value, round(self.dx.value)):
            self.dx = units.Quantity(round(self.dx.value), self.dx.unit)

    @property
    def duration(self):
        """Duration of this series in seconds
        """
        return units.Quantity(self.span[1] - self.span[0], self.xunit)

    times = property(fget=Series.xindex.__get__,
                     fset=Series.xindex.__set__,
                     fdel=Series.xindex.__delete__,
                     doc="""Series of GPS times for each sample""")

    # -------------------------------------------
    # TimeSeries accessors

    interpolate_docstring.update({
        'timeseries-read1': """source : `str`, `~glue.lal.Cache`
            source of data, any of the following:

            - `str` path of single data file
            - `str` path of LAL-format cache file
            - `~glue.lal.Cache` describing one or more data files,

        channel : `str`, `~gwpy.detector.Channel`
            the name of the channel to read, or a `Channel` object.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS start time of required data, defaults to start of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine""",

        'timeseries-read2': """format : `str`, optional
            source format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats.

        nproc : `int`, optional, default: `1`
            number of parallel processes to use, serial process by
            default.

            .. note::

               Parallel frame reading, via the ``nproc`` keyword argument,
               is only available when giving a `~glue.lal.Cache` of
               frames, or using the ``format='cache'`` keyword argument.

        gap : `str`, optional
            how to handle gaps in the cache, one of

            - 'ignore': do nothing, let the undelying reader method handle it
            - 'warn': do nothing except print a warning to the screen
            - 'raise': raise an exception upon finding a gap (default)
            - 'pad': insert a value to fill the gaps

        pad : `float`, optional
            value with which to fill gaps in the source data, only used if
            gap is not given, or `gap='pad'` is given""",

        'timeseries-fetch1': """channel : `str`, `~gwpy.detector.Channel`
            the name of the channel to read, or a `Channel` object.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS end time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine""",

        'timeseries-fetch2': """host : `str`, optional
            URL of NDS server to use, defaults to observatory site host

        port : `int`, optional
            port number for NDS server query, must be given with `host`

        verify : `bool`, optional, default: `True`
            check channels exist in database before asking for data

        connection : :class:`~gwpy.io.nds.NDS2Connection`
            open NDS connection to use

        verbose : `bool`, optional
            print verbose output about NDS progress

        type : `int`, optional
            NDS2 channel type integer

        dtype : `type`, `numpy.dtype`, `str`, optional
            identifier for desired output data type""",
    })

    @classmethod
    @interpolate_docstring
    @with_import('nds2')
    def fetch(cls, channel, start, end, host=None, port=None, verbose=False,
              connection=None, verify=False, pad=None,
              type=NDS2_FETCH_TYPE_MASK, dtype=None):
        """Fetch data from NDS into a `TimeSeries`.

        Parameters
        ----------
        %(timeseries-fetch1)s

        %(timeseries-fetch2)s
        """
        return cls.DictClass.fetch(
            [channel], start, end, host=host, port=port,
            verbose=verbose, connection=connection, verify=verify,
            pad=pad, type=type, dtype=dtype)[str(channel)]

    @classmethod
    def fetch_open_data(cls, ifo, start, end, name='strain/Strain',
                        sample_rate=4096, host='https://losc.ligo.org'):
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

        sample_rate : `float`, optional, default: `4096`
            the sample rate of desired data. Most data are stored
            by LOSC at 4096 Hz, however there may be event-related
            data releases with a 16384 Hz rate

        host : `str`, optional
            HTTP host name of LOSC server to access
        """
        from .io.losc import fetch_losc_data
        return fetch_losc_data(ifo, start, end, channel=name, cls=cls,
                               sample_rate=sample_rate, host=host)

    @classmethod
    @interpolate_docstring
    def find(cls, channel, start, end, frametype=None,
             pad=None, dtype=None, nproc=1, verbose=False, **readargs):
        """Find and read data from frames for a channel

        Parameters
        ----------
        %(timeseries-fetch1)s

        frametype : `str`, optional
            name of frametype in which this channel is stored, will search
            for containing frame types if necessary

        pad : `float`, optional
            value with which to fill gaps in the source data, only used if
            gap is not given, or `gap='pad'` is given

        nproc : `int`, optional, default: `1`
            number of parallel processes to use, serial process by
            default.

        dtype : `numpy.dtype`, `str`, `type`, or `dict`
            numeric data type for returned data, e.g. `numpy.float`, or
            `dict` of (`channel`, `dtype`) pairs

        verbose : `bool`, optional
            print verbose output about NDS progress.

        **readargs
            any other keyword arguments to be passed to `.read()`
        """
        return cls.DictClass.find(
            [channel], start, end, frametype=frametype, verbose=verbose,
            pad=pad, dtype=dtype, nproc=nproc, **readargs)[str(channel)]

    @classmethod
    @interpolate_docstring
    def get(cls, channel, start, end, pad=None, dtype=None, verbose=False,
            **kwargs):
        """Get data for this channel from frames or NDS

        This method dynamically accesses either frames on disk, or a
        remote NDS2 server to find and return data for the given interval

        Parameters
        ----------
        %(timeseries-fetch1)s

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
        TimeSeries.fetch
            for grabbing data from a remote NDS2 server
        TimeSeries.find
            for discovering and reading data from local GWF files
        """
        return cls.DictClass.get(
            [channel], start, end, pad=pad, dtype=dtype, verbose=verbose,
            **kwargs)[str(channel)]

    # -------------------------------------------
    # Utilities

    def plot(self, **kwargs):
        """Plot the data for this `TimeSeries`
        """
        from ..plotter import TimeSeriesPlot
        return TimeSeriesPlot(self, **kwargs)

    @classmethod
    @with_import('nds2')
    def from_nds2_buffer(cls, buffer_, **metadata):
        """Construct a new `TimeSeries` from an `nds2.buffer` object

        Parameters
        ----------
        buffer_ : `nds2.buffer`
            the input NDS2-client buffer to read
        **metadata
            any other metadata keyword arguments to pass to the `TimeSeries`
            constructor

        Returns
        -------
        timeseries : `TimeSeries`
            a new `TimeSeries` containing the data from the `nds2.buffer`,
            and the appropriate metadata

        Notes
        -----
        This classmethod requires the nds2-client package
        """
        # cast as TimeSeries and return
        epoch = Time(buffer_.gps_seconds, buffer_.gps_nanoseconds,
                     format='gps')
        channel = Channel.from_nds2(buffer_.channel)
        return cls(buffer_.data, epoch=epoch, channel=channel, **metadata)

    @classmethod
    @with_import('lal')
    def from_lal(cls, lalts, copy=True):
        """Generate a new TimeSeries from a LAL TimeSeries of any type.
        """
        from ..utils.lal import from_lal_unit
        try:
            unit = from_lal_unit(lalts.sampleUnits)
        except TypeError:
            unit = None
        channel = Channel(lalts.name, sample_rate=1/lalts.deltaT, unit=unit,
                          dtype=lalts.data.data.dtype)
        out = cls(lalts.data.data, channel=channel, epoch=float(lalts.epoch),
                  copy=False)
        if copy:
            return out.copy()
        else:
            return out

    @with_import('lal')
    def to_lal(self):
        """Convert this `TimeSeries` into a LAL TimeSeries.
        """
        from ..utils.lal import (LAL_TYPE_STR_FROM_NUMPY, to_lal_unit)
        typestr = LAL_TYPE_STR_FROM_NUMPY[self.dtype.type]
        try:
            unit = to_lal_unit(self.unit)
        except (TypeError, AttributeError):
            try:
                unit = lal.DimensionlessUnit
            except AttributeError:
                unit = lal.lalDimensionlessUnit
        create = getattr(lal, 'Create%sTimeSeries' % typestr.upper())
        lalts = create(self.name, lal.LIGOTimeGPS(self.epoch.gps), 0,
                       self.dt.value, unit, self.size)
        lalts.data.data = self.value
        return lalts

    @classmethod
    def from_pycbc(cls, ts):
        """Convert a `pycbc.types.timeseries.TimeSeries` into a `TimeSeries`

        Parameters
        ----------
        ts : `pycbc.types.timeseries.TimeSeries`
            the input PyCBC `~pycbc.types.timeseries.TimeSeries` array

        Returns
        -------
        timeseries : `TimeSeries`
            a GWpy version of the input timeseries
        """
        return cls(ts.data, epoch=ts.start_time, sample_rate=1/ts.delta_t)

    @with_import('pycbc.types')
    def to_pycbc(self, copy=True):
        """Convert this `TimeSeries` into a PyCBC
        `~pycbc.types.timeseries.TimeSeries`

        Parameters
        ----------
        copy : `bool`, optional, default: `True`
            if `True`, copy these data to a new array

        Returns
        -------
        timeseries : `~pycbc.types.timeseries.TimeSeries`
            a PyCBC representation of this `TimeSeries`
        """
        return types.TimeSeries(self.data,
                                delta_t=self.dx.to('s').value,
                                epoch=self.epoch.gps, copy=copy)

    # -------------------------------------------
    # TimeSeries operations

    def __array_wrap__(self, obj, context=None):
        """Wrap an array into a TimeSeries, or a StateTimeSeries if
        dtype == bool.
        """
        if obj.dtype == numpy.dtype(bool):
            from .statevector import StateTimeSeries
            ufunc = context[0]
            value = context[1][-1]
            try:
                op_ = _UFUNC_STRING[ufunc.__name__]
            except KeyError:
                op_ = ufunc.__name__
            result = obj.view(StateTimeSeries)
            result.name = '%s %s %s' % (obj.name, op_, value)
            if hasattr(obj, 'unit') and str(obj.unit):
                result.name += ' %s' % str(obj.unit)
        else:
            result = super(TimeSeriesBase, self).__array_wrap__(
                obj, context=context)
        return result


class ArrayTimeSeries(TimeSeriesBase, Array2D):
    _default_xunit = TimeSeriesBase._default_xunit

    def __new__(cls, data, times=None, epoch=None, channel=None, unit=None,
                sample_rate=None, name=None, **kwargs):
        """Generate a new ArrayTimeSeries.
        """
        warnings.warn("The ArrayTimeSeries is deprecated and will be removed "
                      "before the 1.0 release", DeprecationWarning)
        # parse Channel input
        if channel:
            channel = (isinstance(channel, Channel) and channel or
                       Channel(channel))
            name = name or channel.name
            unit = unit or channel.unit
            sample_rate = sample_rate or channel.sample_rate
        # generate TimeSeries
        new = Array2D.__new__(cls, data, name=name, unit=unit, epoch=epoch,
                              channel=channel, x0=1/sample_rate,
                              xindex=times, **kwargs)
        return new


def as_series_dict_class(seriesclass):
    """Decorate a `dict` class to declare itself as the `DictClass` for
    its `EntryClass`

    This method should be used to decorate sub-classes of the
    `TimeSeriesBaseDict` to provide a reference to that class from the
    relevant subclass of `TimeSeriesBase`.
    """
    def decorate_class(cls):
        seriesclass.DictClass = cls
        return cls
    return decorate_class


@as_series_dict_class(TimeSeriesBase)
class TimeSeriesBaseDict(OrderedDict):
    """Ordered key-value mapping of named `TimeSeriesBase` objects

    This object is designed to hold data for many different sources (channels)
    for a single time span.

    The main entry points for this object are the
    :meth:`~TimeSeriesBaseDict.read` and :meth:`~TimeSeriesBaseDict.fetch`
    data access methods.
    """
    EntryClass = TimeSeriesBase

    # use input/output registry to allow multi-format reading
    read = classmethod(reader(doc="""
        Read data into a `TimeSeriesBaseDict`.

        Parameters
        ----------
        source : `str`, `~glue.lal.Cache`
            a single file path `str`, or a `~glue.lal.Cache` containing
            a contiguous list of files.
        channels : `~gwpy.detector.channel.ChannelList`, `list`
            a list of channels to read from the source.
        start : `~gwpy.time.Time`, `float`, optional
            GPS start time of required data.
        end : `~gwpy.time.Time`, `float`, optional
            GPS end time of required data.
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
        dict : `TimeSeriesBaseDict`
            a new `TimeSeriesBaseDict` containing data for the given channel.

        Raises
        ------
        Exception
            if no format could be automatically identified.

        Notes
        -----"""))

    write = writer(doc="""Write this `TimeSeriesDict` to a file

        Notes
        -----""")

    def __iadd__(self, other):
        return self.append(other)

    def copy(self):
        new = self.__class__()
        for key, val in self.iteritems():
            new[key] = val.copy()
        return new

    def append(self, other, copy=True, **kwargs):
        for key, ts in other.iteritems():
            if key in self:
                self[key].append(ts, **kwargs)
            elif copy:
                self[key] = ts.copy()
            else:
                self[key] = ts
        return self

    def prepend(self, other, **kwargs):
        for key, ts in other.iteritems():
            if key in self:
                self[key].prepend(ts, **kwargs)
            else:
                self[key] = ts
        return self

    def crop(self, start=None, end=None, copy=False):
        """Crop each entry of this `TimeSeriesBaseDict`.

        This method calls the :meth:`crop` method of all entries and
        modifies this dict in place.

        Parameters
        ----------
        start : `Time`, `float`
            GPS start time to crop `TimeSeries` at left
        end : `Time`, `float`
            GPS end time to crop `TimeSeries` at right

        See Also
        --------
        TimeSeries.crop
            for more details
        """
        for key, val in self.iteritems():
            self[key] = val.crop(start=start, end=end, copy=copy)
        return self

    def resample(self, rate, **kwargs):
        """Resample items in this dict.

        This operation over-writes items inplace.

        Parameters
        ----------
        rate : `dict`, `float`
            either a `dict` of (channel, `float`) pairs for key-wise
            resampling, or a single float/int to resample all items.
        kwargs
             other keyword arguments to pass to each item's resampling
             method.
        """
        if not isinstance(rate, dict):
            rate = dict((c, rate) for c in self)
        for key, resamp in rate.iteritems():
            self[key] = self[key].resample(resamp, **kwargs)
        return self

    @classmethod
    @with_import('nds2')
    def fetch(cls, channels, start, end, host=None, port=None,
              verify=False, verbose=False, connection=None,
              pad=None, type=NDS2_FETCH_TYPE_MASK, dtype=None):
        """Fetch data from NDS for a number of channels.

        Parameters
        ----------
        channels : `list`
            required data channels.
        start : `~gwpy.time.Time`, or float
            GPS start time of data span.
        end : `~gwpy.time.Time`, or float
            GPS end time of data span.
        host : `str`, optional
            URL of NDS server to use, defaults to observatory site host.
        port : `int`, optional
            port number for NDS server query, must be given with `host`.
        verify : `bool`, optional, default: `True`
            check channels exist in database before asking for data
        verbose : `bool`, optional
            print verbose output about NDS progress.
        connection : :class:`~gwpy.io.nds.NDS2Connection`
            open NDS connection to use.
        type : `int`, `str`,
            NDS2 channel type integer or string name.
        dtype : `numpy.dtype`, `str`, `type`, or `dict`
            numeric data type for returned data, e.g. `numpy.float`, or
            `dict` of (`channel`, `dtype`) pairs

        Returns
        -------
        data : :class:`~gwpy.timeseries.TimeSeriesBaseDict`
            a new `TimeSeriesBaseDict` of (`str`, `TimeSeries`) pairs fetched
            from NDS.
        """
        from ..segments import (Segment, SegmentList)
        from ..io import nds as ndsio
        # parse times
        start = to_gps(start)
        end = to_gps(end)
        istart = start.seconds
        iend = ceil(end)

        # test S6 h(t) channel
        if any(re.match('[HL]1:LDAS-STRAIN\Z', str(c)) for c in channels):
            verify = True

        # parse dtype
        if isinstance(dtype, (tuple, list)):
            dtype = [numpy.dtype(r) if r is not None else None for r in dtype]
            dtype = dict(zip(channels, dtype))
        elif not isinstance(dtype, dict):
            if dtype is not None:
                dtype = numpy.dtype(dtype)
            dtype = dict((channel, dtype) for channel in channels)

        # open connection for specific host
        if host and not port and re.match('[a-z]1nds[0-9]\Z', host):
            port = 8088
        elif host and not port:
            port = 31200
        if host is not None and port is not None and connection is None:
            if verbose:
                gprint("Connecting to %s:%s..." % (host, port), end=' ')
            connection = ndsio.auth_connect(host, port)
            if verbose:
                gprint("Connected.")
        elif connection is not None and verbose:
            gprint("Received connection to %s:%d."
                   % (connection.get_host(), connection.get_port()))
        # otherwise cycle through connections in logical order
        if connection is None:
            ifos = set([Channel(channel).ifo for channel in channels])
            if len(ifos) == 1:
                ifo = list(ifos)[0]
            else:
                ifo = None
            hostlist = ndsio.host_resolution_order(ifo, epoch=start)
            for host, port in hostlist:
                try:
                    return cls.fetch(channels, start, end, host=host,
                                     port=port, verbose=verbose, type=type,
                                     verify=verify, dtype=dtype, pad=pad)
                except (RuntimeError, ValueError) as e:
                    if verbose:
                        gprint('Something went wrong:', file=sys.stderr)
                        # if error and user supplied their own server, raise
                        warnings.warn(str(e), ndsio.NDSWarning)

            # if we got this far, we can't get all of the channels in one go
            if len(channels) > 1:
                return cls(
                    (c, cls.EntryClass.fetch(c, start, end, verbose=verbose,
                                             type=type, verify=verify,
                                             dtype=dtype.get(c), pad=pad))
                    for c in channels)
            e = "Cannot find all relevant data on any known server."
            if not verbose:
                e += (" Try again using the verbose=True keyword argument to "
                      "see detailed failures.")
            raise RuntimeError(e)

        # at this point we must have an open connection, so we can proceed
        # normally

        # verify channels
        if verify:
            if verbose:
                gprint("Checking channels against the NDS database...",
                       end=' ')
            else:
                warnings.filterwarnings('ignore', category=ndsio.NDSWarning,
                                        append=False)
            try:
                qchannels = ChannelList.query_nds2(channels,
                                                   connection=connection,
                                                   type=type, unique=True)
            except ValueError as e:
                try:
                    channels2 = ['%s*' % c for c in map(str, channels)]
                    qchannels = ChannelList.query_nds2(channels2,
                                                       connection=connection,
                                                       type=type, unique=True)
                except ValueError:
                    raise e
            if verbose:
                gprint("Complete.")
            else:
                warnings.filters.pop(0)
        else:
            qchannels = ChannelList(map(Channel, channels))

        # test for minute trends
        if (any([c.type == 'm-trend' for c in qchannels]) and
                (start % 60 or end % 60)):
            warnings.warn("Requested at least one minute trend, but "
                          "start and stop GPS times are not modulo "
                          "60-seconds (from GPS epoch). Times will be "
                          "expanded outwards to compensate")
            if start % 60:
                start = int(start) // 60 * 60
                istart = start
            if end % 60:
                end = int(end) // 60 * 60 + 60
                iend = end
            have_minute_trends = True
        else:
            have_minute_trends = False

        # get segments for data
        allsegs = SegmentList([Segment(istart, iend)])
        qsegs = SegmentList([Segment(istart, iend)])
        if pad is not None:
            from subprocess import CalledProcessError
            try:
                segs = ChannelList.query_nds2_availability(
                    channels, istart, iend, host=connection.get_host())
            except (RuntimeError, CalledProcessError) as e:
                warnings.warn(str(e), ndsio.NDSWarning)
            else:
                for channel in segs:
                    try:
                        csegs = sorted(segs[channel].values(),
                                       key=lambda x: abs(x))[-1]
                    except IndexError:
                        csegs = SegmentList([])
                    qsegs &= csegs

            if verbose:
                gprint('Found %d viable segments of data with %.2f%% coverage'
                       % (len(qsegs), abs(qsegs) / abs(allsegs) * 100))

        out = cls()
        for (istart, iend) in qsegs:
            istart = int(istart)
            iend = int(iend)
            # fetch data
            if verbose:
                gprint('Downloading data... ', end='\r')

            # determine buffer duration
            data = connection.iterate(istart, iend,
                                      [c.ndsname for c in qchannels])
            nsteps = 0
            i = 0
            for buffers in data:
                for buffer_, c in zip(buffers, channels):
                    ts = cls.EntryClass.from_nds2_buffer(
                        buffer_, dtype=dtype.get(c))
                    out.append({c: ts}, pad=pad,
                               gap=pad is None and 'raise' or 'pad')
                if not nsteps:
                    if have_minute_trends:
                        dur = buffer_.length * 60
                    else:
                        dur = buffer_.length / buffer_.channel.sample_rate
                    nsteps = ceil((iend - istart) / dur)
                i += 1
                if verbose:
                    gprint('Downloading data... %d%%' % (100 * i // nsteps),
                           end='\r')
                    if i == nsteps:
                        gprint('')

        # pad to end of request if required
        if len(qsegs) and iend < float(end):
            dt = float(end) - float(iend)
            for channel in out:
                nsamp = dt * out[channel].sample_rate.value
                out[channel].append(
                    numpy.ones(nsamp, dtype=out[channel].dtype) * pad)
        # match request exactly
        for channel in out:
            if istart > start or iend < end:
                out[channel] = out[channel].crop(start, end)

        if verbose:
            gprint('Success.')
        return out

    @classmethod
    def find(cls, channels, start, end, frametype=None,
             pad=None, dtype=None, nproc=1, verbose=False,
             allow_tape=True, observatory=None, **readargs):
        """Find and read data from frames for a number of channels.

        Parameters
        ----------
        channels : `list`
            required data channels.
        start : `~gwpy.time.Time`, or float
            GPS start time of data span.
        end : `~gwpy.time.Time`, or float
            GPS end time of data span.
        frametype : `str`, optional
            name of frametype in which this channel is stored, by default
            will search for all required frame types
        pad : `float`, optional
            value with which to fill gaps in the source data, only used if
            gap is not given, or `gap='pad'` is given
        dtype : `numpy.dtype`, `str`, `type`, or `dict`
            numeric data type for returned data, e.g. `numpy.float`, or
            `dict` of (`channel`, `dtype`) pairs
        nproc : `int`, optional, default: `1`
            number of parallel processes to use, serial process by
            default.
        verbose : `bool`, optional
            print verbose output about NDS progress.
        allow_tape : `bool`, optional, default: `True`
            allow reading from frames on tape
        **readargs
            any other keyword arguments to be passed to `.read()`
        """
        start = to_gps(start)
        end = to_gps(end)
        # -- find frametype(s)
        if frametype is None:
            frametypes = dict()
            for c in channels:
                ft = datafind.find_best_frametype(c, start, end,
                                                  allow_tape=allow_tape)
                try:
                    frametypes[ft].append(c)
                except KeyError:
                    frametypes[ft] = [c]
            if verbose and len(frametypes) > 1:
                gprint("Determined %d frametypes to read" % len(frametypes))
            elif verbose:
                gprint("Determined best frametype as %r"
                       % frametypes.keys()[0])
        else:
            frametypes = {frametype: channels}
        # -- read data
        out = cls()
        for ft, clist in frametypes.iteritems():
            if verbose:
                gprint("Reading data from %s frames..." % ft, end=' ')
            # parse as a ChannelList
            channellist = ChannelList.from_names(*clist)
            # strip trend tags from channel names
            names = [c.name for c in channellist]
            # find observatory for this group
            if observatory is None:
                try:
                    observatory = ''.join(
                        sorted(set(c.ifo[0] for c in channellist)))
                except TypeError as e:
                    e.args = ("Cannot parse list of IFOs from channel names",)
                    raise
            # find frames
            connection = datafind.connect()
            cache = connection.find_frame_urls(observatory, ft, start, end,
                                               urltype='file')
            if len(cache) == 0:
                raise RuntimeError("No %s-%s frame files found for [%d, %d)"
                                   % (observatory, ft, start, end))
            # read data
            readargs.setdefault('format', 'gwf')
            new = cls.read(cache, names, start=start, end=end, pad=pad,
                           dtype=dtype, nproc=nproc, **readargs)
            # map back to user-given channel name and append
            out.append(type(new)((key, new[c]) for
                                 (key, c) in zip(clist, names)))
            if verbose:
                gprint("Done")
        return out

    @classmethod
    def get(cls, channels, start, end, pad=None, dtype=None, verbose=False,
            allow_tape=False, **kwargs):
        """Retrieve data for multiple channels from frames or NDS

        This method dynamically accesses either frames on disk, or a
        remote NDS2 server to find and return data for the given interval

        Parameters
        ----------
        channels : `list`
            required data channels.
        start : `~gwpy.time.Time`, or float
            GPS start time of data span.
        end : `~gwpy.time.Time`, or float
            GPS end time of data span.
        frametype : `str`, optional
            name of frametype in which this channel is stored, by default
            will search for all required frame types
        pad : `float`, optional
            value with which to fill gaps in the source data, only used if
            gap is not given, or `gap='pad'` is given
        dtype : `numpy.dtype`, `str`, `type`, or `dict`
            numeric data type for returned data, e.g. `numpy.float`, or
            `dict` of (`channel`, `dtype`) pairs
        nproc : `int`, optional, default: `1`
            number of parallel processes to use, serial process by
            default.
        verbose : `bool`, optional
            print verbose output about NDS progress.
        allow_tape : `bool`, optional, default: `False`
            allow the use of frames that are held on tape, default is `False`
            to attempt to allow the `TimeSeries.fetch` method to
            intelligently select a server that doesn't use tapes for
            data storage (doesn't always work)
        **kwargs
            other keyword arguments to pass to either
            `TimeSeriesBaseDict.find` (for direct GWF file access) or
            `TimeSeriesBaseDict.fetch` for remote NDS2 access
        """
        try_frames = True
        # work out whether to use NDS2 or frames
        if not os.getenv('LIGO_DATAFIND_SERVER'):
            try_frames = False
        host = kwargs.get('host', None)
        if host is not None and host.startswith('nds'):
            try_frames = False
        # try and find from frames
        if try_frames:
            if verbose:
                gprint("Attempting to access data from frames...")
            try:
                return cls.find(channels, start, end, pad=pad, dtype=dtype,
                                verbose=verbose, allow_tape=allow_tape,
                                **kwargs)
            except (ImportError, RuntimeError, ValueError) as e:
                if verbose:
                    gprint(str(e), file=sys.stderr)
                    gprint("Failed to access data from frames, trying NDS...")
        # otherwise fetch from NDS
        kwargs.pop('nproc', None)
        kwargs.pop('frametype', None)
        kwargs.pop('observatory', None)
        try:
            return cls.fetch(channels, start, end, pad=pad, dtype=dtype,
                             verbose=verbose, **kwargs)
        except RuntimeError as e:
            # if all else fails, try and get each channel individually
            if len(channels) == 1:
                raise
            else:
                if verbose:
                    gprint(str(e), file=sys.stderr)
                    gprint("Failed to access data for all channels as a "
                           "group, trying individually:")
                return cls(
                    (c, cls.EntryClass.get(c, start, end, pad=pad, dtype=dtype,
                                           verbose=verbose, **kwargs))
                    for c in channels)

    def plot(self, label='key', **kwargs):
        """Plot the data for this `TimeSeriesBaseDict`.

        Parameters
        ----------
        label : `str`, optional
            labelling system to use, or fixed label for all elements
            Special values include

            - ``'key'``: use the key of the `TimeSeriesBaseDict`,
            - ``'name'``: use the :attr:`~TimeSeries.name` of each element

            If anything else, that fixed label will be used for all lines.

        **kwargs
            all other keyword arguments are passed to the plotter as
            appropriate
        """
        from ..plotter import TimeSeriesPlot
        figargs = dict()
        for key in ['figsize', 'dpi']:
            if key in kwargs:
                figargs[key] = kwargs.pop(key)
        plot_ = TimeSeriesPlot(**figargs)
        ax = plot_.gca()
        for lab, ts in self.iteritems():
            if label.lower() == 'name':
                lab = ts.name
            elif label.lower() != 'key':
                lab = label
            ax.plot(ts, label=lab, **kwargs)
        return plot_


class TimeSeriesBaseList(list):
    """Fancy list representing a list of `TimeSeriesBase`

    The `TimeSeriesBaseList` provides an easy way to collect and organise
    `TimeSeriesBase` for a single `Channel` over multiple segments.

    Parameters
    ----------
    *items
        any number of `TimeSeriesBase`

    Returns
    -------
    list
        a new `TimeSeriesBaseList`

    Raises
    ------
    TypeError
        if any elements are not `TimeSeriesBase`
    """
    EntryClass = TimeSeriesBase

    def __init__(self, *items):
        """Initialise a new list
        """
        super(TimeSeriesBaseList, self).__init__()
        for item in items:
            self.append(item)

    @property
    def segments(self):
        from ..segments import SegmentList
        return SegmentList([item.span for item in self])

    def append(self, item):
        if not isinstance(item, self.EntryClass):
            raise TypeError("Cannot append type '%s' to %s"
                            % (item.__class__.__name__,
                               self.__class__.__name__))
        super(TimeSeriesBaseList, self).append(item)
        return self
    append.__doc__ = list.append.__doc__

    def extend(self, item):
        item = TimeSeriesBaseList(item)
        super(TimeSeriesBaseList, self).extend(item)
    extend.__doc__ = list.extend.__doc__

    def coalesce(self):
        """Merge contiguous elements of this list into single objects

        This method implicitly sorts and potentially shortens the this list.
        """
        self.sort(key=lambda ts: ts.x0.value)
        i = j = 0
        N = len(self)
        while j < N:
            this = self[j]
            j += 1
            if j < N and this.is_contiguous(self[j]) == 1:
                while j < N and this.is_contiguous(self[j]):
                    try:
                        this = self[i] = this.append(self[j])
                    except ValueError as e:
                        if 'cannot resize this array' in str(e):
                            this = this.copy()
                            this = self[i] = this.append(self[j])
                        else:
                            raise
                    j += 1
            else:
                self[i] = this
            i += 1
        del self[i:]
        return self

    def join(self, pad=0.0, gap='raise'):
        """Concatenate all of the elements of this list into a single object

        Parameters
        ----------
        pad : `float`, optional, default: `0.0`
            value with which to pad gaps
        gap : `str`, optional, default: `'raise'`
            what to do in the event of a discontguity in the data, one of

            - 'raise': raise an exception
            - 'warn': print a warning
            - 'ignore': append series as if there was no gap
            - 'pad': pad the gap with a value

        Returns
        -------
        `TimeSeriesBase`
             a single `TimeSeriesBase` covering the full span of all entries
             in this list

        See Also
        --------
        TimeSeriesBase.append
            for details on how the individual series are concatenated together
        """
        if len(self) == 0:
            return self.EntryClass(numpy.empty((0,) * self.EntryClass._ndim))
        self.sort(key=lambda t: t.epoch.gps)
        out = self[0].copy()
        for ts in self[1:]:
            out.append(ts, gap=gap, pad=pad)
        return out

    def __getslice__(self, i, j):
        return type(self)(*super(TimeSeriesBaseList, self).__getslice__(i, j))
    __getslice__.__doc__ = list.__getslice__.__doc__
