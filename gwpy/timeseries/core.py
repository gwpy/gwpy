# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Array with metadata
"""

from __future__ import division

import numbers
import numpy
import warnings
from math import modf
from scipy import (fftpack, signal)

from astropy import units

import lal
from lal import (gpstime, utils as lalutils)
from lalframe import frread

from .. import version
from ..data import Series
from ..detector import Channel
from ..segments import Segment
from ..time import Time
from ..window import get_window

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ["TimeSeries"]


class TimeSeries(Series):
    """A data array holding some metadata to represent a time series of
    instrumental or analysis data.

    Parameters
    ----------
    data : `numpy.ndarray`, `list`
        Data values to initialise TimeSeries
    epoch : `float` GPS time, or :class:`~gwpy.time.Time`, optional
        TimeSeries start time
    channel : :class:`~gwpy.detector.Channel`, or `str`, optional
        Data channel for this TimeSeries
    unit : :class:`~astropy.units.Unit`, optional
        The units of the data

    Returns
    -------
    TimeSeries
        a new `TimeSeries`

    Notes
    -----
    Any regular array, i.e. any iterable collection of data, can be
    easily converted into a `TimeSeries`.

    >>> data = numpy.asarray([1,2,3])
    >>> series = TimeSeries(data)

    The necessary metadata to reconstruct timing information are recorded
    in the `epoch` and `sample_rate` attributes. This time-stamps can be
    returned via the :attr:`~TimeSeries.times` property.

    Attributes
    ----------
    name
    epoch
    unit
    channel
    sample_rate
    duration
    span

    Methods
    -------
    psd
    asd
    spectrogram
    plot
    read
    fetch
    resample
    highpass
    lowpass
    bandpass
    """
    _metadata_slots = ['name', 'unit', 'epoch', 'channel', 'sample_rate']
    xunit = units.Unit('s')
    def __new__(cls, data, times=None, epoch=None, channel=None, unit=None,
                sample_rate=None, name=None, **kwargs):
        """Generate a new TimeSeries.
        """
        # parse Channel input
        if channel:
            channel = Channel(channel)
            name = name or channel.name
            unit = unit or channel.unit
            sample_rate = sample_rate or channel.sample_rate
        # generate TimeSeries
        new = super(TimeSeries, cls).__new__(cls, data, name=name, unit=unit,
                                             epoch=epoch, channel=channel,
                                             sample_rate=sample_rate,
                                             times=times)
        return new

    # -------------------------------------------
    # TimeSeries properties

    @property
    def epoch(self):
        """Starting GPS time epoch for this `TimeSeries`.

        This attribute is recorded as a `~gwpy.time.Time` object in the
        GPS format, allowing native conversion into other formats.

        See `~astropy.time` for details on the `Time` object.
        """
        try:
            return Time(self.x0, format='gps')
        except KeyError:
            raise AttributeError("No epoch has been set for this %s"
                                 % self.__class__.__name__)

    @epoch.setter
    def epoch(self, epoch):
        if isinstance(epoch, Time):
            self.x0 = epoch.gps
        elif isinstance(epoch, units.Quantity):
            self.x0 = epoch
        else:
            self.x0 = float(epoch)

    dt = Series.dx

    @property
    def sample_rate(self):
        """Data rate for this `TimeSeries` in samples per second (Hertz).
        """
        return (1 / self.dx).to('Hertz')

    @sample_rate.setter
    def sample_rate(self, val):
        rate = units.Quantity(val, units.Hertz)
        self.dx = (1 / units.Quantity(val, units.Hertz)).to(self.xunit)

    @property
    def span(self):
        """Time Segment encompassed by thie `TimeSeries`.
        """
        return Segment(*map(numpy.float64, super(TimeSeries, self).span))

    @property
    def duration(self):
        """Duration of this `TimeSeries` in seconds
        """
        return units.Quantity(self.span[1] - self.span[0], self.xunit)

    times = property(fget=Series.index.__get__,
                     fset=Series.index.__set__,
                     fdel=Series.index.__delete__,
                     doc="""Series of GPS times for each sample""")

    # -------------------------------------------
    # TimeSeries accessors

    @classmethod
    def read(cls, source, channel, epoch=None, duration=None, datatype=None,
             verbose=False):
        """Read data into a `TimeSeries` from files on disk

        Parameters
        ----------
        source : `str`, :class:`glue.lal.Cache`, :lalsuite:`LALCache`
            source for data, one of:

            - a filepath for a GWF-format frame file,
            - a filepath for a LAL-format Cache file
            - a Cache object from GLUE or LAL

        channel : `str`, :class:`~gwpy.detector.channel.Channel`
            channel (name or object) to read
        epoch : :class:`~gwpy.time.Time`, optional
            start time of desired data
        duration : `float`, optional
            duration of desired data
        datatype : `type`, `numpy.dtype`, `str`, optional
            identifier for desired output data type
        verbose : `bool`, optional
            print verbose output

        Returns
        -------
        TimeSeries
            a new `TimeSeries` containing the data read from disk
        """
        if isinstance(channel, Channel):
            channel = channel.name
            if datatype is None:
                datatype = channel.dtype
        if epoch and isinstance(epoch, Time):
            epoch = epoch.gps
        lalts = frread.read_timeseries(source, channel, start=epoch,
                                       duration=duration, datatype=datatype,
                                       verbose=verbose)
        return cls.from_lal(lalts)

    @classmethod
    def fetch(cls, channel, start, end, host=None, port=None, verbose=False):
        """Fetch data from NDS into a TimeSeries

        Parameters
        ----------
        channel : :class:`~gwpy.detector.channel.Channel`, or `str`
            required data channel
        start : `~gwpy.time.Time`, or float
            GPS start time of data span
        end : `~gwpy.time.Time`, or float
            GPS end time of data span
        host : `str`, optional
            URL of NDS server to use, defaults to observatory site host
        port : `int`, optional
            port number for NDS server query, must be given with `host`
        verbose : `bool`, optional
            print verbose output about NDS progress

        Returns
        -------
        TimeSeries
            a new `TimeSeries` containing the data read from NDS
        """
        from ..io import nds
        channel = Channel(channel)
        if verbose:
            import warnings
            warnings.filterwarnings('always', '(.*)', nds.NDSWarning)
        def _nds_connect(host, port):
            try:
                connection = nds.NDSConnection(host, port) 
            except RuntimeError as e:
                if str(e).startswith('Request SASL authentication'):
                    print('\nError authenticating against %s' % host)
                    nds.kinit()
                    connection = nds.NDSConnection(host, port)
                else:
                    raise
            return connection

        # get type
        ndschanneltype = (nds.nds2.channel.CHANNEL_TYPE_RAW |
                          nds.nds2.channel.CHANNEL_TYPE_RDS |
                          nds.nds2.channel.CHANNEL_TYPE_STREND|
                          nds.nds2.channel.CHANNEL_TYPE_MTREND)

        # user-defined host
        if host:
            hostlist = [(host, port)]
        else:
            hostlist = nds.host_resolution_order(channel.ifo)
        for host,port in hostlist:
            if verbose:
                print("Connecting to %s:%s" % (host, port))
            connection = _nds_connect(host, port)
            try:
                if verbose:
                    print("Downloading data...")
                data = connection.fetch(start, end, channel, ndschanneltype,
                                        silent=not verbose)
            except RuntimeError as e:
                if verbose:
                    warnings.warn(str(e), nds.NDSWarning)
            else:
                if verbose:
                    warnings.filterwarnings('default', '(.*)', nds.NDSWarning)
                channel = Channel.from_nds2(data.channel)
                return cls(data.data, channel=channel,
                           epoch=lal.LIGOTimeGPS(data.gps_seconds,
                                                 data.gps_nanoseconds))
        raise RuntimeError("Cannot find relevant data on any known server")

    # -------------------------------------------
    # TimeSeries product methods

    def fft(self, fftlength=None):
        """Compute the one-dimensional discrete Fourier transform of
        this `TimeSeries`

        Parameters
        ----------
        fftlength : `int`, optional
            length of the desired Fourier transform.
            Input will be cropped or padded to match the desired length.
            If fftlength is not given, the length of the `TimeSeries`
            will be used

        Returns
        -------
        out : complex `Series`
            the transformed output, with populated frequencies array
            metadata

        See Also
        --------
        :mod:`scipy.fftpack` for the definition of the DFT and conventions
        used.
        """
        new = fftpack.fft(self.data, n=fftlength).view(Series)
        new.frequencies = fftpack.fftfreq(new.size, d=numpy.float64(self.dx))
        return new

    def psd(self, fftlength=None, fftstride=None, method='welch', window=None):
        """Calculate the power spectral density (PSD) `Spectrum` for this
        `TimeSeries`.

        The `method` argument can be one of
            * 'welch'
            * 'bartlett'
            * 'medianmean'
            * 'median'
        and any keyword arguments will be passed to the relevant method
        in `gwpy.spectrum`.

        Parameters
        ----------
        fftlength : `float`, default: :attr:`TimeSeries.duration`
            number of seconds in single FFT
        fftstride : `float`, optional, default: fftlength
            number of seconds between FFTs, default: no overlap
        method : `str`, optional, default: 'welch'
            average spectrum method
        window : `timeseries.Window`, optional
            window function to apply to timeseries prior to FFT

        Returns
        -------
        psd :  :class:`~gwpy.spectrum.core.Spectrum`
            a data series containing the PSD.
        """
        from ..spectrum import psd
        if fftlength is None:
            fftlength = self.duration
        if fftstride is None:
            fftstride = fftlength
        fftlength *= self.sample_rate.value
        fftstride *= self.sample_rate.value
        if window is not None:
            window = get_window(window, fftlength)
        psd_ = psd._lal_psd(self, method, fftlength, fftstride, window=window)
        if psd_.unit:
            psd_.unit.__doc__ = "Power spectral density"
        return psd_

    def asd(self, fftlength=None, fftstride=None, method='welch', window=None):
        """Calculate the amplitude spectral density (ASD) `Spectrum` for this
        `TimeSeries`.

        The `method` argument can be one of

            * 'welch'
            * 'bartlett'
            * 'medianmean'
            * 'median'

        and any keyword arguments will be passed to the relevant method
        in `gwpy.spectrum`.

        Parameters
        ----------
        fftlength : `float`, default: :attr:`TimeSeries.duration`
            number of seconds in single FFT
        fftstride : `float`, optional, default: fftlength
            number of seconds between FFTs, default: no overlap
        method : `str`, optional, default: 'welch'
            average spectrum method
        window : `timeseries.Window`, optional
            window function to apply to timeseries prior to FFT

        Returns
        -------
        psd :  :class:`~gwpy.spectrum.core.Spectrum`
            a data series containing the ASD.
        """
        asd = self.psd(fftlength, fftstride=fftstride, method=method,
                       window=window)
        asd **= 1/2.
        if asd.unit:
            asd.unit.__doc__ = "Amplitude spectral density"
        return asd

    def spectrogram(self, stride, fftlength=None, fftstride=None,
                    method='welch', window=None, logf=False):
        """Calculate the average power spectrogram of this `TimeSeries`
        using the specified average spectrum method

        Parameters
        ----------
        stride : `float`
            number of seconds in single PSD (column of spectrogram)
        fftlength : `float`
            number of seconds in single FFT
        method : `str`, optional, default: 'welch'
            average spectrum method
        fftstride : `int`, optiona, default: fftlength
            number of seconds between FFTs
        window : `timeseries.window.Window`, optional, default: `None`
            window function to apply to timeseries prior to FFT
        logf : `bool`, optional, default: `False`
            make frequency axis logarithmic
        """
        from ..spectrum import psd
        from ..spectrogram import Spectrogram
        if fftlength == None:
            fftlength = stride
        if fftstride == None:
            fftstride = fftlength
        dt = stride
        df = 1/fftlength
        stride *= self.sample_rate.value

        # get size of Spectrogram
        nsteps = int(self.size // stride)
        # get number of frequencies
        nfreqs = int(fftlength*self.sample_rate.value // 2 + 1)

        # generate output spectrogram
        out = Spectrogram(numpy.zeros((nsteps, nfreqs)), name=self.name,
                          epoch=self.epoch, f0=0, df=df, dt=dt,
                          logf=logf)
        if not nsteps:
            return out

        # stride through TimeSeries, recording PSDs as columns of spectrogram
        for step in range(nsteps):
            # find step TimeSeries
            idx = stride * step
            idx_end = idx + stride
            stepseries = self[idx:idx_end]
            steppsd = stepseries.psd(fftlength, fftstride, method,
                                     window=window)
            if logf:
                steppsd = steppsd.to_logf()
            out.data[step,:] = steppsd.data
        try:
            out.unit = self.unit / units.Hertz
        except KeyError:
            out.unit = 1 / units.Hertz
        return out

    def plot(self, **kwargs):
        """Plot the data for this TimeSeries.
        """
        from ..plotter import TimeSeriesPlot
        return TimeSeriesPlot(self, **kwargs)

    @classmethod
    def from_lal(cls, lalts):
        """Generate a new TimeSeries from a LAL TimeSeries of any type
        """
        # write Channel
        channel = Channel(lalts.name, 1/lalts.deltaT,
                          unit=lal.UnitToString(lalts.sampleUnits),
                          dtype=lalts.data.data.dtype)
        return cls(lalts.data.data, channel=channel, epoch=lalts.epoch,
                   unit=lal.UnitToString(lalts.sampleUnits))

    def to_lal(self):
        """Convert this `TimeSeries` into a LAL TimeSeries
        """
        laltype = lalutils.LAL_TYPE_FROM_NUMPY[self.dtype.type]
        typestr = lalutils.LAL_TYPE_STR[laltype]
        create = getattr(lal, 'Create%sTimeSeries' % typestr.upper())
        lalts = create(self.name, lal.LIGOTimeGPS(self.epoch.gps), 0,
                       self.dt.value, lal.lalDimensionlessUnit, self.size)
        lalts.data.data = self.data
        return lalts

    @classmethod
    def read(cls, source, channel, epoch=None, duration=None, datatype=None,
             verbose=False):
        """Read data into a `TimeSeries` from files on disk

        Parameters
        ----------
        source : `str`, :class:`glue.lal.Cache`, :lalsuite:`LALCache`
            source for data, one of:

            - a filepath for a GWF-format frame file,
            - a filepath for a LAL-format Cache file
            - a Cache object from GLUE or LAL

        channel : `str`, :class:`~gwpy.detector.channel.Channel`
            channel (name or object) to read
        epoch : :class:`~gwpy.time.Time`, optional
            start time of desired data
        duration : `float`, optional
            duration of desired data
        datatype : `type`, `numpy.dtype`, `str`, optional
            identifier for desired output data type
        verbose : `bool`, optional
            print verbose output

        Returns
        -------
        TimeSeries
            a new `TimeSeries` containing the data read from disk
        """
        if isinstance(channel, Channel):
            channel = channel.name
            if datatype is None:
                datatype = channel.dtype
        if epoch and isinstance(epoch, Time):
            epoch = epoch.gps
        lalts = frread.read_timeseries(source, channel, start=epoch,
                                       duration=duration, datatype=datatype,
                                       verbose=verbose)
        return cls.from_lal(lalts)

    @classmethod
    def fetch(cls, channel, start, end, host=None, port=None, verbose=False):
        """Fetch data from NDS into a TimeSeries

        Parameters
        ----------
        channel : :class:`~gwpy.detector.channel.Channel`, or `str`
            required data channel
        start : `~gwpy.time.Time`, or float
            GPS start time of data span
        end : `~gwpy.time.Time`, or float
            GPS end time of data span
        host : `str`, optional
            URL of NDS server to use, defaults to observatory site host
        port : `int`, optional
            port number for NDS server query, must be given with `host`
        verbose : `bool`, optional
            print verbose output about NDS progress

        Returns
        -------
        TimeSeries
            a new `TimeSeries` containing the data read from NDS
        """
        from ..io import nds
        channel = Channel(channel)
        if verbose:
            import warnings
            warnings.filterwarnings('always', '(.*)', nds.NDSWarning)
        def _nds_connect(host, port):
            try:
                connection = nds.NDSConnection(host, port) 
            except RuntimeError as e:
                if str(e).startswith('Request SASL authentication'):
                    print('\nError authenticating against %s' % host)
                    nds.kinit()
                    connection = nds.NDSConnection(host, port)
                else:
                    raise
            return connection

        # get type
        ndschanneltype = (nds.nds2.channel.CHANNEL_TYPE_RAW |
                          nds.nds2.channel.CHANNEL_TYPE_RDS |
                          nds.nds2.channel.CHANNEL_TYPE_STREND|
                          nds.nds2.channel.CHANNEL_TYPE_MTREND)

        # user-defined host
        if host:
            hostlist = [(host, port)]
        else:
            hostlist = nds.host_resolution_order(channel.ifo)
        for host,port in hostlist:
            if verbose:
                print("Connecting to %s:%s" % (host, port))
            connection = _nds_connect(host, port)
            try:
                if verbose:
                    print("Downloading data...")
                data = connection.fetch(start, end, channel, ndschanneltype,
                                        silent=not verbose)
            except RuntimeError as e:
                if 'start and stop times are not multiples' in str(e):
                    raise
                if verbose:
                    warnings.warn(str(e), nds.NDSWarning)
            else:
                if verbose:
                    warnings.filterwarnings('default', '(.*)', nds.NDSWarning)
                channel = Channel.from_nds2(data.channel)
                epoch = lal.LIGOTimeGPS(data.gps_seconds, data.gps_nanoseconds)
                return TimeSeries(data.data, channel=channel, epoch=epoch)
        raise RuntimeError("Cannot find relevant data on any known server")

    # -------------------------------------------
    # TimeSeries filtering

    def highpass(self, frequency, amplitude=0.9, order=8, method='scipy'):
        """Filter this `TimeSeries` with a Butterworth high-pass filter

        See (for example) :lalsuite:`XLALHighPassREAL8TimeSeries` for more
        information.

        Parameters
        ----------
        frequency : `float`
            minimum frequency for high-pass
        amplitude : `float`, optional
            desired amplitude response of the filter
        order : `int`, optional
            desired order of the Butterworth filter
        method : `str`, optional, default: 'scipy'
            choose method of high-passing, LAL or SciPy

        Returns
        -------
        TimeSeries

        See Also
        --------
        See :lalsuite:`XLALHighPassREAL8TimeSeries` for information on
        the LAL method, otherwise see :mod:`scipy.signal` for the SciPy
        method.
        """
        if method.lower() == 'lal':
            lalts = self.to_lal()
            highpass = getattr(lal, 'HighPass%s' % lalts.__class__.__name__)
            highpass(lalts, float(frequency), amplitude, order)
            return TimeSeries.from_lal(lalts)
        elif method.lower() == 'scipy':
            # build filter
            B,A = signal.butter(order, numpy.float64(frequency * 2.0 /
                                                     self.sample_rate),
                                btype='highpass')
            new = signal.lfilter(B, A, self, axis=0).view(self.__class__)
            new.metadata = self.metadata.copy()
            return new
        raise NotImplementedError("Highpass filter method '%s' not "
                                  "recognised, please choose one of "
                                  "'scipy' or 'lal'")

    def lowpass(self, frequency, amplitude=0.9, order=4, method='scipy'):
        """Filter this `TimeSeries` with a Butterworth low-pass filter

        Parameters
        ----------
        frequency : `float`
            minimum frequency for low-pass
        amplitude : `float`, optional
            desired amplitude response of the filter
        order : `int`, optional
            desired order of the Butterworth filter
        method : `str`, optional, default: 'scipy'
            choose method of high-passing, LAL or SciPy

        Returns
        -------
        TimeSeries

        See Also
        --------
        See :lalsuite:`XLALLowPassREAL8TimeSeries` for information on
        the LAL method, otherwise see :mod:`scipy.signal` for the SciPy
        method.
        """
        if method.lower() == 'lal':
            lalts = self.to_lal()
            lowpass = getattr(lal, 'LowPass%s' % lalts.__class__.__name__)
            lowpass(lalts, float(frequency), amplitude, order)
            return TimeSeries.from_lal(lalts)
        elif method.lower() == 'scipy':
            # build filter
            B,A = signal.butter(order, numpy.float64(frequency * 2.0 /
                                                     self.sample_rate),
                                btype='lowpass')
            new = signal.lfilter(B, A, self, axis=0).view(self.__class__)
            new.metadata = self.metadata.copy()
            return new
        raise NotImplementedError("Lowpass filter method '%s' not "
                                  "recognised, please choose one of "
                                  "'scipy' or 'lal'")


    def bandpass(self, flow, fhigh, amplitude=0.9, order=6, method='scipy'):
        """Filter this `TimeSeries` by applying both low- and high-pass
        filters.

        See (for example) :lalsuite:`XLALLowPassREAL8TimeSeries` for more
        information.

        Parameters
        ----------
        flow : `float`
            minimum frequency for high-pass
        fhigh : `float`
            maximum frequency for low-pass
        amplitude : `float`, optional
            desired amplitude response of the filter
        order : `int`, optional
            desired order of the Butterworth filter

        Returns
        -------
        TimeSeries

        See Also
        --------
        See :lalsuite:`XLALLowPassREAL8TimeSeries` for information on
        the LAL method, otherwise see :mod:`scipy.signal` for the SciPy
        method.
        """
        try:
            high = self.highpass(flow, amplitude=amplitude, order=order)
        except NotImplementedError as e:
            raise NotImplementedError(str(e).replace('Lowpass', 'Bandpass'))
        else:
            return high.lowpass(fhigh, amplitude=amplitude, order=order)

    # -------------------------------------------
    # Utilities

    def plot(self, **kwargs):
        """Plot the data for this TimeSeries.
        """
        from ..plotter import TimeSeriesPlot
        return TimeSeriesPlot(self, **kwargs)

    @classmethod
    def from_lal(cls, lalts):
        """Generate a new TimeSeries from a LAL TimeSeries of any type
        """
        # write Channel
        channel = Channel(lalts.name, 1/lalts.deltaT,
                          unit=lal.UnitToString(lalts.sampleUnits),
                          dtype=lalts.data.data.dtype)
        return cls(lalts.data.data, channel=channel, epoch=lalts.epoch,
                   unit=lal.UnitToString(lalts.sampleUnits))

    def to_lal(self):
        """Convert this `TimeSeries` into a LAL TimeSeries
        """
        laltype = lalutils.LAL_TYPE_FROM_NUMPY[self.dtype.type]
        typestr = lalutils.LAL_TYPE_STR[laltype]
        create = getattr(lal, 'Create%sTimeSeries' % typestr.upper())
        lalts = create(self.name, lal.LIGOTimeGPS(self.epoch.gps), 0,
                       self.dt.value, lal.lalDimensionlessUnit, self.size)
        lalts.data.data = self.data
        return lalts

    @staticmethod
    def watch(channel, duration, stride=1, host=None, port=None,
              outputfile=None, verbose=False):
        """Stream data over NDS2, with an updating display
        """
        from ..io import nds
        # user-defined host
        if host:
            hostlist = [(host, port)]
        else:
            hostlist = nds.host_resolution_order(channel.ifo)
        for host,port in hostlist:
            # connect and find channel
            if verbose:
                print("Connecting to %s:%s" % (host, port))
            connection = nds.NDSConnection(host, port)
            try:
                channel = connection.find(
                          str(channel), nds.nds2.channel.CHANNEL_TYPE_ONLINE)[0]
            except IndexError:
                continue
            # begin iteration
            sampling = channel.sample_rate
            timeseries = TimeSeries(numpy.zeros((sampling*duration,)),
                                    channel=Channel.from_nds2(channel))
            timeseries.epoch = gpstime.gps_time_now()-duration
            plot = timeseries.plot(auto_refresh=True)
            trace = plot._layers[timeseries.name]
            try:
                for buf in connection.iterate([channel.name]):
                    newdata = buf[0]
                    epoch = lal.LIGOTimeGPS(newdata.gps_seconds,
                                            newdata.gps_nanoseconds)
                    # shift
                    timeseries.epoch = epoch + stride - duration
                    timeseries.data[:-newdata.length] = (
                        timeseries.data[newdata.length:])
                    timeseries.data[-newdata.length:] = newdata.data
                    trace.set_xdata(timeseries.times.data)
                    trace.set_ydata(timeseries.data)
                    plot.epoch = epoch
                    del plot.ylim
                    plot.xlim = timeseries.span
            except KeyboardInterrupt:
                return
