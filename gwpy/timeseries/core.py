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

import sys
import warnings
import re
from math import (ceil, pi)
from multiprocessing import (Process, Queue as ProcessQueue)

import numpy
from numpy import fft as npfft
from scipy import signal

from matplotlib import mlab

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

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


from .. import version
from ..data import (Array2D, Series)
from ..detector import (Channel, ChannelList)
from ..io import reader
from ..segments import (Segment, SegmentList)
from ..time import (Time, to_gps)
from ..utils import (gprint, update_docstrings, with_import)
from . import common

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

__all__ = ['TimeSeries', 'ArrayTimeSeries', 'TimeSeriesList',
           'TimeSeriesDict']

_UFUNC_STRING = {'less': '<',
                 'less_equal': '<=',
                 'equal': '==',
                 'greater_equal': '>=',
                 'greater': '>',
                 }


@update_docstrings
class TimeSeries(Series):
    """An `Array` with time-domain metadata.

    Parameters
    ----------
    data : `numpy.ndarray`, `list`
        Data values to initialise TimeSeries
    epoch : `float` GPS time, or :class:`~gwpy.time.Time`, optional
        TimeSeries start time
    channel : :class:`~gwpy.detector.channels.Channel`, `str`, optional
        Data channel for this TimeSeries
    unit : :class:`~astropy.units.core.Unit`, optional
        The units of the data

    Returns
    -------
    TimeSeries
        a new `TimeSeries`

    Notes
    -----
    Any regular array, i.e. any iterable collection of data, can be
    easily converted into a `TimeSeries`::

        >>> data = numpy.asarray([1,2,3])
        >>> series = TimeSeries(data)

    The necessary metadata to reconstruct timing information are recorded
    in the `epoch` and `sample_rate` attributes. This time-stamps can be
    returned via the :attr:`~TimeSeries.times` property.

    All comparison operations performed on a `TimeSeries` will return a
    :class:`~gwpy.timeseries.statevector.StateTimeSeries` - a boolean array
    with metadata copied from the starting `TimeSeries`.
    """
    _metadata_slots = ['name', 'unit', 'epoch', 'channel', 'sample_rate']
    xunit = units.Unit('s')

    def __new__(cls, data, times=None, epoch=None, channel=None, unit=None,
                sample_rate=None, name=None, **kwargs):
        """Generate a new TimeSeries.
        """
        # parse Channel input
        if channel:
            channel = (isinstance(channel, Channel) and channel or
                       Channel(channel))
            name = name or channel.name
            unit = unit or channel.unit
            sample_rate = sample_rate or channel.sample_rate
        # generate TimeSeries
        new = super(TimeSeries, cls).__new__(cls, data, name=name, unit=unit,
                                             epoch=epoch, channel=channel,
                                             sample_rate=sample_rate,
                                             times=times, **kwargs)
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
            return Time(self.x0, format='gps', scale='utc')
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
        if isinstance(val, int):
            val = float(val)
        self.dx = (1 / units.Quantity(val, units.Hertz)).to(self.xunit)
        if numpy.isclose(self.dx.value, round(self.dx.value)):
            self.dx = units.Quantity(round(self.dx.value), self.dx.unit)

    @property
    def span(self):
        """Time Segment encompassed by thie `TimeSeries`.
        """
        return Segment(*Series.span.fget(self))

    @property
    def duration(self):
        """Duration of this `TimeSeries` in seconds.
        """
        return units.Quantity(self.span[1] - self.span[0], self.xunit)

    times = property(fget=Series.index.__get__,
                     fset=Series.index.__set__,
                     fdel=Series.index.__delete__,
                     doc="""Series of GPS times for each sample""")

    unit = property(fget=Series.unit.__get__,
                    fset=Series.unit.__set__,
                    fdel=Series.unit.__delete__,
                    doc="""Unit for this `TimeSeries`

                        :type: :class:`~astropy.units.core.Unit`
                        """)

    channel = property(fget=Series.channel.__get__,
                       fset=Series.channel.__set__,
                       fdel=Series.channel.__delete__,
                       doc="""Source data `Channel` for this `TimeSeries`

                           :type: :class:`~gwpy.detector.channel.Channel`
                           """)

    # -------------------------------------------
    # TimeSeries accessors

    # use input/output registry to allow multi-format reading
    read = classmethod(reader(doc="""
        Read data into a `TimeSeries`.

        Parameters
        ----------
        source : `str`, `~glue.lal.Cache`
            a single file path `str`, or a `~glue.lal.Cache` containing
            a contiguous list of files.
        channel : `str`, `~gwpy.detector.core.Channel`
            the name of the channel to read, or a `Channel` object.
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

        Returns
        -------
        timeseries : `TimeSeries`
            a new `TimeSeries` containing data for the given channel.

        Raises
        ------
        Exception
            if no format could be automatically identified.

        Notes
        -----"""))

    @classmethod
    @with_import('nds2')
    def fetch(cls, channel, start, end, host=None, port=None, verbose=False,
              connection=None, verify=False, pad=None,
              type=NDS2_FETCH_TYPE_MASK, dtype=None):
        """Fetch data from NDS into a TimeSeries.

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
        verify : `bool`, optional, default: `True`
            check channels exist in database before asking for data
        connection : :class:`~gwpy.io.nds.NDS2Connection`
            open NDS connection to use
        verbose : `bool`, optional
            print verbose output about NDS progress
        type : `int`, optional
            NDS2 channel type integer
        dtype : `type`, `numpy.dtype`, `str`, optional
            identifier for desired output data type

        Returns
        -------
        TimeSeries
            a new `TimeSeries` containing the data read from NDS
        """
        return TimeSeriesDict.fetch(
            [channel], start, end, host=host, port=port,
            verbose=verbose, connection=connection, verify=verify,
            pad=pad, type=type, dtype=dtype)[str(channel)]

    # -------------------------------------------
    # TimeSeries product methods

    def fft(self, nfft=None):
        """Compute the one-dimensional discrete Fourier transform of
        this `TimeSeries`.

        Parameters
        ----------
        nfft : `int`, optional
            length of the desired Fourier transform.
            Input will be cropped or padded to match the desired length.
            If nfft is not given, the length of the `TimeSeries`
            will be used

        Returns
        -------
        out : :class:`~gwpy.spectrum.Spectrum`
            the normalised, complex-valued FFT `Spectrum`.

        See Also
        --------
        :mod:`scipy.fftpack` for the definition of the DFT and conventions
        used.

        Notes
        -----
        This method, in constrast to the :meth:`numpy.fft.rfft` method
        it calls, applies the necessary normalisation such that the
        amplitude of the output :class:`~gwpy.spectrum.Spectrum` is
        correct.
        """
        from ..spectrum import Spectrum
        dft = npfft.rfft(self.data, n=nfft) / nfft
        dft[1:] *= 2.0
        new = Spectrum(dft, epoch=self.epoch, channel=self.channel,
                       unit=self.unit)
        new.frequencies = npfft.rfftfreq(self.size, d=self.dx.value)
        return new

    def average_fft(self, fftlength=None, overlap=0, window=None):
        """Compute the averaged one-dimensional DFT of this `TimeSeries`.

        This method computes a number of FFTs of duration ``fftlength``
        and ``overlap`` (both given in seconds), and returns the mean
        average. This method is analogous to the Welch average method
        for power spectra.

        Parameters
        ----------
        fftlength : `float`
            number of seconds in single FFT, default, use
            whole `TimeSeries`
        overlap : `float`
            numbers of seconds by which to overlap neighbouring FFTs,
            by default, no overlap is used.
        window : `str`, :class:`numpy.ndarray`
            name of the window function to use, or an array of length
            ``fftlength * TimeSeries.sample_rate`` to use as the window.

        Returns
        -------
        out : complex-valued :class:`~gwpy.spectrum.Spectrum`
            the transformed output, with populated frequencies array
            metadata

        See Also
        --------
        :mod:`scipy.fftpack` for the definition of the DFT and conventions
        used.
        """
        from gwpy.spectrogram import Spectrogram
        # format lengths
        if fftlength is None:
            fftlength = self.duration
        if isinstance(fftlength, units.Quantity):
            fftlength = fftlength.value
        nfft = int((fftlength * self.sample_rate).decompose().value)
        noverlap = int((overlap * self.sample_rate).decompose().value)

        navg = divmod(self.size-noverlap, (nfft-noverlap))[0]

        # format window
        if window is None:
            window = 'boxcar'
        if isinstance(window, str) or type(window) is tuple:
            win = signal.get_window(window, nfft)
        else:
            win = numpy.asarray(window)
            if len(win.shape) != 1:
                raise ValueError('window must be 1-D')
            elif win.shape[0] != nfft:
                raise ValueError('Window is the wrong size.')
        scaling = 1. / numpy.absolute(win).mean()

        if nfft % 2:
            nfreqs = (nfft + 1) // 2
        else:
            nfreqs = nfft // 2 + 1
        ffts = Spectrogram(numpy.zeros((navg, nfreqs), dtype=numpy.complex),
                           channel=self.channel, epoch=self.epoch, f0=0,
                           df=1 / fftlength, dt=1, copy=True)
        # stride through TimeSeries, recording FFTs as columns of Spectrogram
        idx = 0
        for i in range(navg):
            # find step TimeSeries
            idx_end = idx + nfft
            if idx_end > self.size:
                continue
            stepseries = self[idx:idx_end]
            # detrend
            stepseries -= stepseries.data.mean()
            # window
            stepseries *= win
            # calculated FFT, weight, and stack
            fft_ = stepseries.fft(nfft=nfft) * scaling
            ffts.data[i, :] = fft_.data
            idx += (nfft - noverlap)
        mean = ffts.mean(0)
        mean.name = self.name
        mean.epoch = self.epoch
        mean.channel = self.channel
        return mean

    def psd(self, fftlength=None, overlap=None, method='welch', **kwargs):
        """Calculate the PSD `Spectrum` for this `TimeSeries`.

        Parameters
        ----------
        method : `str`, optional, default: 'welch'
            average spectrum method
        fftlength : `float`, default: :attr:`TimeSeries.duration`
            number of seconds in single FFT
        overlap : `float`, optional, default: `None`
            number of seconds of overlap between FFTs, defaults to that of
            the relevant method.
        window : `timeseries.Window`, optional
            window function to apply to timeseries prior to FFT
        plan : :lalsuite:`REAL8FFTPlan`, optional
            LAL FFT plan to use when generating average spectrum,
            substitute type 'REAL8' as appropriate.

        Returns
        -------
        psd :  :class:`~gwpy.spectrum.core.Spectrum`
            a data series containing the PSD.

        Notes
        -----
        The available methods are:

        """
        from ..spectrum.registry import get_method
        # get method
        scaling = kwargs.get('scaling', 'density')
        method_func = get_method(method, scaling=scaling)
        # type-cast arguments
        if fftlength is None:
            fftlength = self.duration
        nfft = int((fftlength * self.sample_rate).decompose().value)
        if overlap is not None:
            kwargs['noverlap'] = int(
                (overlap * self.sample_rate).decompose().value)
        # calculate and return spectrum
        psd_ = method_func(self, nfft, **kwargs)
        psd_.unit.__doc__ = 'Power spectral density'
        return psd_

    def asd(self, fftlength=None, overlap=None, method='welch', **kwargs):
        """Calculate the ASD `Spectrum` of this `TimeSeries`.

        Parameters
        ----------
        method : `str`, optional, default: 'welch'
            average spectrum method
        fftlength : `float`, default: :attr:`TimeSeries.duration`
            number of seconds in single FFT
        overlap : `float`, optional, default: `None`
            number of seconds of overlap between FFTs, defaults to that of
            the relevant method.
        window : `timeseries.Window`, optional
            window function to apply to timeseries prior to FFT
        plan : :lalsuite:`REAL8FFTPlan`, optional
            LAL FFT plan to use when generating average spectrum,
            substitute type 'REAL8' as appropriate.

        Returns
        -------
        psd :  :class:`~gwpy.spectrum.core.Spectrum`
            a data series containing the PSD.

        Notes
        -----
        The available methods are:

        """
        asd_ = self.psd(method=method, fftlength=fftlength,
                        overlap=overlap, **kwargs) ** (1/2.)
        asd_.unit.__doc__ = 'Amplitude spectral density'
        return asd_

    def spectrogram(self, stride, fftlength=None, overlap=None,
                    method='welch', window=None, nproc=1, **kwargs):
        """Calculate the average power spectrogram of this `TimeSeries`
        using the specified average spectrum method.

        Parameters
        ----------
        timeseries : :class:`~gwpy.timeseries.core.TimeSeries`
            input time-series to process.
        stride : `float`
            number of seconds in single PSD (column of spectrogram).
        fftlength : `float`
            number of seconds in single FFT.
        overlap : `int`, optiona, default: fftlength
            number of seconds between FFTs.
        method : `str`, optional, default: 'welch'
            average spectrum method.
        window : `timeseries.window.Window`, optional, default: `None`
            window function to apply to timeseries prior to FFT.
        plan : :lalsuite:`REAL8FFTPlan`, optional
            LAL FFT plan to use when generating average spectrum,
            substitute type 'REAL8' as appropriate.
        nproc : `int`, default: ``1``
            maximum number of independent frame reading processes, default
            is set to single-process file reading.

        Returns
        -------
        spectrogram : :class:`~gwpy.spectrogram.core.Spectrogram`
            time-frequency power spectrogram as generated from the
            input time-series.
        """
        from ..spectrum.utils import (safe_import, scale_timeseries_units)
        from ..spectrum.registry import get_method
        from ..spectrogram import (Spectrogram, SpectrogramList)

        # format FFT parameters
        if fftlength is None:
            fftlength = stride
        stride = units.Quantity(stride, 's').value
        fftlength = units.Quantity(fftlength, 's').value
        overlap = units.Quantity(overlap, 's').value

        # sanity check parameters
        if stride > abs(self.span):
            raise ValueError("stride cannot be greater than the duration of "
                             "this TimeSeries")
        if fftlength > stride:
            raise ValueError("fftlength cannot be greater than stride")
        if overlap >= fftlength:
            raise ValueError("overlap must be less than fftlength")


        # get size of spectrogram
        nsamp = int((stride * self.sample_rate).decompose().value)
        nfft = int((fftlength * self.sample_rate).decompose().value)
        nsteps = int(self.size // nsamp)
        nproc = min(nsteps, nproc)

        # generate window and plan if needed
        method_func = get_method(method)
        if method_func.__module__.endswith('lal_'):
            safe_import('lal', method)
            from ..spectrum.lal_ import (generate_lal_fft_plan,
                                         generate_lal_window)
            if kwargs.get('window', None) is None:
                kwargs['window'] = generate_lal_window(nfft, dtype=self.dtype)
            if kwargs.get('plan', None) is None:
                kwargs['plan'] = generate_lal_fft_plan(nfft, dtype=self.dtype)
        elif window is not None:
            kwargs['window'] = window

        # set up single process Spectrogram generation
        def _from_timeseries(ts):
            """Generate a `Spectrogram` from a `TimeSeries`.
            """
            # calculate specgram parameters
            dt = stride
            df = 1 / fftlength

            # get size of spectrogram
            nsteps_ = int(ts.size // nsamp)
            nfreqs = int(fftlength * ts.sample_rate.value // 2 + 1)

            # generate output spectrogram
            out = Spectrogram(numpy.zeros((nsteps_, nfreqs)),
                              channel=ts.channel, epoch=ts.epoch, f0=0, df=df,
                              dt=dt, copy=True)
            out.unit = scale_timeseries_units(
                ts.unit, kwargs.get('scaling', 'density'))

            if not nsteps_:
                return out

            # stride through TimeSeries, calcaulting PSDs
            for step in range(nsteps_):
                # find step TimeSeries
                idx = nsamp * step
                idx_end = idx + nsamp
                stepseries = ts[idx:idx_end]
                steppsd = stepseries.psd(fftlength=fftlength, overlap=overlap,
                                         method=method, **kwargs)
                out.data[step, :] = steppsd.data

            return out

        # single-process return
        if nsteps == 0 or nproc == 1:
            return _from_timeseries(self)

        # wrap spectrogram generator
        def _specgram(q, ts):
            try:
                q.put(_from_timeseries(ts))
            except Exception as e:
                q.put(e)

        # otherwise build process list
        stepperproc = int(ceil(nsteps / nproc))
        nsampperproc = stepperproc * nsamp
        queue = ProcessQueue(nproc)
        processlist = []
        for i in range(nproc):
            process = Process(target=_specgram,
                              args=(queue, self[i * nsampperproc:
                                                (i + 1) * nsampperproc]))
            process.daemon = True
            processlist.append(process)
            process.start()
            if ((i + 1) * nsampperproc) >= self.size:
                break

        # get data
        data = []
        for process in processlist:
            result = queue.get()
            if isinstance(result, Exception):
                raise result
            else:
                data.append(result)
        # and block
        for process in processlist:
            process.join()

        # format and return
        out = SpectrogramList(*data)
        out.sort(key=lambda spec: spec.epoch.gps)
        return out.join()

    def spectrogram2(self, fftlength, overlap=0, window='hanning',
                     scaling='density', **kwargs):
        """Calculate the non-averaged power `Spectrogram` of this `TimeSeries`

        Parameters
        ----------
        fftlength : `float`
            number of seconds in single FFT.
        overlap : `float`, optional
            number of seconds between FFTs.
        window : `str` or `tuple` or `array-like`, optional
            desired window to use. See `~scipy.signal.get_window` for a list
            of windows and required parameters. If `window` is array_like it
            will be used directly as the window.
        scaling : [ 'density' | 'spectrum' ], optional
            selects between computing the power spectral density ('density')
            where the `Spectrogram` has units of V**2/Hz if the input is
            measured in V and computing the power spectrum ('spectrum')
            where the `Spectrogram` has units of V**2 if the input is
            measured in V. Defaults to 'density'.
        **kwargs
            other parameters to be passed to `scipy.signal.periodogram` for
            each column of the `Spectrogram`

        Returns
        -------
        spectrogram: `~gwpy.spectrogram.Spectrogram`
            a power `Spectrogram` with `1/fftlength` frequency resolution and
            (fftlength - overlap) time resolution.

        See also
        --------
        scipy.signal.periodogram
            for documentation on the Fourier methods used in this calculation

        Notes
        -----
        This method calculates overlapping periodograms for all possible
        chunks of data entirely containing within the span of the input
        `TimeSeries`, then normalises the power in overlapping chunks using
        a triangular window centred on that chunk which most overlaps the
        given `Spectrogram` time sample.
        """
        from ..spectrogram import Spectrogram
        from ..spectrum import scale_timeseries_units
        # get parameters
        sampling = units.Quantity(self.sample_rate, 'Hz').value
        if isinstance(fftlength, units.Quantity):
            fftlength = units.Quantity(fftlength, 's').value
        if isinstance(overlap, units.Quantity):
            overlap = units.Quantity(overlap, 's').value

        # sanity check
        if fftlength > abs(self.span):
            raise ValueError("fftlength cannot be greater than the duration "
                             "of this TimeSeries")
        if overlap >= fftlength:
            raise ValueError("overlap must be less than fftlength")

        # convert to samples
        nfft = int(fftlength * sampling)  # number of points per FFT
        noverlap = int(overlap * sampling)  # number of points of overlap
        nstride = nfft - noverlap  # number of points between FFTs

        # create output object
        nsteps = int(self.size // nstride) - 1  # number of columns
        nfreqs = int(nfft // 2 + 1)  # number of rows
        unit = scale_timeseries_units(self.unit, scaling)
        tmp = numpy.zeros((nsteps, nfreqs), dtype=self.dtype)
        out = Spectrogram(numpy.zeros((nsteps, nfreqs), dtype=self.dtype),
                          epoch=self.epoch, channel=self.channel,
                          name=self.name, unit=unit, dt=fftlength-overlap,
                          f0=0, df=1/fftlength)

        # calculate overlapping periodograms
        for i in xrange(nsteps):
            idx = i * nstride
            # don't proceed past end of data, causes artefacts
            if idx+nfft > self.size:
                break
            ts = self.data[idx:idx+nfft]
            tmp[i, :] = signal.periodogram(ts, fs=sampling, window=window,
                                           nfft=nfft, scaling=scaling,
                                           **kwargs)[1]
        # normalize for over-dense grid
        density = nfft//nstride
        weights = signal.triang(density)
        for i in xrange(nsteps):
            # get indices of overlapping columns
            x0 = max(0, i+1-density)
            x1 = min(i+1, nsteps-density+1)
            if x0 == 0:
                w = weights[-x1:]
            elif x1 == nsteps - density + 1:
                w = weights[:x1-x0]
            else:
                w = weights
            # calculate weighted average
            out[i, :] = numpy.average(tmp[x0:x1], axis=0, weights=w)

        return out

    def fftgram(self, stride):
        """Calculate the Fourier-gram of this `TimeSeries`.

        At every ``stride``, a single, complex FFT is calculated.

        Parameters
        ----------
        stride : `float`
            number of seconds in single PSD (column of spectrogram)

        Returns
        -------
        fftgram : :class:`~gwpy.spectrogram.core.Spectrogram`
            a Fourier-gram
        """
        from ..spectrogram import Spectrogram

        fftlength = stride
        dt = stride
        df = 1/fftlength
        stride *= self.sample_rate.value
        # get size of Spectrogram
        nsteps = int(self.size // stride)
        # get number of frequencies
        nfreqs = int(fftlength*self.sample_rate.value)

        # generate output spectrogram
        dtype = numpy.complex
        out = Spectrogram(numpy.zeros((nsteps, nfreqs), dtype=dtype),
                          name=self.name, epoch=self.epoch, f0=0, df=df,
                          dt=dt, copy=True, unit=self.unit, dtype=dtype)
        # stride through TimeSeries, recording FFTs as columns of Spectrogram
        for step in range(nsteps):
            # find step TimeSeries
            idx = stride * step
            idx_end = idx + stride
            stepseries = self[idx:idx_end]
            # calculated FFT and stack
            stepfft = stepseries.fft()
            out[step] = stepfft.data
            if step == 0:
                out.frequencies = stepfft.frequencies
        return out

    def spectral_variance(self, stride, fftlength=None, overlap=None,
                          method='welch', window=None, nproc=1,
                          filter=None, bins=None, low=None, high=None,
                          nbins=500, log=False, norm=False, density=False):
        """Calculate the `SpectralVariance` of this `TimeSeries`.

        Parameters
        ----------
        stride : `float`
            number of seconds in single PSD (column of spectrogram)
        fftlength : `float`
            number of seconds in single FFT
        method : `str`, optional, default: 'welch'
            average spectrum method
        overlap : `int`, optiona, default: fftlength
            number of seconds between FFTs
        window : `timeseries.window.Window`, optional, default: `None`
            window function to apply to timeseries prior to FFT
        nproc : `int`, default: ``1``
            maximum number of independent frame reading processes, default
            is set to single-process file reading.
        bins : :class:`~numpy.ndarray`, optional, default `None`
            array of histogram bin edges, including the rightmost edge
        low : `float`, optional, default: `None`
            left edge of lowest amplitude bin, only read
            if ``bins`` is not given
        high : `float`, optional, default: `None`
            right edge of highest amplitude bin, only read
            if ``bins`` is not given
        nbins : `int`, optional, default: `500`
            number of bins to generate, only read if ``bins`` is not
            given
        log : `bool`, optional, default: `False`
            calculate amplitude bins over a logarithmic scale, only
            read if ``bins`` is not given
        norm : `bool`, optional, default: `False`
            normalise bin counts to a unit sum
        density : `bool`, optional, default: `False`
            normalise bin counts to a unit integral

        Returns
        -------
        specvar : `SpectralVariance`
            2D-array of spectral frequency-amplitude counts

        See Also
        --------
        :func:`numpy.histogram`
            for details on specifying bins and weights
        """
        specgram = self.spectrogram(stride, fftlength=fftlength,
                                    overlap=overlap, method=method,
                                    window=window, nproc=nproc)
        specgram **= 1/2.
        if filter:
            specgram = specgram.filter(*filter)
        return specgram.variance(bins=bins, low=low, high=high, nbins=nbins,
                                 log=log, norm=norm, density=density)

    def rayleigh_spectrum(self, fftlength=None, overlap=None, **kwargs):
        """Calculate the Rayleigh `Spectrum` for this `TimeSeries`.

        Parameters
        ----------
        fftlength : `float`, default: :attr:`TimeSeries.duration`
            number of seconds in single FFT
        overlap : `float`, optional, default: `None`
            number of seconds of overlap between FFTs, defaults to that of
            the relevant method.
        window : `timeseries.Window`, optional
            window function to apply to timeseries prior to FFT
        plan : :lalsuite:`REAL8FFTPlan`, optional
            LAL FFT plan to use when generating average spectrum,
            substitute type 'REAL8' as appropriate.

        Returns
        -------
        psd :  :class:`~gwpy.spectrum.core.Spectrum`
            a data series containing the PSD.

        Notes
        -----
        The available methods are:

        """
        from ..spectrum.registry import get_method
        # get method
        method_func = get_method('rayleigh')
        # type-cast arguments
        if fftlength is None:
            fftlength = self.duration
        nfft = int((fftlength * self.sample_rate).decompose().value)
        if overlap is not None:
            kwargs['noverlap'] = int(
                (overlap * self.sample_rate).decompose().value)
        # calculate and return spectrum
        spec_ = method_func(self, nfft, **kwargs)
        spec_.unit.__doc__ = 'Rayleigh statistic'
        return spec_

    def rayleigh_spectrogram(self, stride, fftlength=None, overlap=0,
                             window=None, nproc=1, **kwargs):
        """Calculate the Rayleigh statistic spectrogram of this `TimeSeries`

        Parameters
        ----------
        timeseries : :class:`~gwpy.timeseries.core.TimeSeries`
            input time-series to process.
        stride : `float`
            number of seconds in single PSD (column of spectrogram).
        fftlength : `float`
            number of seconds in single FFT.
        overlap : `int`, optiona, default: fftlength
            number of seconds between FFTs.
        window : `timeseries.window.Window`, optional, default: `None`
            window function to apply to timeseries prior to FFT.
        plan : :lalsuite:`REAL8FFTPlan`, optional
            LAL FFT plan to use when generating average spectrum,
            substitute type 'REAL8' as appropriate.
        nproc : `int`, default: ``1``
            maximum number of independent frame reading processes, default
            is set to single-process file reading.

        Returns
        -------
        spectrogram : :class:`~gwpy.spectrogram.core.Spectrogram`
            time-frequency Rayleigh spectrogram as generated from the
            input time-series.
        """
        rspecgram = self.spectrogram(stride, method='rayleigh',
                                     fftlength=fftlength, overlap=overlap,
                                     window=window, nproc=nproc, **kwargs)
        rspecgram.unit = None
        return rspecgram

    # -------------------------------------------
    # TimeSeries filtering

    def highpass(self, frequency, gpass=2, gstop=30, stop=None):
        """Filter this `TimeSeries` with a Butterworth high-pass filter.

        Parameters
        ----------
        frequency : `float`
            minimum frequency for high-pass
        gpass : `float`
            the maximum loss in the passband (dB).
        gstop : `float`
            the minimum attenuation in the stopband (dB).
        stop : `float`
            stop-band edge frequency, defaults to `frequency/2`

        Returns
        -------
        hpseries : `TimeSeries`
            a high-passed version of the input `TimeSeries`

        See Also
        --------
        scipy.signal.buttord
        scipy.signal.butter
            for details on how the filter is designed
        TimeSeries.filter
            for details on how the filter is applied

        .. note::

           When using `scipy < 0.16.0` some higher-order filters may be
           unstable. With `scipy >= 0.16.0` higher-order filters are
           decomposed into second-order-sections, and so are much more stable.
        """

        nyq = self.sample_rate.value / 2.
        if stop is None:
            stop = .5 * frequency
        # convert to float in Hertz
        cutoff = units.Quantity(frequency, 'Hz').value / nyq
        stop = units.Quantity(stop, 'Hz').value / nyq
        # design filter
        order, wn = signal.buttord(wp=cutoff, ws=stop, gpass=gpass,
                                   gstop=gstop, analog=False)
        zpk = signal.butter(order, wn, btype='high',
                            analog=False, output='zpk')
        # apply filter
        return self.filter(*zpk)

    def lowpass(self, frequency, gpass=2, gstop=30, stop=None):
        """Filter this `TimeSeries` with a Butterworth low-pass filter.

        Parameters
        ----------
        frequency : `float`
            low-pass corner frequency
        gpass : `float`
            the maximum loss in the passband (dB).
        gstop : `float`
            the minimum attenuation in the stopband (dB).
        stop: `float`
            stop-band edge frequency, defaults to `frequency * 1.5`

        Returns
        -------
        lpseries : `TimeSeries`
            a low-passed version of the input `TimeSeries`

        See Also
        --------
        scipy.signal.buttord
        scipy.signal.butter
            for details on how the filter is designed
        TimeSeries.filter
            for details on how the filter is applied

        .. note::

           When using `scipy < 0.16.0` some higher-order filters may be
           unstable. With `scipy >= 0.16.0` higher-order filters are
           decomposed into second-order-sections, and so are much more stable.
        """
        nyq = self.sample_rate.value / 2.
        if stop is None:
            stop = 1.5 * frequency
        # convert to float in Hertz
        cutoff = units.Quantity(frequency, 'Hz').value / nyq
        stop = units.Quantity(stop, 'Hz').value / nyq
        # design filter
        order, wn = signal.buttord(wp=cutoff, ws=stop, gpass=gpass,
                                   gstop=gstop, analog=False)
        zpk = signal.butter(order, wn, btype='low', analog=False, output='zpk')
        # apply filter
        return self.filter(*zpk)

    def bandpass(self, flow, fhigh, gpass=2, gstop=30, stops=(None, None)):
        """Filter this `TimeSeries` by applying low- and high-pass filters.

        Parameters
        ----------
        flow : `float`
            band-pass lower corner frequency
        fhigh : `float`
            band-pass upper corner frequency
        gpass : `float`
            the maximum loss in the pass band (dB).
        gstop : `float`
            the minimum attenuation in the stop band (dB).
        stops: 2-`tuple` of `float`
            stop-band edge frequencies, defaults to `[flow/2., fhigh*1.5]`

        Returns
        -------
        bpseries : `TimeSeries`
            a band-passed version of the input `TimeSeries`

        See Also
        --------
        scipy.signal.buttord
        scipy.signal.butter
            for details on how the filter is designed
        TimeSeries.filter
            for details on how the filter is applied

        .. note::

           When using `scipy < 0.16.0` some higher-order filters may be
           unstable. With `scipy >= 0.16.0` higher-order filters are
           decomposed into second-order-sections, and so are much more stable.
        """
        nyq = self.sample_rate.value / 2.
        if stops is None:
            stops = [None, None]
        stops = list(stops)
        if stops[0] is None:
            stops[0] = flow * 0.5
        if stops[1] is None:
            stops[1] = fhigh * 1.5
        # make sure all are in Hertz
        low = units.Quantity(flow, 'Hz').value / nyq
        high = units.Quantity(fhigh, 'Hz').value / nyq
        stops = [units.Quantity(s, 'Hz').value / nyq for s in stops]
        # design filter
        order, wn = signal.buttord(wp=[low, high], ws=stops, gpass=gpass,
                                   gstop=gstop, analog=False)
        zpk = signal.butter(order, wn, btype='band',
                            analog=False, output='zpk')
        # apply filter
        return self.filter(*zpk)

    def resample(self, rate, window='hamming', numtaps=61):
        """Resample this Series to a new rate

        Parameters
        ----------
        rate : `float`
            rate to which to resample this `Series`
        window : array_like, callable, string, float, or tuple, optional
            specifies the window applied to the signal in the Fourier
            domain.
        numtaps : `int`, default: ``61``
            length of the filter (number of coefficients, i.e. the filter
            order + 1). This option is only valid for an integer-scale
            downsampling.

        Returns
        -------
        Series
            a new Series with the resampling applied, and the same
            metadata
        """
        if isinstance(rate, units.Quantity):
            rate = rate.value
        factor = (self.sample_rate.value / rate)
        # if integer down-sampling, use decimate
        if factor.is_integer():
            factor = int(factor)
            new = signal.decimate(self.data, factor, numtaps-1,
                                  ftype='fir').view(self.__class__)
        # otherwise use Fourier filtering
        else:
            nsamp = int(self.shape[0] * self.dx.value * rate)
            new = signal.resample(self.data, nsamp,
                                  window=window).view(self.__class__)
        new.metadata = self.metadata.copy()
        new.sample_rate = rate
        return new

    def zpk(self, zeros, poles, gain, digital=False, unit='Hz'):
        """Filter this `TimeSeries` by applying a zero-pole-gain filter

        Parameters
        ----------
        zeros : `array-like`
            list of zero frequencies
        poles : `array-like`
            list of pole frequencies
        gain : `float`
            DC gain of filter
        digital : `bool`, optional, default: `False`
            give `True` if zeros, poles, and gain are already in Z-domain
            digital format, otherwise they will be converted
        unit : `str`, `~astropy.units.Unit`, optional, default: `'Hz'`
            unit of zeros and poles, either 'Hz' or 'rad/s'

        Returns
        -------
        timeseries : `TimeSeries`
            the filtered version of the input data

        See Also
        --------
        TimeSeries.filter
            for details on how a digital ZPK-format filter is applied

        Examples
        --------
        To apply a zpk filter with file poles at 100 Hz, and five zeros at
        1 Hz (giving an overall DC gain of 1e-10)::

            >>> data2 = data.zpk([100]*5, [1]*5, 1e-10)
        """
        if not digital:
            # cast to arrays for ease
            z = numpy.array(zeros, dtype=float)
            p = numpy.array(poles, dtype=float)
            k = float(gain)
            # convert from Hz to rad/s if needed
            unit = units.Unit(unit)
            if unit == units.Unit('Hz'):
                z *= -2 * pi
                p *= -2 * pi
            elif unit != units.Unit('rad/s'):
                raise ValueError("zpk can only be given with unit='Hz' "
                                 "or 'rad/s'")
            # convert to Z-domain via bilinear transform
            fs = 2 * self.sample_rate.to('Hz').value
            z = z[numpy.isfinite(z)]
            pd = (1 + p/fs) / (1 - p/fs)
            zd = (1 + z/fs) / (1 - z/fs)
            kd = k * numpy.prod(fs - z)/numpy.prod(fs - p)
            zd = numpy.concatenate((zd, -numpy.ones(len(pd)-len(zd))))
            zeros, poles, gain = zd, pd, kd
        # apply filter
        return self.filter(zeros, poles, gain)

    def filter(self, *filt):
        """Apply the given filter to this `TimeSeries`.

        All recognised filter arguments are converted either into cascading
        second-order sections (if scipy >= 0.16 is installed), or into the
        ``(numerator, denominator)`` representation before being applied
        to this `TimeSeries`.

        .. note::

           All filters are presumed to be digital (Z-domain), if you have
           an analog ZPK (in Hertz or in rad/s) you should be using
           `TimeSeries.zpk` instead.

        .. note::

           When using `scipy` < 0.16 some higher-order filters may be
           unstable. With `scipy` >= 0.16 higher-order filters are
           decomposed into second-order-sections, and so are much more stable.

        Parameters
        ----------
        *filt
            one of:

            - :class:`scipy.signal.lti`
            - `MxN` `numpy.ndarray` of second-order-sections
              (`scipy` >= 0.16 only)
            - ``(numerator, denominator)`` polynomials
            - ``(zeros, poles, gain)``
            - ``(A, B, C, D)`` 'state-space' representation

        Returns
        -------
        result : `TimeSeries`
            the filtered version of the input `TimeSeries`

        See also
        --------
        TimeSeries.zpk
            for instructions on how to filter using a ZPK with frequencies
            in Hertz
        scipy.signal.sosfilter
            for details on the second-order section filtering method
            (`scipy` >= 0.16 only)
        scipy.signal.lfilter
            for details on the filtering method

        Raises
        ------
        ValueError
            If ``filt`` arguments cannot be interpreted properly
        """
        sos = None
        # single argument given
        if len(filt) == 1:
            filt = filt[0]
            # detect LTI
            if isinstance(filt, signal.lti):
                filt = filt
                a = filt.den
                b = filt.num
            # detect SOS
            elif isinstance(filt, numpy.ndarray) and filt.ndim == 2:
                sos = filt
            # detect taps
            else:
                b = filt
                a = [1]
        # detect TF
        elif len(filt) == 2:
            b, a = filt
        elif len(filt) == 3:
            try:
                sos = signal.zpk2sos(*filt)
            except AttributeError:
                b, a = signal.zpk2tf(*filt)
        elif len(filt) == 4:
            try:
                zpk = signal.ss2zpk(*filt)
                sos = signal.zpk2sos(zpk)
            except AttributeError:
                b, a = signal.ss2tf(*filt)
        else:
            raise ValueError("Cannot interpret filter arguments. Please "
                             "give either a signal.lti object, or a "
                             "tuple in zpk or ba format. See "
                             "scipy.signal docs for details.")
        if sos is not None:
            new = signal.sosfilt(sos, self, axis=0).view(self.__class__)
        else:
            new = signal.lfilter(b, a, self, axis=0).view(self.__class__)
        new.metadata = self.metadata.copy()
        return new

    def coherence(self, other, fftlength=None, overlap=None,
                  window=None, **kwargs):
        """Calculate the frequency-coherence between this `TimeSeries`
        and another.

        Parameters
        ----------
        other : `TimeSeries`
            `TimeSeries` signal to calculate coherence with
        fftlength : `float`, optional, default: `TimeSeries.duration`
            number of seconds in single FFT, defaults to a single FFT
        overlap : `float`, optional, default: `None`
            number of seconds of overlap between FFTs, defaults to no
            overlap
        window : `timeseries.window.Window`, optional, default: `HanningWindow`
            window function to apply to timeseries prior to FFT,
            default HanningWindow of the relevant size
        **kwargs
            any other keyword arguments accepted by
            :func:`matplotlib.mlab.cohere` except ``NFFT``, ``window``,
            and ``noverlap`` which are superceded by the above keyword
            arguments

        Returns
        -------
        coherence : :class:`~gwpy.spectrum.core.Spectrum`
            the coherence `Spectrum` of this `TimeSeries` with the other

        Notes
        -----
        If `self` and `other` have difference
        :attr:`TimeSeries.sample_rate` values, the higher sampled
        `TimeSeries` will be down-sampled to match the lower.

        See Also
        --------
        :func:`matplotlib.mlab.cohere`
            for details of the coherence calculator
        """
        from ..spectrum import Spectrum
        # check sampling rates
        if self.sample_rate.to('Hertz') != other.sample_rate.to('Hertz'):
            sampling = min(self.sample_rate.value, other.sample_rate.value)
            # resample higher rate series
            if self.sample_rate.value == sampling:
                other = other.resample(sampling)
                self_ = self
            else:
                self_ = self.resample(sampling)
        else:
            sampling = self.sample_rate.value
            self_ = self
        # check fft lengths
        if overlap is None:
            overlap = 0
        else:
            overlap = int((overlap * self_.sample_rate).decompose().value)
        if fftlength is None:
            fftlength = int(self_.size/2. + overlap/2.)
        else:
            fftlength = int((fftlength * self_.sample_rate).decompose().value)
        if window is not None:
            kwargs['window'] = window
        coh, f = mlab.cohere(self_.data, other.data, NFFT=fftlength,
                             Fs=sampling, noverlap=overlap, **kwargs)
        out = coh.view(Spectrum)
        out.f0 = f[0]
        out.df = (f[1] - f[0])
        out.epoch = self.epoch
        out.name = 'Coherence between %s and %s' % (self.name, other.name)
        return out

    def auto_coherence(self, dt, fftlength=None, overlap=None,
                       window=None, **kwargs):
        """Calculate the frequency-coherence between this `TimeSeries`
        and a time-shifted copy of itself.

        The standard :meth:`TimeSeries.coherence` is calculated between
        the input `TimeSeries` and a :meth:`cropped <TimeSeries.crop>`
        copy of itself. Since the cropped version will be shorter, the
        input series will be shortened to match.

        Parameters
        ----------
        dt : `float`
            duration (in seconds) of time-shift
        fftlength : `float`, optional, default: `TimeSeries.duration`
            number of seconds in single FFT, defaults to a single FFT
        overlap : `int`, optiona, default: fftlength
            number of seconds of overlap between FFTs, defaults to no
            overlap
        window : `timeseries.window.Window`, optional, default: `HanningWindow`
            window function to apply to timeseries prior to FFT,
            default HanningWindow of the relevant size
        **kwargs
            any other keyword arguments accepted by
            :func:`matplotlib.mlab.cohere` except ``NFFT``, ``window``,
            and ``noverlap`` which are superceded by the above keyword
            arguments

        Returns
        -------
        coherence : :class:`~gwpy.spectrum.core.Spectrum`
            the coherence `Spectrum` of this `TimeSeries` with the other

        Notes
        -----
        The :meth:`TimeSeries.auto_coherence` will perform best when
        ``dt`` is approximately ``fftlength / 2``.

        See Also
        --------
        :func:`matplotlib.mlab.cohere`
            for details of the coherence calculator
        """
        # shifting self backwards is the same as forwards
        dt = abs(dt)
        # crop inputs
        self_ = self.crop(self.span[0], self.span[1] - dt)
        other = self.crop(self.span[0] + dt, self.span[1])
        return self_.coherence(other, fftlength=fftlength,
                               overlap=overlap, window=window, **kwargs)

    def coherence_spectrogram(self, other, stride, fftlength=None,
                              overlap=None, window=None, nproc=1):
        """Calculate the coherence spectrogram between this `TimeSeries`
        and other.

        Parameters
        ----------
        stride : `float`
            number of seconds in single PSD (column of spectrogram)
        fftlength : `float`
            number of seconds in single FFT
        overlap : `int`, optiona, default: fftlength
            number of seconds of overlap between FFTs, defaults to no
            overlap
        window : `timeseries.window.Window`, optional, default: `None`
            window function to apply to timeseries prior to FFT
        nproc : `int`, default: ``1``
            number of parallel processes to use when calculating
            individual coherence spectra.

        Returns
        -------
        spectrogram : :class:`~gwpy.spectrogram.core.Spectrogram`
            time-frequency coherence spectrogram as generated from the
            input time-series.
        """
        from ..spectrogram.coherence import from_timeseries
        return from_timeseries(self, other, stride, fftlength=fftlength,
                               overlap=overlap, window=window,
                               nproc=nproc)

    def rms(self, stride=1):
        """Calculate the root-mean-square value of this `TimeSeries`
        once per stride.

        Parameters
        ----------
        stride : `float`
            stride (seconds) between RMS calculations

        Returns
        -------
        rms : `TimeSeries`
            a new `TimeSeries` containing the RMS value with dt=stride
        """
        stridesamp = stride * self.sample_rate.value
        nsteps = int(self.size // stridesamp)
        # stride through TimeSeries, recording RMS
        data = numpy.zeros(nsteps)
        for step in range(nsteps):
            # find step TimeSeries
            idx = stridesamp * step
            idx_end = idx + stridesamp
            stepseries = self[idx:idx_end]
            rms_ = numpy.sqrt(numpy.mean(numpy.abs(stepseries.data)**2))
            data[step] = rms_
        name = '%s %.2f-second RMS' % (self.name, stride)
        return self.__class__(data, channel=self.channel, epoch=self.epoch,
                              name=name, sample_rate=(1/float(stride)))

    # -------------------------------------------
    # connectors

    def is_compatible(self, other):
        """Check whether metadata attributes for self and other match.
        """
        if isinstance(other, type(self)):
            if not self.sample_rate == other.sample_rate:
                raise ValueError("TimeSeries sampling rates do not match: "
                                 "%s vs %s." % (self.sample_rate,
                                                other.sample_rate))
            if not self.unit == other.unit:
                raise ValueError("TimeSeries units do not match: %s vs %s."
                                 % (str(self.unit), str(other.unit)))
        else:
            arr = numpy.asarray(other)
            if arr.ndim != self.ndim:
                raise ValueError("Dimensionality does not match")
            if arr.dtype != self.dtype:
                warnings.warn("dtype mismatch: %s vs %s"
                              % (self.dtype, other.dtype))
        return True

    # -------------------------------------------
    # Common operations

    crop = common.crop
    is_contiguous = common.is_contiguous
    append = common.append
    prepend = common.prepend
    update = common.update

    # -------------------------------------------
    # Utilities

    def pad(self, pad_width, **kwargs):
        """Pad this `TimeSeries`.

        Parameters
        ----------
        pad_width : `int`, pair of `ints`
            number of samples by which to pad each end of the array.
            Single int to pad both ends by the same amount, or
            (before, after) `tuple` to give uneven padding
        **kwargs
            see :meth:`numpy.pad` for kwarg documentation

        Returns
        -------
        t2 : `TimeSeries`
            the padded version of the input

        See also
        --------
        numpy.pad
            for details on the underlying functionality
        """
        kwargs.setdefault('mode', 'constant')
        if isinstance(pad_width, int):
            pad_width = (pad_width,)
        new = numpy.pad(self.data, pad_width, **kwargs).view(self.__class__)
        new.metadata = self.metadata.copy()
        new.epoch = self.epoch.gps - self.dt.value * pad_width[0]
        return new

    def plot(self, **kwargs):
        """Plot the data for this TimeSeries.
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
    def from_lal(cls, lalts):
        """Generate a new TimeSeries from a LAL TimeSeries of any type.
        """
        from ..utils.lal import from_lal_unit
        try:
            unit = from_lal_unit(lalts.sampleUnits)
        except TypeError:
            unit = None
        channel = Channel(lalts.name, 1/lalts.deltaT, unit=unit,
                          dtype=lalts.data.data.dtype)
        return cls(lalts.data.data, channel=channel, epoch=lalts.epoch,
                   copy=True)

    @with_import('lal.lal')
    def to_lal(self):
        """Convert this `TimeSeries` into a LAL TimeSeries.
        """
        from ..utils.lal import (LAL_TYPE_STR_FROM_NUMPY, to_lal_unit)
        typestr = LAL_TYPE_STR_FROM_NUMPY[self.dtype.type]
        try:
            unit = to_lal_unit(self.unit)
        except TypeError:
            try:
                unit = lal.DimensionlessUnit
            except AttributeError:
                unit = lal.lalDimensionlessUnit
        create = getattr(lal, 'Create%sTimeSeries' % typestr.upper())
        lalts = create(self.name, lal.LIGOTimeGPS(self.epoch.gps), 0,
                       self.dt.value, unit, self.size)
        lalts.data.data = self.data
        return lalts

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
            result.metadata = self.metadata.copy()
            result.unit = ""
            result.name = '%s %s %s' % (obj.name, op_, value)
            if hasattr(obj, 'unit') and str(obj.unit):
                result.name += ' %s' % str(obj.unit)
        else:
            result = super(TimeSeries, self).__array_wrap__(obj,
                                                            context=context)
        return result


class ArrayTimeSeries(TimeSeries, Array2D):
    xunit = TimeSeries.xunit

    def __new__(cls, data, times=None, epoch=None, channel=None, unit=None,
                sample_rate=None, name=None, **kwargs):
        """Generate a new ArrayTimeSeries.
        """
        # parse Channel input
        if channel:
            channel = (isinstance(channel, Channel) and channel or
                       Channel(channel))
            name = name or channel.name
            unit = unit or channel.unit
            sample_rate = sample_rate or channel.sample_rate
        # generate TimeSeries
        new = Array2D.__new__(cls, data, name=name, unit=unit, epoch=epoch,
                              channel=channel, sample_rate=sample_rate,
                              times=times, **kwargs)
        return new


class TimeSeriesList(list):
    """Fancy list representing a list of `TimeSeries`

    The `TimeSeriesList` provides an easy way to collect and organise
    `TimeSeries` for a single `Channel` over multiple segments.

    Parameters
    ----------
    *items
        any number of `TimeSeries`

    Returns
    -------
    list
        a new `TimeSeriesList`

    Raises
    ------
    TypeError
        if any elements are not `TimeSeries`
    """
    EntryClass = TimeSeries

    def __init__(self, *items):
        """Initalise a new `TimeSeriesList`
        """
        super(TimeSeriesList, self).__init__()
        for item in items:
            self.append(item)

    @property
    def segments(self):
        return SegmentList([item.span for item in self])

    def append(self, item):
        if not isinstance(item, self.EntryClass):
            raise TypeError("Cannot append type '%s' to %s"
                            % (item.__class__.__name__,
                               self.__class__.__name__))
        super(TimeSeriesList, self).append(item)
        return self
    append.__doc__ = list.append.__doc__

    def extend(self, item):
        item = TimeSeriesList(item)
        super(TimeSeriesList, self).extend(item)
    extend.__doc__ = list.extend.__doc__

    def coalesce(self):
        """Sort the elements of this `TimeSeriesList` by epoch and merge
        contiguous `TimeSeries` elements into single objects.
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
        """Concatenate all of the `TimeSeries` in this list into a
        a single object

        Returns
        -------
        `TimeSeries`
             a single `TimeSeries covering the full span of all entries
             in this list
        """
        if len(self) == 0:
            return self.EntryClass([])
        self.sort(key=lambda t: t.epoch.gps)
        out = self[0].copy()
        for ts in self[1:]:
            out.append(ts, gap=gap, pad=pad)
        return out


class TimeSeriesDict(OrderedDict):
    """Ordered key-value mapping of named `TimeSeries` containing data
    for many channels over the same time interval.

    Currently the class doesn't do anything special. FIXME.
    """
    EntryClass = TimeSeries

    # use input/output registry to allow multi-format reading
    read = classmethod(reader(doc="""
        Read data into a `TimeSeriesDict`.

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

        Returns
        -------
        dict : `TimeSeriesDict`
            a new `TimeSeriesDict` containing data for the given channel.

        Raises
        ------
        Exception
            if no format could be automatically identified.

        Notes
        -----"""))

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
        """Crop each entry of this `TimeSeriesDict`.

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
        data : :class:`~gwpy.timeseries.core.TimeSeriesDict`
            a new `TimeSeriesDict` of (`str`, `TimeSeries`) pairs fetched
            from NDS.
        """
        from ..io import nds as ndsio
        # parse times
        start = to_gps(start)
        end = to_gps(end)
        istart = start.seconds
        iend = ceil(end)

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
            if iend < float(end):
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

    def plot(self, label='key', **kwargs):
        """Plot the data for this `TimeSeriesDict`.

        Parameters
        ----------
        label : `str`, optional
            labelling system to use, or fixed label for all `TimeSeries`.
            Special values include

            - ``'key'``: use the key of the `TimeSeriesDict`,
            - ``'name'``: use the :attr:`~TimeSeries.name` of the `TimeSeries`

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
