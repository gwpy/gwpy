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
from math import (ceil, log)
from dateutil import parser as dateparser
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


from ..data import Array2D
from ..detector import (Channel, ChannelList)
from ..io import reader
from ..io.kerberos import kinit
from ..segments import (Segment, SegmentList)
from ..time import Time
from ..window import *
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
              connection=None, type=NDS2_FETCH_TYPE_MASK):
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
        verbose : `bool`, optional
            print verbose output about NDS progress
        connection : :class:`~gwpy.io.nds.NDS2Connection`
            open NDS connection to use
        type : `int`
            NDS2 channel type integer

        Returns
        -------
        TimeSeries
            a new `TimeSeries` containing the data read from NDS
        """
        return TimeSeriesDict.fetch(
                   [channel], start, end, host=host, port=port,
                   verbose=verbose, connection=connection,
                   type=type)[str(channel)]

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
            ffts[i] = fft_.data
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
                out[step] = steppsd.data

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

    # -------------------------------------------
    # TimeSeries filtering

    def highpass(self, frequency, numtaps=101, window='hamming'):
        """Filter this `TimeSeries` with a Butterworth high-pass filter.

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
        filter_ = signal.firwin(numtaps, frequency, window=window,
                                nyq=self.sample_rate.value/2.,
                                pass_zero=False)
        return self.filter(*filter_)

    def lowpass(self, frequency, numtaps=61, window='hamming'):
        """Filter this `TimeSeries` with a Butterworth low-pass filter.

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
        filter_ = signal.firwin(numtaps, frequency, window=window,
                                nyq=self.sample_rate.value/2.)
        return self.filter(*filter_)

    def bandpass(self, flow, fhigh, lowtaps=61, hightaps=101,
                 window='hamming'):
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
        high = self.highpass(flow, numtaps=lowtaps, window=window)
        return high.lowpass(fhigh, numtaps=hightaps, window=window)

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

    def filter(self, *filt):
        """Apply the given filter to this `TimeSeries`.

        Recognised filter arguments are converted into the standard
        ``(numerator, denominator)`` representation before being applied
        to this `TimeSeries`.

        Parameters
        ----------
        *filt
            one of:

            - :class:`scipy.signal.lti`
            - ``(numerator, denominator)`` polynomials
            - ``(zeros, poles, gain)``
            - ``(A, B, C, D)`` 'state-space' representation

        Returns
        -------
        result : `TimeSeries`
            the filtered version of the input `TimeSeries`

        See also
        --------
        scipy.signal.zpk2tf
            for details on converting ``(zeros, poles, gain)`` into
            transfer function format
        scipy.signal.ss2tf
            for details on converting ``(A, B, C, D)`` to transfer function
            format
        scipy.signal.lfilter
            for details on the filtering method

        Examples
        --------
        To apply a zpk filter with a pole at 0 Hz, a zero at 100 Hz and
        a gain of 25::

            >>> data2 = data.filter([100], [0], 25)

        Raises
        ------
        ValueError
            If ``filt`` arguments cannot be interpreted properly
        """
        if len(filt) == 1 and isinstance(filt, signal.lti):
            filt = filt[0]
            a = filt.den
            b = filt.num
        elif len(filt) == 2:
            b, a = filt
        elif len(filt) == 3:
            b, a = signal.zpk2tf(*filt)
        elif len(filt) == 4:
            b, a = signal.ss2tf(*filt)
        else:
            try:
                b = numpy.asarray(filt)
                assert b.ndim == 1
            except (ValueError, AssertionError):
                raise ValueError("Cannot interpret filter arguments. Please "
                                 "give either a signal.lti object, or a "
                                 "tuple in zpk or ba format. See "
                                 "scipy.signal docs for details.")
            else:
                a = 1.0
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
        overlap : `int`, optiona, default: fftlength
            number of seconds between FFTs, defaults to no overlap
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
        if fftlength is None:
            fftlength = self_.duration.value
        if overlap is None:
            overlap = fftlength
        fftlength = int(numpy.float64(fftlength) * self_.sample_rate.value)
        overlap = int(numpy.float64(overlap) * self_.sample_rate.value)
        if window is None:
            window = HanningWindow(fftlength)
        coh, f = mlab.cohere(self_.data, other.data, NFFT=fftlength,
                             Fs=sampling, window=window,
                             noverlap=fftlength-overlap, **kwargs)
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
            number of seconds between FFTs, defaults to no overlap
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
            number of seconds between FFTs
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
        if not self.sample_rate == other.sample_rate:
            raise ValueError("TimeSeries sampling rates do not match: "
                             "%s vs %s." % (self.sample_rate,
                                            other.sample_rate))
        if not self.unit == other.unit:
            raise ValueError("TimeSeries units do not match: %s vs %s."
                             % (str(self.unit), str(other.unit)))
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

    def plot(self, **kwargs):
        """Plot the data for this TimeSeries.
        """
        from ..plotter import TimeSeriesPlot
        return TimeSeriesPlot(self, **kwargs)

    @classmethod
    @with_import('nds2')
    def from_nds2_buffer(cls, buffer_):
        """Construct a new `TimeSeries` from an `nds2.buffer` object
        """
        # cast as TimeSeries and return
        epoch = Time(buffer_.gps_seconds, buffer_.gps_nanoseconds,
                     format='gps')
        channel = Channel.from_nds2(buffer_.channel)
        return cls(buffer_.data, epoch=epoch, channel=channel)

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

    @with_import('lal')
    def to_lal(self):
        """Convert this `TimeSeries` into a LAL TimeSeries.
        """
        from ..utils.lal import (LAL_TYPE_STR_FROM_NUMPY, to_lal_unit)
        typestr = LAL_TYPE_STR_FROM_NUMPY[self.dtype.type]
        try:
            unit = to_lal_unit(self.unit)
        except TypeError:
            unit = lal.lalDimensionlessUnit
        create = getattr(lal, 'Create%sTimeSeries' % typestr.upper())
        lalts = create(self.name, lal.LIGOTimeGPS(self.epoch.gps), 0,
                       self.dt.value, unit, self.size)
        lalts.data.data = self.data
        return lalts

    #@staticmethod
    #def watch(channel, duration, stride=1, host=None, port=None,
    #          outputfile=None, verbose=False):
    #    """Stream data over NDS2, with an updating display
    #    """
    #    from ..io import nds
    #    # user-defined host
    #    if host:
    #        hostlist = [(host, port)]
    #    else:
    #        hostlist = nds.host_resolution_order(channel.ifo)
    #    for host,port in hostlist:
    #        # connect and find channel
    #        if verbose:
    #            print("Connecting to %s:%s" % (host, port))
    #        connection = nds.NDS2Connection(host, port)
    #        try:
    #            channel = connection.find(
    #                      str(channel), nds.nds2.channel.CHANNEL_TYPE_ONLINE)[0]
    #        except IndexError:
    #            continue
    #        # begin iteration
    #        sampling = channel.sample_rate
    #        timeseries = TimeSeries(numpy.zeros((sampling*duration,)),
    #                                channel=Channel.from_nds2(channel))
    #        timeseries.epoch = gpstime.gps_time_now()-duration
    #        plot = timeseries.plot(auto_refresh=True)
    #        trace = plot._layers[timeseries.name]
    #        try:
    #            for buf in connection.iterate([channel.name]):
    #                newdata = buf[0]
    #                epoch = lal.LIGOTimeGPS(newdata.gps_seconds,
    #                                        newdata.gps_nanoseconds)
    #                # shift
    #                timeseries.epoch = epoch + stride - duration
    #                timeseries.data[:-newdata.length] = (
    #                    timeseries.data[newdata.length:])
    #                timeseries.data[-newdata.length:] = newdata.data
    #                trace.set_xdata(timeseries.times.data)
    #                trace.set_ydata(timeseries.data)
    #                plot.epoch = epoch
    #                del plot.ylim
    #                plot.xlim = timeseries.span
    #        except KeyboardInterrupt:
    #            return

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

    def __add__(self, item):
        return self.append(item, inplace=False)

    def __iadd__(self, item):
        return self.append(item, inplace=True)


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
            if j < N and this.span[1] >= self[j].span[0]:
                while j < N and this.span[1] >= self[j].span[0]:
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

    def join(self, pad=0.0, gap='pad'):
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
              verbose=False, connection=None, type=NDS2_FETCH_TYPE_MASK):
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
        verbose : `bool`, optional
            print verbose output about NDS progress.
        connection : :class:`~gwpy.io.nds.NDS2Connection`
            open NDS connection to use.
        type : `int`, `str`,
            NDS2 channel type integer or string name.

        Returns
        -------
        data : :class:`~gwpy.timeseries.core.TimeSeriesDict`
            a new `TimeSeriesDict` of (`str`, `TimeSeries`) pairs fetched
            from NDS.
        """
        from ..io import nds as ndsio
        # import module and type-cast arguments
        if isinstance(start, Time):
            start = start.gps
        elif isinstance(start, (unicode, str)):
            try:
                start = float(start)
            except ValueError:
                d = dateparser.parse(start)
                start = float(str(Time(d, scale='utc').gps))
        start = int(float(start))
        if isinstance(end, Time):
            end = end.gps
        elif isinstance(end, (unicode, str)):
            try:
                end = float(end)
            except ValueError:
                d = dateparser.parse(end)
                end = float(str(Time(d, scale='utc').gps))
        end = int(ceil(end))

        # open connection for specific host
        if host and not port and re.match('[a-z]1nds0\Z', host):
            port = 8088
        elif host and not port:
            port = 31200
        if host is not None and port is not None and connection is None:
            if verbose:
                gprint("Connecting to %s:%s..." % (host, port), end=' ')
            try:
                connection = nds2.connection(host, port)
            except RuntimeError as e:
                if str(e).startswith('Request SASL authentication'):
                    gprint('\nError authenticating against %s' % host,
                           file=sys.stderr)
                    kinit()
                    connection = nds2.connection(host, port)
                else:
                    raise
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
            hostlist = ndsio.host_resolution_order(ifo)
            for host, port in hostlist:
                try:
                    return cls.fetch(channels, start, end, host=host,
                                     port=port, verbose=verbose, type=type)
                except (RuntimeError, ValueError) as e:
                    gprint('Something went wrong:', file=sys.stderr)
                    # if error and user supplied their own server, raise
                    warnings.warn(str(e), ndsio.NDSWarning)

            # if we got this far, we can't get all of the channels in one go
            if len(channels) > 1:
                return cls(
                    (c, cls.EntryClass.fetch(c, start, end, verbose=verbose,
                                             type=type)) for c in channels)
            e = "Cannot find all relevant data on any known server."
            if not verbose:
                e += (" Try again using the verbose=True keyword argument to "
                      "see detailed failures.")
            raise RuntimeError(e)

        # at this point we must have an open connection, so we can proceed
        # normally

        # verify channels
        qchannels = []
        if verbose:
            gprint("Checking channels against the NDS database...", end=' ')
        for channel in channels:
            if (type and
                    (isinstance(type, (unicode, str)) or
                    (isinstance(type, int) and log(type, 2).is_integer()))):
                c = Channel(channel, type=type)
            else:
                c = Channel(channel)
            if c.ndstype is not None:
                found = connection.find_channels(c.ndsname, c.ndstype)
            elif type is not None:
                found = connection.find_channels('%s*' % c.name, type)
            else:
                found = connection.find_channels('%s*' % c.name)
            # sieve out multiple channels with same type and different
            # sample rates
            funiq = ChannelList()
            for nds2channel in found:
                channel = Channel.from_nds2(nds2channel)
                known = funiq.sieve(name=channel.name, type=channel.type)
                if len(known) >= 1:
                    continue
                else:
                    funiq.append(channel)
            if len(funiq) == 0:
                raise ValueError("Channel '%s' not found" % c.name)
            elif len(funiq) > 1:
                raise ValueError(
                    "Multiple matches for channel '%s' in NDS database, "
                    "ambiguous request:\n    %s"
                    % (c.name, '\n    '.join(['%s (%s, %s)'
                        % (str(c), c.type,
                           c.channel_type_to_string(c.channel_type))
                           for c in found])))
            else:
                qchannels.append(funiq[0])
        if verbose:
            gprint("Complete.")

        # test for minute trends
        if (any([c.type == 'm-trend' for c in qchannels]) and
                (start % 60 or end % 60)):
            warnings.warn("Requested at least one minute trend, but "
                          "start and stop GPS times are not modulo "
                          "60-seconds (from GPS epoch). Times will be "
                          "expanded outwards to compensate")
            if start % 60:
                start = start // 60 * 60
            if end % 60:
                end = end // 60 * 60 + 60

        # fetch data
        if verbose:
            gprint('Downloading data...', end=' ')
        out = cls()
        try:
            data = [connection.fetch(start, end,
                                     [c.ndsname for c in qchannels])]
        except RuntimeError as e:
            # XXX: hack to fix potential problem with fetch
            # Can remove once fetch has been patched (DMM, 02/2014)
            if ('Server sent more data than were expected' in str(e)
                    and len(qchannels) == 1
                    and qchannels[0].name.endswith(',rds')):
                c2 = nds2.connection(connection.get_host(),
                                     connection.get_port())
                data = c2.iterate(start, end, 60,
                                  [c.ndsname for c in qchannels])
            else:
                raise
        for buffers in data:
            for buffer_, c in zip(buffers, channels):
                out.append({c: cls.EntryClass.from_nds2_buffer(buffer_)})
        if verbose:
            gprint('Success.')
        return out
