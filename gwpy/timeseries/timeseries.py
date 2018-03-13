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

from warnings import warn

from six.moves import range

import numpy
from numpy import fft as npfft
from scipy import signal

from astropy import units

from ..segments import Segment
from ..signal import filter_design
from ..signal.filter import sosfiltfilt
from ..signal.fft import (registry as fft_registry, ui as fft_ui)
from ..signal.window import recommended_overlap
from .core import (TimeSeriesBase, TimeSeriesBaseDict, TimeSeriesBaseList,
                   as_series_dict_class)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- utilities ----------------------------------------------------------------

def _update_doc_with_fft_methods(func):
    """Update a function's docstring to append a table of FFT methods

    See `gwpy.signal.fft.registry` for more details
    """
    fft_registry.update_doc(func)
    return func


# -- TimeSeries ---------------------------------------------------------------

class TimeSeries(TimeSeriesBase):
    """A time-domain data array.

    Parameters
    ----------
    value : array-like
        input data array

    unit : `~astropy.units.Unit`, optional
        physical unit of these data

    t0 : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine

    dt : `float`, `~astropy.units.Quantity`, optional
        time between successive samples (seconds), can also be given inversely
        via `sample_rate`

    sample_rate : `float`, `~astropy.units.Quantity`, optional
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

    copy : `bool`, optional
        choose to copy the input data to new memory

    subok : `bool`, optional
        allow passing of sub-classes by the array generator

    Notes
    -----
    The necessary metadata to reconstruct timing information are recorded
    in the `epoch` and `sample_rate` attributes. This time-stamps can be
    returned via the :attr:`~TimeSeries.times` property.

    All comparison operations performed on a `TimeSeries` will return a
    `~gwpy.timeseries.StateTimeSeries` - a boolean array
    with metadata copied from the starting `TimeSeries`.

    Examples
    --------
    >>> from gwpy.timeseries import TimeSeries

    To create an array of random numbers, sampled at 100 Hz, in units of
    'metres':

    >>> from numpy import random
    >>> series = TimeSeries(random.random(1000), sample_rate=100, unit='m')

    which can then be simply visualised via

    >>> plot = series.plot()
    >>> plot.show()
    """
    def fft(self, nfft=None):
        """Compute the one-dimensional discrete Fourier transform of
        this `TimeSeries`.

        Parameters
        ----------
        nfft : `int`, optional
            length of the desired Fourier transform, input will be
            cropped or padded to match the desired length.
            If nfft is not given, the length of the `TimeSeries`
            will be used

        Returns
        -------
        out : `~gwpy.frequencyseries.FrequencySeries`
            the normalised, complex-valued FFT `FrequencySeries`.

        See Also
        --------
        :mod:`scipy.fftpack` for the definition of the DFT and conventions
        used.

        Notes
        -----
        This method, in constrast to the :func:`numpy.fft.rfft` method
        it calls, applies the necessary normalisation such that the
        amplitude of the output `~gwpy.frequencyseries.FrequencySeries` is
        correct.
        """
        from ..frequencyseries import FrequencySeries
        if nfft is None:
            nfft = self.size
        dft = npfft.rfft(self.value, n=nfft) / nfft
        dft[1:] *= 2.0
        new = FrequencySeries(dft, epoch=self.epoch, unit=self.unit,
                              name=self.name, channel=self.channel)
        try:
            new.frequencies = npfft.rfftfreq(nfft, d=self.dx.value)
        except AttributeError:
            new.frequencies = numpy.arange(new.size) / (nfft * self.dx.value)
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

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        Returns
        -------
        out : complex-valued `~gwpy.frequencyseries.FrequencySeries`
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
        if isinstance(window, (str, tuple)):
            win = signal.get_window(window, nfft)
        else:
            win = numpy.asarray(window)
            if len(win.shape) != 1:
                raise ValueError('window must be 1-D')
            elif win.shape[0] != nfft:
                raise ValueError('Window is the wrong size.')
        win = win.astype(self.dtype)
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
            stepseries = self[idx:idx_end].detrend() * win
            # calculated FFT, weight, and stack
            fft_ = stepseries.fft(nfft=nfft) * scaling
            ffts.value[i, :] = fft_.value
            idx += (nfft - noverlap)
        mean = ffts.mean(0)
        mean.name = self.name
        mean.epoch = self.epoch
        mean.channel = self.channel
        return mean

    @_update_doc_with_fft_methods
    def psd(self, fftlength=None, overlap=None, window='hann',
            method='scipy-welch', **kwargs):
        """Calculate the PSD `FrequencySeries` for this `TimeSeries`

        Parameters
        ----------
        fftlength : `float`
            number of seconds in single FFT, defaults to a single FFT
            covering the full duration

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        method : `str`, optional
            FFT-averaging method, default: ``'scipy-welch'``,
            see *Notes* for more details

        **kwargs
            other keyword arguments are passed to the underlying
            PSD-generation method

        Returns
        -------
        psd :  `~gwpy.frequencyseries.FrequencySeries`
            a data series containing the PSD.

        Notes
        -----"""
        # get method
        scaling = kwargs.get('scaling', 'density')
        method_func = fft_registry.get_method(method, scaling=scaling)

        # calculate PSD using UI method
        return fft_ui.psd(self, method_func, fftlength=fftlength,
                          overlap=overlap, window=window, **kwargs)

    @_update_doc_with_fft_methods
    def asd(self, fftlength=None, overlap=None, window='hann',
            method='scipy-welch', **kwargs):
        """Calculate the ASD `FrequencySeries` of this `TimeSeries`

        Parameters
        ----------
        fftlength : `float`
            number of seconds in single FFT, defaults to a single FFT
            covering the full duration

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        method : `str`, optional
            FFT-averaging method, default: ``'scipy-welch'``,
            see *Notes* for more details

        Returns
        -------
        psd :  `~gwpy.frequencyseries.FrequencySeries`
            a data series containing the PSD.

        See also
        --------
        TimeSeries.psd

        Notes
        -----
        The available methods are:

        """
        return self.psd(method=method, fftlength=fftlength, overlap=overlap,
                        window=window, **kwargs) ** (1/2.)

    def csd(self, other, fftlength=None, overlap=None, window='hann',
            **kwargs):
        """Calculate the CSD `FrequencySeries` for two `TimeSeries`

        Parameters
        ----------
        other : `TimeSeries`
            the second `TimeSeries` in this CSD calculation

        fftlength : `float`
            number of seconds in single FFT, defaults to a single FFT
            covering the full duration

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        Returns
        -------
        csd :  `~gwpy.frequencyseries.FrequencySeries`
            a data series containing the CSD.
        """
        # get method
        method_func = fft_registry.get_method('scipy-csd', scaling='other')

        # calculate CSD using UI method
        return fft_ui.psd((self, other), method_func, fftlength=fftlength,
                          overlap=overlap, window=window, **kwargs)

    @_update_doc_with_fft_methods
    def spectrogram(self, stride, fftlength=None, overlap=None,
                    window='hann', method='scipy-welch', nproc=1, **kwargs):
        """Calculate the average power spectrogram of this `TimeSeries`
        using the specified average spectrum method.

        Each time-bin of the output `Spectrogram` is calculated by taking
        a chunk of the `TimeSeries` in the segment
        `[t - overlap/2., t + stride + overlap/2.)` and calculating the
        :meth:`~gwpy.timeseries.TimeSeries.psd` of those data.

        As a result, each time-bin is calculated using `stride + overlap`
        seconds of data.

        Parameters
        ----------
        stride : `float`
            number of seconds in single PSD (column of spectrogram).

        fftlength : `float`
            number of seconds in single FFT.

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        method : `str`, optional
            FFT-averaging method, default: ``'scipy-welch'``,
            see *Notes* for more details

        nproc : `int`
            number of CPUs to use in parallel processing of FFTs

        Returns
        -------
        spectrogram : `~gwpy.spectrogram.Spectrogram`
            time-frequency power spectrogram as generated from the
            input time-series.

        Notes
        -----"""
        # handle deprecated kwargs - TODO: remove before 1.0 release
        try:
            other = kwargs.pop('cross')
        except KeyError:
            pass
        else:
            warn('the `cross` keyword argument has been deprecated, '
                 'please use the csd_spectrogram() method directly, this '
                 'warning will become an error before the 1.0 release',
                 DeprecationWarning)
            return self.csd_spectrogram(other, stride, fftlength=fftlength,
                                        overlap=overlap, window=window,
                                        nproc=nproc, **kwargs)

        # get method
        scaling = kwargs.get('scaling', 'density')
        method_func = fft_registry.get_method(method, scaling=scaling)

        # calculate PSD using UI method
        return fft_ui.average_spectrogram(self, method_func, stride,
                                          fftlength=fftlength, overlap=overlap,
                                          window=window, **kwargs)

    def spectrogram2(self, fftlength, overlap=0, **kwargs):
        """Calculate the non-averaged power `Spectrogram` of this `TimeSeries`

        Parameters
        ----------
        fftlength : `float`
            number of seconds in single FFT.

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

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
        # set kwargs for periodogram()
        kwargs.setdefault('fs', self.sample_rate.to('Hz').value)
        # run
        return fft_ui.spectrogram(self, signal.periodogram,
                                  fftlength=fftlength, overlap=overlap,
                                  **kwargs)

    def fftgram(self, stride):
        """Calculate the Fourier-gram of this `TimeSeries`.

        At every ``stride``, a single, complex FFT is calculated.

        Parameters
        ----------
        stride : `float`
            number of seconds in single PSD (column of spectrogram)

        Returns
        -------
        fftgram : `~gwpy.spectrogram.Spectrogram`
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
                          name=self.name, t0=self.t0, f0=0, df=df,
                          dt=dt, copy=False, unit=self.unit, dtype=dtype)
        # stride through TimeSeries, recording FFTs as columns of Spectrogram
        for step in range(nsteps):
            # find step TimeSeries
            idx = stride * step
            idx_end = idx + stride
            stepseries = self[idx:idx_end]
            # calculated FFT and stack
            stepfft = stepseries.fft()
            out[step] = stepfft.value
            if step == 0:
                out.frequencies = stepfft.frequencies
        return out

    @_update_doc_with_fft_methods
    def spectral_variance(self, stride, fftlength=None, overlap=None,
                          method='scipy-welch', window='hann', nproc=1,
                          filter=None, bins=None, low=None, high=None,
                          nbins=500, log=False, norm=False, density=False):
        """Calculate the `SpectralVariance` of this `TimeSeries`.

        Parameters
        ----------
        stride : `float`
            number of seconds in single PSD (column of spectrogram)

        fftlength : `float`
            number of seconds in single FFT

        method : `str`, optional
            FFT-averaging method, default: ``'scipy-welch'``,
            see *Notes* for more details

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        nproc : `int`
            maximum number of independent frame reading processes, default
            is set to single-process file reading.

        bins : `numpy.ndarray`, optional, default `None`
            array of histogram bin edges, including the rightmost edge

        low : `float`, optional
            left edge of lowest amplitude bin, only read
            if ``bins`` is not given

        high : `float`, optional
            right edge of highest amplitude bin, only read
            if ``bins`` is not given

        nbins : `int`, optional
            number of bins to generate, only read if ``bins`` is not
            given

        log : `bool`, optional
            calculate amplitude bins over a logarithmic scale, only
            read if ``bins`` is not given

        norm : `bool`, optional
            normalise bin counts to a unit sum

        density : `bool`, optional
            normalise bin counts to a unit integral

        Returns
        -------
        specvar : `SpectralVariance`
            2D-array of spectral frequency-amplitude counts

        See Also
        --------
        :func:`numpy.histogram`
            for details on specifying bins and weights

        Notes
        -----"""
        specgram = self.spectrogram(stride, fftlength=fftlength,
                                    overlap=overlap, method=method,
                                    window=window, nproc=nproc) ** (1/2.)
        if filter:
            specgram = specgram.filter(*filter)
        return specgram.variance(bins=bins, low=low, high=high, nbins=nbins,
                                 log=log, norm=norm, density=density)

    def rayleigh_spectrum(self, fftlength=None, overlap=None):
        """Calculate the Rayleigh `FrequencySeries` for this `TimeSeries`.

        Parameters
        ----------
        fftlength : `float`
            number of seconds in single FFT, defaults to a single FFT
            covering the full duration

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to that of
            the relevant method.

        Returns
        -------
        psd :  `~gwpy.frequencyseries.FrequencySeries`
            a data series containing the PSD.
        """
        method_func = fft_registry.get_method('scipy-rayleigh',
                                              scaling='other')
        return fft_ui.psd(self, method_func, fftlength=fftlength,
                          overlap=overlap)

    def rayleigh_spectrogram(self, stride, fftlength=None, overlap=0,
                             nproc=1, **kwargs):
        """Calculate the Rayleigh statistic spectrogram of this `TimeSeries`

        Parameters
        ----------
        stride : `float`
            number of seconds in single PSD (column of spectrogram).

        fftlength : `float`
            number of seconds in single FFT.

        overlap : `float`, optional
            number of seconds of overlap between FFTs, default: ``0``

        nproc : `int`, optional
            maximum number of independent frame reading processes, default
            default: ``1``

        Returns
        -------
        spectrogram : `~gwpy.spectrogram.Spectrogram`
            time-frequency Rayleigh spectrogram as generated from the
            input time-series.
        """
        method_func = fft_registry.get_method('scipy-rayleigh',
                                              scaling='other')
        specgram = fft_ui.average_spectrogram(self, method_func, stride,
                                              fftlength=fftlength,
                                              overlap=overlap, nproc=nproc,
                                              **kwargs)
        specgram.override_unit('')
        return specgram

    def csd_spectrogram(self, other, stride, fftlength=None, overlap=0,
                        window='hann', nproc=1, **kwargs):
        """Calculate the cross spectral density spectrogram of this
           `TimeSeries` with 'other'.

        Parameters
        ----------
        other : `~gwpy.timeseries.TimeSeries`
            second time-series for cross spectral density calculation

        stride : `float`
            number of seconds in single PSD (column of spectrogram).

        fftlength : `float`
            number of seconds in single FFT.

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        nproc : `int`
            maximum number of independent frame reading processes, default
            is set to single-process file reading.

        Returns
        -------
        spectrogram : `~gwpy.spectrogram.Spectrogram`
            time-frequency cross spectrogram as generated from the
            two input time-series.
        """
        method_func = fft_registry.get_method('scipy-csd', scaling='other')
        specgram = fft_ui.average_spectrogram((self, other), method_func,
                                              stride, fftlength=fftlength,
                                              overlap=overlap, window=window,
                                              nproc=nproc, **kwargs)
        return specgram

    # -- TimeSeries filtering -------------------

    def highpass(self, frequency, gpass=2, gstop=30, fstop=None, type='iir',
                 filtfilt=True, **kwargs):
        """Filter this `TimeSeries` with a high-pass filter.

        Parameters
        ----------
        frequency : `float`
            high-pass corner frequency

        gpass : `float`
            the maximum loss in the passband (dB).

        gstop : `float`
            the minimum attenuation in the stopband (dB).

        fstop : `float`
            stop-band edge frequency, defaults to `frequency * 1.5`

        type : `str`
            the filter type, either ``'iir'`` or ``'fir'``

        **kwargs
            other keyword arguments are passed to
            :func:`gwpy.signal.filter_design.highpass`

        Returns
        -------
        hpseries : `TimeSeries`
            a high-passed version of the input `TimeSeries`

        See Also
        --------
        gwpy.signal.filter_design.highpass
            for details on the filter design
        TimeSeries.filter
            for details on how the filter is applied

        .. note::

           When using `scipy < 0.16.0` some higher-order filters may be
           unstable. With `scipy >= 0.16.0` higher-order filters are
           decomposed into second-order-sections, and so are much more stable.
        """
        # design filter
        filt = filter_design.highpass(frequency, self.sample_rate,
                                      fstop=fstop, gpass=gpass, gstop=gstop,
                                      analog=False, type=type, **kwargs)
        # apply filter
        return self.filter(*filt, filtfilt=filtfilt)

    def lowpass(self, frequency, gpass=2, gstop=30, fstop=None, type='iir',
                filtfilt=True, **kwargs):
        """Filter this `TimeSeries` with a Butterworth low-pass filter.

        Parameters
        ----------
        frequency : `float`
            low-pass corner frequency

        gpass : `float`
            the maximum loss in the passband (dB).

        gstop : `float`
            the minimum attenuation in the stopband (dB).

        fstop : `float`
            stop-band edge frequency, defaults to `frequency * 1.5`

        type : `str`
            the filter type, either ``'iir'`` or ``'fir'``

        **kwargs
            other keyword arguments are passed to
            :func:`gwpy.signal.filter_design.lowpass`

        Returns
        -------
        lpseries : `TimeSeries`
            a low-passed version of the input `TimeSeries`

        See Also
        --------
        gwpy.signal.filter_design.lowpass
            for details on the filter design
        TimeSeries.filter
            for details on how the filter is applied

        .. note::

           When using `scipy < 0.16.0` some higher-order filters may be
           unstable. With `scipy >= 0.16.0` higher-order filters are
           decomposed into second-order-sections, and so are much more stable.
        """
        # design filter
        filt = filter_design.lowpass(frequency, self.sample_rate,
                                     fstop=fstop, gpass=gpass, gstop=gstop,
                                     analog=False, type=type, **kwargs)
        # apply filter
        return self.filter(*filt, filtfilt=filtfilt)

    def bandpass(self, flow, fhigh, gpass=2, gstop=30, fstop=None, type='iir',
                 filtfilt=True, **kwargs):
        """Filter this `TimeSeries` with a band-pass filter.

        Parameters
        ----------
        flow : `float`
            lower corner frequency of pass band

        fhigh : `float`
            upper corner frequency of pass band

        gpass : `float`
            the maximum loss in the passband (dB).

        gstop : `float`
            the minimum attenuation in the stopband (dB).

        fstop : `tuple` of `float`, optional
            `(low, high)` edge-frequencies of stop band

        type : `str`
            the filter type, either ``'iir'`` or ``'fir'``

        **kwargs
            other keyword arguments are passed to
            :func:`gwpy.signal.filter_design.bandpass`

        Returns
        -------
        bpseries : `TimeSeries`
            a band-passed version of the input `TimeSeries`

        See Also
        --------
        gwpy.signal.filter_design.bandpass
            for details on the filter design
        TimeSeries.filter
            for details on how the filter is applied

        .. note::

           When using `scipy < 0.16.0` some higher-order filters may be
           unstable. With `scipy >= 0.16.0` higher-order filters are
           decomposed into second-order-sections, and so are much more stable.
        """
        # design filter
        filt = filter_design.bandpass(flow, fhigh, self.sample_rate,
                                      fstop=fstop, gpass=gpass, gstop=gstop,
                                      analog=False, type=type, **kwargs)
        # apply filter
        return self.filter(*filt, filtfilt=filtfilt)

    def resample(self, rate, window='hamming', ftype='fir', n=None):
        """Resample this Series to a new rate

        Parameters
        ----------
        rate : `float`
            rate to which to resample this `Series`

        window : `str`, `numpy.ndarray`, optional
            window function to apply to signal in the Fourier domain,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats, only used for `ftype='fir'` or irregular downsampling

        ftype : `str`, optional
            type of filter, either 'fir' or 'iir', defaults to 'fir'

        n : `int`, optional
            if `ftype='fir'` the number of taps in the filter, otherwise
            the order of the Chebyshev type I IIR filter

        Returns
        -------
        Series
            a new Series with the resampling applied, and the same
            metadata
        """
        if n is None and ftype == 'iir':
            n = 8
        elif n is None:
            n = 60

        if isinstance(rate, units.Quantity):
            rate = rate.value
        factor = (self.sample_rate.value / rate)
        # if integer down-sampling, use decimate
        if factor.is_integer():
            if ftype == 'iir':
                filt = signal.cheby1(n, 0.05, 0.8/factor, output='zpk')
            else:
                filt = signal.firwin(n+1, 1./factor, window=window)
            return self.filter(filt, filtfilt=True)[::int(factor)]
        # otherwise use Fourier filtering
        else:
            nsamp = int(self.shape[0] * self.dx.value * rate)
            new = signal.resample(self.value, nsamp,
                                  window=window).view(self.__class__)
            new.__metadata_finalize__(self)
            new._unit = self.unit
            new.sample_rate = rate
            return new

    def zpk(self, zeros, poles, gain, analog=True, **kwargs):
        """Filter this `TimeSeries` by applying a zero-pole-gain filter

        Parameters
        ----------
        zeros : `array-like`
            list of zero frequencies (in Hertz)

        poles : `array-like`
            list of pole frequencies (in Hertz)

        gain : `float`
            DC gain of filter

        analog : `bool`, optional
            type of ZPK being applied, if `analog=True` all parameters
            will be converted in the Z-domain for digital filtering

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
        return self.filter(zeros, poles, gain, analog=analog, **kwargs)

    def filter(self, *filt, **kwargs):
        """Filter this `TimeSeries` with an IIR or FIR filter

        Parameters
        ----------
        *filt : filter arguments
            1, 2, 3, or 4 arguments defining the filter to be applied,

                - an ``Nx1`` `~numpy.ndarray` of FIR coefficients
                - an ``Nx6`` `~numpy.ndarray` of SOS coefficients
                - ``(numerator, denominator)`` polynomials
                - ``(zeros, poles, gain)``
                - ``(A, B, C, D)`` 'state-space' representation

        filtfilt : `bool`, optional
            filter forward and backwards to preserve phase,
            default: `False`

        analog : `bool`, optional
            if `True`, filter coefficients will be converted from Hz
            to Z-domain digital representation, default: `False`

        inplace : `bool`, optional
            if `True`, this array will be overwritten with the filtered
            version, default: `False`

        **kwargs
            other keyword arguments are passed to the filter method

        Returns
        -------
        result : `TimeSeries`
            the filtered version of the input `TimeSeries`

        Notes
        -----
        IIR filters are converted either into cascading
        second-order sections (if `scipy >= 0.16` is installed), or into the
        ``(numerator, denominator)`` representation before being applied
        to this `TimeSeries`.

        .. note::

           When using `scipy < 0.16` some higher-order filters may be
           unstable. With `scipy >= 0.16` higher-order filters are
           decomposed into second-order-sections, and so are much more stable.

        FIR filters are passed directly to :func:`scipy.signal.lfilter` or
        :func:`scipy.signal.filtfilt` without any conversions.

        See also
        --------
        scipy.signal.sosfilt
            for details on filtering with second-order sections
            (`scipy >= 0.16` only)

        scipy.signal.sosfiltfilt
            for details on forward-backward filtering with second-order
            sections (`scipy >= 0.16` only)

        scipy.signal.lfilter
            for details on filtering (without SOS)

        scipy.signal.filtfilt
            for details on forward-backward filtering (without SOS)

        Raises
        ------
        ValueError
            if ``filt`` arguments cannot be interpreted properly

        Examples
        --------
        We can design an arbitrarily complicated filter using
        :mod:`gwpy.signal.filter_design`

        >>> from gwpy.signal import filter_design
        >>> bp = filter_design.bandpass(50, 250, 4096.)
        >>> notches = [filter_design.notch(f, 4096.) for f in (60, 120, 180)]
        >>> zpk = filter_design.concatenate_zpks(bp, *notches)

        And then can download some data from LOSC to apply it using
        `TimeSeries.filter`:

        >>> from gwpy.timeseries import TimeSeries
        >>> data = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
        >>> filtered = data.filter(zpk, filtfilt=True)

        We can plot the original signal, and the filtered version, cutting
        off either end of the filtered data to remove filter-edge artefacts

        >>> from gwpy.plotter import TimeSeriesPlot
        >>> plot = TimeSeriesPlot(data, filtered[128:-128], sep=True)
        >>> plot.show()
        """
        # parse keyword arguments
        filtfilt = kwargs.pop('filtfilt', False)

        # parse filter
        form, filt = filter_design.parse_filter(
                filt, analog=kwargs.pop('analog', False),
                sample_rate=self.sample_rate.to('Hz').value,
        )
        if form == 'zpk':
            try:
                sos = signal.zpk2sos(*filt)
            except AttributeError:  # scipy < 0.16, no SOS filtering
                sos = None
                b, a = signal.zpk2tf(*filt)
        else:
            sos = None
            b, a = filt

        # perform filter
        kwargs.setdefault('axis', 0)
        if sos is not None and filtfilt:
            out = sosfiltfilt(sos, self, **kwargs)
        elif sos is not None:
            out = signal.sosfilt(sos, self, **kwargs)
        elif filtfilt:
            out = signal.filtfilt(b, a, self, **kwargs)
        else:
            out = signal.lfilter(b, a, self, **kwargs)

        # format as type(self)
        new = out.view(type(self))
        new.__metadata_finalize__(self)
        new._unit = self.unit
        return new

    def coherence(self, other, fftlength=None, overlap=None,
                  window='hann', **kwargs):
        """Calculate the frequency-coherence between this `TimeSeries`
        and another.

        Parameters
        ----------
        other : `TimeSeries`
            `TimeSeries` signal to calculate coherence with

        fftlength : `float`, optional
            number of seconds in single FFT, defaults to a single FFT
            covering the full duration

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        **kwargs
            any other keyword arguments accepted by
            :func:`matplotlib.mlab.cohere` except ``NFFT``, ``window``,
            and ``noverlap`` which are superceded by the above keyword
            arguments

        Returns
        -------
        coherence : `~gwpy.frequencyseries.FrequencySeries`
            the coherence `FrequencySeries` of this `TimeSeries`
            with the other

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
        from matplotlib import mlab
        from ..frequencyseries import FrequencySeries
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
            kwargs['window'] = signal.get_window(window, fftlength)
        coh, freqs = mlab.cohere(self_.value, other.value, NFFT=fftlength,
                                 Fs=sampling, noverlap=overlap, **kwargs)
        out = coh.view(FrequencySeries)
        out.xindex = freqs
        out.epoch = self.epoch
        out.name = 'Coherence between %s and %s' % (self.name, other.name)
        out.unit = 'coherence'
        return out

    def auto_coherence(self, dt, fftlength=None, overlap=None,
                       window='hann', **kwargs):
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

        fftlength : `float`, optional
            number of seconds in single FFT, defaults to a single FFT
            covering the full duration

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        **kwargs
            any other keyword arguments accepted by
            :func:`matplotlib.mlab.cohere` except ``NFFT``, ``window``,
            and ``noverlap`` which are superceded by the above keyword
            arguments

        Returns
        -------
        coherence : `~gwpy.frequencyseries.FrequencySeries`
            the coherence `FrequencySeries` of this `TimeSeries`
            with the other

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
                              overlap=None, window='hann', nproc=1):
        """Calculate the coherence spectrogram between this `TimeSeries`
        and other.

        Parameters
        ----------
        other : `TimeSeries`
            the second `TimeSeries` in this CSD calculation

        stride : `float`
            number of seconds in single PSD (column of spectrogram)

        fftlength : `float`
            number of seconds in single FFT

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        nproc : `int`
            number of parallel processes to use when calculating
            individual coherence spectra.

        Returns
        -------
        spectrogram : `~gwpy.spectrogram.Spectrogram`
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
        stridesamp = int(stride * self.sample_rate.value)
        nsteps = int(self.size // stridesamp)
        # stride through TimeSeries, recording RMS
        data = numpy.zeros(nsteps)
        for step in range(nsteps):
            # find step TimeSeries
            idx = int(stridesamp * step)
            idx_end = idx + stridesamp
            stepseries = self[idx:idx_end]
            rms_ = numpy.sqrt(numpy.mean(numpy.abs(stepseries.value)**2))
            data[step] = rms_
        name = '%s %.2f-second RMS' % (self.name, stride)
        return self.__class__(data, channel=self.channel, t0=self.t0,
                              name=name, sample_rate=(1/float(stride)))

    def demodulate(self, f, stride=1, exp=False, deg=True):
        """Compute the average magnitude and phase of this `TimeSeries`
           once per stride at a given frequency.

        Parameters
        ----------
        f : `float`
            frequency (Hz) at which to demodulate the signal

        stride : `float`, optional
            stride (seconds) between calculations, defaults to 1 second

        exp : `bool`, optional
            return the demodulated magnitude and phase trends as one
            `TimeSeries` object representing a complex exponential

        deg : `bool`, optional
            if `exp=False`, calculates the phase in degrees

        Returns
        -------
        mag, phase : `TimeSeries`
            if `exp=False`, returns a pair of `TimeSeries` objects representing
            magnitude and phase trends with `dt=stride`

        out : `TimeSeries`
            if `exp=True`, returns a single `TimeSeries` with magnitude and
            phase trends represented as `mag * exp(1j*phase)` with `dt=stride`

        Examples
        --------
        Demodulation is useful when trying to examine steady sinusoidal
        signals we know to be contained within data. For instance,
        we can download some data from LOSC to look at trends of the
        amplitude and phase of Livingston's calibration line at 331.3 Hz:

        >>> from gwpy.timeseries import TimeSeries
        >>> data = TimeSeries.fetch_open_data('L1', 1131350417, 1131357617)

        We can demodulate the `TimeSeries` at 331.3 Hz with a stride of once
        per minute:

        >>> amp, phase = data.demodulate(331.3, stride=60)

        We can then plot these trends to visualize changes in the amplitude
        and phase of the calibration line:

        >>> from gwpy.plotter import TimeSeriesPlot
        >>> plot = TimeSeriesPlot(amp, phase, sep=True)
        >>> plot.show()
        """
        stridesamp = int(stride * self.sample_rate.value)
        nsteps = int(self.size // stridesamp)
        # mix with a complex oscillator and stride through the TimeSeries,
        # taking the average over each stride
        out = numpy.zeros(nsteps, dtype=complex).view(type(self))
        out.__metadata_finalize__(self)
        out.sample_rate = 1/float(stride)
        out._unit = self.unit
        mixed = 2 * numpy.exp(-2*numpy.pi*1j*f*self.times.value) * self.value
        # stride through the TimeSeries
        for step in range(nsteps):
            idx = int(stridesamp * step)
            idx_end = idx + stridesamp
            stepseries = mixed[idx:idx_end]
            demod_ = numpy.average(stepseries)
            out.value[step] = demod_
        if exp:
            return out
        mag = numpy.abs(out)
        phase = numpy.angle(out, deg=deg).view(type(self))
        phase.__metadata_finalize__(out)
        phase.override_unit('deg' if deg else 'rad')
        return mag, phase

    def whiten(self, fftlength, overlap=0, method='scipy-welch',
               window='hanning', detrend='constant', asd=None, **kwargs):
        """White this `TimeSeries` against its own ASD

        Parameters
        ----------
        fftlength : `float`
            number of seconds in single FFT

        overlap : `float`, optional
            number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        method : `str`, optional
            FFT-averaging method, default: ``'scipy-welch'``,
            see *Notes* for more details

        window : `str`, `numpy.ndarray`, optional
            window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        detrend : `str`, optional
            type of detrending to do before FFT (see `~TimeSeries.detrend`
            for more details)

        asd : `~gwpy.frequencyseries.FrequencySeries`
            the amplitude-spectral density using which to whiten the data

        **kwargs
            other keyword arguments are passed to the `TimeSeries.asd`
            method to estimate the amplitude spectral density
            `FrequencySeries` of this `TimeSeries`

        Returns
        -------
        out : `TimeSeries`
            a whitened version of the input data

        See Also
        --------
        TimeSeries.asd
            for details on the ASD calculation
        numpy.fft
            for details on the Fourier transform algorithm used her
        scipy.signal

        Notes
        -----"""
        # build whitener
        if asd is None:
            asd = self.asd(fftlength, overlap=overlap,
                           method=method, window=window, **kwargs)
        if isinstance(asd, units.Quantity):
            asd = asd.value
        invasd = 1. / asd
        # build window
        nfft = int((fftlength * self.sample_rate).decompose().value)
        noverlap = int((overlap * self.sample_rate).decompose().value)
        # format window
        if type(window).__module__ == 'lal.lal':
            window = window.data.data
        elif not isinstance(window, numpy.ndarray):
            window = signal.get_window(window, nfft)
        # create output series
        nstride = nfft - noverlap
        nsteps = 1 + int((self.size - nfft) / nstride)
        out = numpy.zeros(nsteps * nstride + noverlap).view(type(self))
        out.__metadata_finalize__(self)
        out._unit = self.unit
        del out.times
        # loop over ffts and whiten each one
        for i in range(nsteps):
            x = i * nstride
            y = x + nfft
            in_ = self[x:y].detrend(detrend) * window
            out.value[x:y] += npfft.irfft(in_.fft().value * invasd)
        return out

    def detrend(self, detrend='constant'):
        """Remove the trend from this `TimeSeries`

        This method just wraps :func:`scipy.signal.detrend` to return
        an object of the same type as the input.

        Parameters
        ----------
        detrend : `str`, optional
            the type of detrending.

        Returns
        -------
        detrended : `TimeSeries`
            the detrended input series

        See Also
        --------
        scipy.signal.detrend
            for details on the options for the `detrend` argument, and
            how the operation is done
        """
        data = signal.detrend(self.value, type=detrend).view(type(self))
        data.__metadata_finalize__(self)
        data._unit = self.unit
        return data

    def notch(self, frequency, type='iir', filtfilt=True, **kwargs):
        """Notch out a frequency in this `TimeSeries`.

        Parameters
        ----------
        frequency : `float`, `~astropy.units.Quantity`
            frequency (default in Hertz) at which to apply the notch

        type : `str`, optional
            type of filter to apply, currently only 'iir' is supported

        **kwargs
            other keyword arguments to pass to `scipy.signal.iirdesign`

        Returns
        -------
        notched : `TimeSeries`
           a notch-filtered copy of the input `TimeSeries`

        See Also
        --------
        TimeSeries.filter
           for details on the filtering method
        scipy.signal.iirdesign
            for details on the IIR filter design method
        """
        zpk = filter_design.notch(frequency, self.sample_rate.value,
                                  type=type, **kwargs)
        return self.filter(*zpk, filtfilt=filtfilt)

    def q_transform(self, qrange=(4, 64), frange=(0, numpy.inf),
                    gps=None, search=.5, tres=.001, fres=.5, norm='median',
                    outseg=None, whiten=True, **asd_kw):
        """Scan a `TimeSeries` using a multi-Q transform

        Parameters
        ----------
        qrange : `tuple` of `float`, optional
            `(low, high)` range of Qs to scan

        frange : `tuple` of `float`, optional
            `(log, high)` range of frequencies to scan

        gps : `float`, optional
            central time of interest for determine loudest Q-plane

        search : `float`, optional
            window around `gps` in which to find peak energies, only
            used if `gps` is given

        tres : `float`, optional
            desired time resolution (seconds) of output `Spectrogram`

        fres : `float`, `None`, optional
            desired frequency resolution (Hertz) of output `Spectrogram`,
            give `None` to skip this step and return the original resolution,
            e.g. if you're going to do your own interpolation

        norm : `bool`, `str`, optional
            whether to normalize the returned Q-transform output, or how,
            default: `True` (``'median'``), other options: `False`,
            ``'mean'``

        outseg : `~gwpy.segments.Segment`, optional
            GPS `[start, stop)` segment for output `Spectrogram`

        whiten : `bool`, `~gwpy.frequencyseries.FrequencySeries`, optional
            boolean switch to enable (`True`) or disable (`False`) data
            whitening, or an ASD `~gwpy.freqencyseries.FrequencySeries`
            with which to whiten the data

        **asd_kw
            keyword arguments to pass to `TimeSeries.asd` to generate
            an ASD to use when whitening the data

        Returns
        -------
        specgram : `~gwpy.spectrogram.Spectrogram`
            output `Spectrogram` of normalised Q energy

        See Also
        --------
        TimeSeries.asd
            for documentation on acceptable `**asd_kw`
        TimeSeries.whiten
            for documentation on how the whitening is done
        gwpy.signal.qtransform
            for code and documentation on how the Q-transform is implemented
        scipy.interpolate
            for details on how the interpolation is implemented. This method
            uses `~scipy.interpolate.InterpolatedUnivariateSpline` to
            cast all frequency rows to the same time-axis, and then
            `~scipy.interpolate.interpd` to apply the desired frequency
            resolution across the band.

        Notes
        -----
        It is highly recommended to use the `outseg` keyword argument when
        only a small window around a given GPS time is of interest. This
        will speed up this method a little, but can greatly speed up
        rendering the resulting `~gwpy.spectrogram.Spectrogram` using
        `~matplotlib.axes.Axes.pcolormesh`.

        If you aren't going to use `pcolormesh` in the end, don't worry.

        Examples
        --------
        >>> from numpy.random import normal
        >>> from scipy.signal import gausspulse
        >>> from gwpy.timeseries import TimeSeries

        Generate a `TimeSeries` containing Gaussian noise sampled at 4096 Hz,
        centred on GPS time 0, with a sine-Gaussian pulse ('glitch') at
        500 Hz:

        >>> noise = TimeSeries(normal(loc=1, size=4096*4), sample_rate=4096, epoch=-2)
        >>> glitch = TimeSeries(gausspulse(noise.times.value, fc=500) * 4, sample_rate=4096)
        >>> data = noise + glitch

        Compute and plot the Q-transform of these data:

        >>> q = data.q_transform()
        >>> plot = q.plot()
        >>> ax = plot.gca()
        >>> ax.set_xlim(-.2, .2)
        >>> ax.set_epoch(0)
        >>> plot.show()
        """  # nopep8
        from scipy.interpolate import (interp2d, InterpolatedUnivariateSpline)
        from ..frequencyseries import FrequencySeries
        from ..spectrogram import Spectrogram
        from ..signal.qtransform import QTiling

        if outseg is None:
            outseg = self.span

        # generate tiling
        planes = QTiling(abs(self.span), self.sample_rate.value,
                         qrange=qrange, frange=frange)

        # condition data
        if whiten:
            if isinstance(whiten, FrequencySeries):
                fftlength = 1/whiten.df.value
                overlap = fftlength / 2.
            else:
                method = asd_kw.pop('method', 'scipy_welch')
                window = asd_kw.pop('window', 'hann')
                fftlength = asd_kw.pop(
                    'fftlength',
                    min(planes.whitening_duration, self.duration.value))
                overlap = asd_kw.pop('overlap', None)
                if overlap is None and fftlength == self.duration.value:
                    method = 'scipy-welch'
                    overlap = 0
                elif overlap is None:
                    overlap = recommended_overlap(window) * fftlength
                whiten = self.asd(fftlength, overlap, method=method, **asd_kw)
            # apply whitening
            wdata = self.whiten(fftlength, overlap, asd=whiten)
            fdata = wdata.fft().value
        else:
            fdata = self.fft().value

        # set up results
        peakq = None
        peakenergy = 0

        # Q-transform data for each `(Q, frequency)` tile
        for plane in planes:
            freqs, normenergies = plane.transform(fdata, norm=norm,
                                                  epoch=self.x0)
            # find peak energy in this plane and record if loudest
            for ts in normenergies:
                if gps is None:
                    peak = ts.value.max()
                else:
                    peak = ts.crop(gps-search, gps+search).value.max()
                if peak > peakenergy:
                    peakenergy = peak
                    peakq = plane.q
                    norms = normenergies
                    frequencies = freqs

        # build regular Spectrogram from peak-Q data by interpolating each
        # (Q, frequency) `TimeSeries` to have the same time resolution
        nx = int(abs(Segment(*outseg)) / tres)
        ny = frequencies.size
        out = Spectrogram(numpy.zeros((nx, ny)), x0=outseg[0], dx=tres,
                          frequencies=frequencies)
        # FIXME: bug in Array2D.yindex setting
        out._yindex = type(out.y0)(frequencies, out.y0.unit)
        # record Q in output
        out.q = peakq
        # interpolate rows
        for i, row in enumerate(norms):
            row = row.crop(*outseg)
            interp = InterpolatedUnivariateSpline(row.times.value, row.value)
            out[:, i] = interp(out.times.value)

        # then interpolate the spectrogram to increase the frequency resolution
        # --- this is done because duncan doesn't like interpolated images
        #     because they don't support log scaling
        if fres is None:  # unless user tells us not to
            return out

        interp = interp2d(out.times.value, frequencies, out.value.T,
                          kind='cubic')
        freqs2 = numpy.arange(planes.frange[0], planes.frange[1], fres)
        new = Spectrogram(interp(out.times.value, freqs2 + fres/2.).T,
                          x0=outseg[0], dx=tres,
                          f0=planes.frange[0], df=fres)
        new.q = peakq
        return new


@as_series_dict_class(TimeSeries)
class TimeSeriesDict(TimeSeriesBaseDict):  # pylint: disable=missing-docstring
    __doc__ = TimeSeriesBaseDict.__doc__.replace('TimeSeriesBase',
                                                 'TimeSeries')
    EntryClass = TimeSeries


class TimeSeriesList(TimeSeriesBaseList):  # pylint: disable=missing-docstring
    __doc__ = TimeSeriesBaseList.__doc__.replace('TimeSeriesBase',
                                                 'TimeSeries')
    EntryClass = TimeSeries
