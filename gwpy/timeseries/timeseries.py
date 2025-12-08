# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""The `TimeSeries`."""

from __future__ import annotations

import math
import warnings
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)

import numpy
from astropy import units
from astropy.units import Quantity
from numpy import fft as npfft
from scipy import signal
from scipy.signal.windows import tukey

from ..io.registry import UnifiedReadWriteMethod
from ..segments import (
    DataQualityFlag,
    Segment,
    SegmentList,
)
from ..signal import (
    filter_design,
    qtransform,
    spectral,
)
from ..signal.window import (
    get_window,
    recommended_overlap,
)
from .connect import (
    TimeSeriesDictGet,
    TimeSeriesDictRead,
    TimeSeriesDictWrite,
    TimeSeriesGet,
    TimeSeriesRead,
    TimeSeriesWrite,
)
from .core import (
    TimeSeriesBase,
    TimeSeriesBaseDict,
    TimeSeriesBaseList,
    as_series_dict_class,
)

if TYPE_CHECKING:
    from typing import (
        Literal,
        Self,
    )

    from astropy.units.typing import QuantityLike
    from numpy.typing import ArrayLike

    from ..frequencyseries import FrequencySeries, SpectralVariance
    from ..signal.filter_design import (
        BAType,
        FilterCompatible,
        SosType,
        TapsType,
        ZpkType,
    )
    from ..signal.window import WindowLike
    from ..spectrogram import Spectrogram
    from ..table import EventTable
    from ..typing import ArrayLike1D

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

DEFAULT_FFT_METHOD = "median"


# -- utilities -----------------------

def _fft_length_default(dt: Quantity) -> int:
    """Choose an appropriate FFT length (in seconds) based on a sample rate.

    Parameters
    ----------
    dt : `~astropy.units.Quantity`
        The sampling time interval, in seconds.

    Returns
    -------
    fftlength : `int`
        A choice of FFT length, in seconds.
    """
    return int(max(2, numpy.ceil(2048 * dt.decompose().value)))


# -- TimeSeries ----------------------

class TimeSeries(TimeSeriesBase):
    """A time-domain data array.

    Parameters
    ----------
    value : array-like
        Input data array.

    unit : `~astropy.units.Unit`, optional
        Physical unit of these data.

    t0 : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine.

    dt : `float`, `~astropy.units.Quantity`, optional
        Time between successive samples (seconds), can also be given inversely
        via `sample_rate`.

    sample_rate : `float`, `~astropy.units.Quantity`, optional
        The rate of samples per second (Hertz), can also be given inversely via `dt`.

    times : `array-like`
        The complete array of GPS times accompanying the data for this series.
        This argument takes precedence over `t0` and `dt` so should be given
        in place of these if relevant, not alongside.

    name : `str`, optional
        Descriptive title for this array.

    channel : `~gwpy.detector.Channel`, `str`, optional
        Source data stream for these data.

    dtype : `~numpy.dtype`, optional
        Input data type.

    copy : `bool`, optional
        Choose to copy the input data to new memory.

    subok : `bool`, optional
        Allow passing of sub-classes by the array generator.

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

    # -- i/o -------------------------

    read = UnifiedReadWriteMethod(TimeSeriesRead)
    write = UnifiedReadWriteMethod(TimeSeriesWrite)
    get = UnifiedReadWriteMethod(TimeSeriesGet)

    # -- signal processing -----------

    def fft(
        self: Self,
        nfft: int | None = None,
    ) -> FrequencySeries:
        """Compute the one-dimensional discrete Fourier transform of this `TimeSeries`.

        Parameters
        ----------
        nfft : `int`, optional
            Length of the desired Fourier transform, input will be
            cropped or padded to match the desired length.
            If nfft is not given, the length of the `TimeSeries`
            will be used.

        Returns
        -------
        out : `~gwpy.frequencyseries.FrequencySeries`
            The normalised, complex-valued FFT `FrequencySeries`.

        See Also
        --------
        numpy.fft.rfft
            The FFT implementation used in this method.

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
        new = FrequencySeries(
            dft,
            epoch=self.epoch,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
        )
        try:
            new.frequencies = npfft.rfftfreq(nfft, d=self.dx.value)
        except AttributeError:
            new.frequencies = numpy.arange(new.size) / (nfft * self.dx.value)
        return new

    def average_fft(
        self,
        fftlength: float | Quantity | None = None,
        overlap: float | Quantity = 0,
        window: WindowLike | None = None,
    ) -> FrequencySeries:
        """Compute the averaged one-dimensional DFT of this `TimeSeries`.

        This method computes a number of FFTs of duration ``fftlength``
        and ``overlap`` (both given in seconds), and returns the mean
        average. This method is analogous to the Welch average method
        for power spectra.

        Parameters
        ----------
        fftlength : `float`
            Number of seconds in single FFT; by default uses whole `TimeSeries`.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        Returns
        -------
        out : complex-valued `~gwpy.frequencyseries.FrequencySeries`
            The transformed output, with populated frequencies array metadata.

        See Also
        --------
        TimeSeries.fft
           The FFT method used.
        """
        from gwpy.spectrogram import Spectrogram

        # format lengths
        if fftlength is None:
            fftlength = self.duration
        fftlength = Quantity(fftlength, units.second)
        nfft = int((cast("Quantity", fftlength * self.sample_rate)).decompose().value)
        noverlap = int((cast("Quantity", overlap * self.sample_rate)).decompose().value)

        navg = divmod(self.size-noverlap, (nfft-noverlap))[0]

        # format window
        if window is None:
            window = "boxcar"
        win = get_window(window, nfft).astype(self.dtype)
        scaling = 1. / numpy.absolute(win).mean()

        if nfft % 2:
            nfreqs = (nfft + 1) // 2
        else:
            nfreqs = nfft // 2 + 1
        ffts = Spectrogram(
            numpy.zeros((navg, nfreqs), dtype=numpy.complex128),
            channel=self.channel,
            epoch=self.epoch,
            f0=0,
            df=1 / fftlength,
            dt=1,
            copy=True,
        )
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

    def psd(
        self,
        fftlength: float | None = None,
        overlap: float | None = None,
        window: WindowLike = "hann",
        method: str = DEFAULT_FFT_METHOD,
        **kwargs,
    ) -> FrequencySeries:
        """Calculate the PSD `FrequencySeries` for this `TimeSeries`.

        Parameters
        ----------
        fftlength : `float`
            Number of seconds in single FFT, defaults to a single FFT
            covering the full duration.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        method : `str`, optional
            FFT-averaging method (default: ``'median'``),
            see *Notes* for more details.

        kwargs
            Other keyword arguments are passed to the underlying PSD-generation method.

        Returns
        -------
        psd :  `~gwpy.frequencyseries.FrequencySeries`
            A data series containing the PSD.

        Notes
        -----
        The accepted ``method`` arguments are:

        - ``'bartlett'`` : a mean average of non-overlapping periodograms
        - ``'median'`` : a median average of overlapping periodograms
        - ``'welch'`` : a mean average of overlapping periodograms
        """
        # get method
        method_func = spectral.get_method(method)

        # calculate PSD using UI method
        return spectral.psd(
            self,
            method_func,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            **kwargs,
        )

    def asd(
        self,
        fftlength: float | None = None,
        overlap: float | None = None,
        window: WindowLike = "hann",
        method: str = DEFAULT_FFT_METHOD,
        **kwargs,
    ) -> FrequencySeries:
        """Calculate the ASD `FrequencySeries` of this `TimeSeries`.

        Parameters
        ----------
        fftlength : `float`
            Number of seconds in single FFT, defaults to a single FFT
            covering the full duration.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        method : `str`, optional
            FFT-averaging method (default: ``'median'``), see *Notes* for more details

        kwargs
            Other keyword arguments are passed to the underlying
            ASD-generation method.

        Returns
        -------
        asd :  `~gwpy.frequencyseries.FrequencySeries`
            A data series containing the ASD.

        See Also
        --------
        TimeSeries.psd

        Notes
        -----
        The accepted ``method`` arguments are:

        - ``'bartlett'`` : a mean average of non-overlapping periodograms
        - ``'median'`` : a median average of overlapping periodograms
        - ``'welch'`` : a mean average of overlapping periodograms
        """
        return self.psd(
            method=method,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            **kwargs,
        ) ** (1/2.)

    def csd(
        self,
        other: TimeSeries,
        fftlength: float | None = None,
        overlap: float | None = None,
        window: WindowLike = "hann",
        **kwargs,
    ) -> FrequencySeries:
        """Calculate the CSD `FrequencySeries` for two `TimeSeries`.

        Parameters
        ----------
        other : `TimeSeries`
            The second `TimeSeries` in this CSD calculation.

        fftlength : `float`
            Number of seconds in single FFT, defaults to a single FFT
            covering the full duration.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        kwargs
            Other keyword arguments are passed to the underlying
            CSD-generation method.

        Returns
        -------
        csd :  `~gwpy.frequencyseries.FrequencySeries`
            A data series containing the CSD.
        """
        return spectral.psd(
            (self, other),
            spectral.csd,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            **kwargs,
        )

    def spectrogram(
        self,
        stride: float,
        fftlength: float | None = None,
        overlap: float | None = None,
        window: WindowLike = "hann",
        method: str = DEFAULT_FFT_METHOD,
        nproc: int = 1,
        **kwargs,
    ) -> Spectrogram:
        """Calculate the average power spectrogram of this `TimeSeries`.

        Each time-bin of the output `Spectrogram` is calculated by taking
        a chunk of the `TimeSeries` in the segment
        `[t - overlap/2., t + stride + overlap/2.)` and calculating the
        :meth:`~gwpy.timeseries.TimeSeries.psd` of those data.

        As a result, each time-bin is calculated using `stride + overlap`
        seconds of data.

        Parameters
        ----------
        stride : `float`
            Number of seconds in single PSD (column of spectrogram).

        fftlength : `float`
            Number of seconds in single FFT.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        method : `str`, optional
            FFT-averaging method (default: ``'median'``),
            see *Notes* for more details.

        nproc : `int`
            Number of CPUs to use in parallel processing of FFTs.

        kwargs
            Other keyword arguments are passed to the underlying
            PSD-generation method.

        Returns
        -------
        spectrogram : `~gwpy.spectrogram.Spectrogram`
            Time-frequency power spectrogram as generated from the
            input time-series.

        Notes
        -----
        The accepted ``method`` arguments are:

        - ``'bartlett'`` : a mean average of non-overlapping periodograms
        - ``'median'`` : a median average of overlapping periodograms
        - ``'welch'`` : a mean average of overlapping periodograms
        """
        # get method
        method_func = spectral.get_method(method)

        # calculate PSD using UI method
        return spectral.average_spectrogram(
            self,
            method_func,
            stride,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            nproc=nproc,
            **kwargs,
        )

    def spectrogram2(
        self,
        fftlength: float,
        overlap: float | None = None,
        window: WindowLike = "hann",
        **kwargs,
    ) -> Spectrogram:
        """Calculate the non-averaged power `Spectrogram` of this `TimeSeries`.

        Parameters
        ----------
        fftlength : `float`
            Number of seconds in single FFT.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        scaling : [ 'density' | 'spectrum' ], optional
            Selects between computing the power spectral density ('density')
            where the `Spectrogram` has units of V**2/Hz if the input is
            measured in V and computing the power spectrum ('spectrum')
            where the `Spectrogram` has units of V**2 if the input is
            measured in V. Defaults to 'density'.

        kwargs
            Other parameters to be passed to `scipy.signal.periodogram` for
            each column of the `Spectrogram`.

        Returns
        -------
        spectrogram: `~gwpy.spectrogram.Spectrogram`
            A power `Spectrogram` with `1/fftlength` frequency resolution and
            (fftlength - overlap) time resolution.

        See Also
        --------
        scipy.signal.periodogram
            For documentation on the Fourier methods used in this calculation.

        Notes
        -----
        This method calculates overlapping periodograms for all possible
        chunks of data entirely containing within the span of the input
        `TimeSeries`, then normalises the power in overlapping chunks using
        a triangular window centred on that chunk which most overlaps the
        given `Spectrogram` time sample.
        """
        # set kwargs for periodogram()
        kwargs.setdefault("fs", self.sample_rate.to("Hz").value)
        # run
        return spectral.spectrogram(
            self,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            **kwargs,
        )

    def fftgram(
        self,
        fftlength: float,
        overlap: float | None = None,
        window: WindowLike = "hann",
        **kwargs,
    ) -> Spectrogram:
        """Calculate the Fourier-gram of this `TimeSeries`.

        At every ``stride``, a single, complex FFT is calculated.

        Parameters
        ----------
        fftlength : `float`
            Number of seconds in single FFT.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        kwargs
            Other keyword arguments are passed to the
            :meth:`scipy.signal.spectrogram` method.

        Returns
        -------
        spectrogram : `~gwpy.spectrogram.Spectrogram`
            A `Spectrogram` containing the complex-valued output of 1D FFTs
            at every ``stride`` in the input `TimeSeries`, with each column
            corresponding to a single FFT.
        """
        from ..spectrogram import Spectrogram

        # format lengths
        if isinstance(fftlength, Quantity):
            fftlength = fftlength.value

        nfft = int((fftlength * self.sample_rate).decompose().value)

        if not overlap:
            # use scipy.signal.spectrogram noverlap default
            noverlap = nfft // 8
        else:
            noverlap = int((overlap * self.sample_rate).decompose().value)

        # generate output spectrogram
        [frequencies, times, sxx] = signal.spectrogram(
            self,
            fs=float(self.sample_rate.value),
            window=window,
            nperseg=nfft,
            noverlap=noverlap,
            mode="complex",
            **kwargs,
        )
        return Spectrogram(
            sxx.T,
            name=self.name,
            unit=self.unit,
            xindex=self.t0.value + times,
            yindex=frequencies,
        )

    def spectral_variance(
        self,
        stride: float,
        fftlength: float | None = None,
        overlap: float | None = None,
        method: str = DEFAULT_FFT_METHOD,
        window: WindowLike = "hann",
        *,
        nproc: int = 1,
        filter_: FilterCompatible | None = None,
        bins: ArrayLike | None = None,
        low: float | None = None,
        high: float | None = None,
        nbins: int = 500,
        log: bool = False,
        norm: bool = False,
        density: bool = False,
    ) -> SpectralVariance:
        """Calculate the `SpectralVariance` of this `TimeSeries`.

        Parameters
        ----------
        stride : `float`
            Number of seconds in single PSD (column of spectrogram).

        fftlength : `float`
            Number of seconds in single FFT.

        method : `str`, optional
            FFT-averaging method (default: ``'median'``),
            see *Notes* for more details.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        nproc : `int`
            Maximum number of independent frame reading processes, default
            is set to single-process file reading.

        filter_ : `FilterCompatible`, optional
            Filter to apply to each time-bin of the spectrogram prior to
            variance calculation.

        bins : `numpy.ndarray`, optional, default `None`
            Array of histogram bin edges, including the rightmost edge.

        low : `float`, optional
            Left edge of lowest amplitude bin, only read
            if ``bins`` is not given.

        high : `float`, optional
            Right edge of highest amplitude bin, only read
            if ``bins`` is not given.

        nbins : `int`, optional
            Number of bins to generate, only read if ``bins`` is not given.

        log : `bool`, optional
            Calculate amplitude bins over a logarithmic scale, only
            read if ``bins`` is not given.

        norm : `bool`, optional
            Normalise bin counts to a unit sum.

        density : `bool`, optional
            Normalise bin counts to a unit integral.

        Returns
        -------
        specvar : `SpectralVariance`
            2D-array of spectral frequency-amplitude counts.

        See Also
        --------
        numpy.histogram
            For details on specifying bins and weights.

        Notes
        -----
        The accepted ``method`` arguments are:

        - ``'bartlett'`` : a mean average of non-overlapping periodograms
        - ``'median'`` : a median average of overlapping periodograms
        - ``'welch'`` : a mean average of overlapping periodograms
        """
        specgram = self.spectrogram(
            stride,
            fftlength=fftlength,
            overlap=overlap,
            method=method,
            window=window,
            nproc=nproc,
        ) ** (1/2.)
        if filter_ is not None:
            specgram = specgram.filter(filter_)
        return specgram.variance(
            bins=bins,
            low=low,
            high=high,
            nbins=nbins,
            log=log,
            norm=norm,
            density=density,
        )

    def rayleigh_spectrum(
        self,
        fftlength: float | None = None,
        overlap: float = 0,
        window: WindowLike = "hann",
    ) -> FrequencySeries:
        """Calculate the Rayleigh `FrequencySeries` for this `TimeSeries`.

        The Rayleigh statistic is calculated as the ratio of the standard
        deviation and the mean of a number of periodograms.

        Parameters
        ----------
        fftlength : `float`
            Number of seconds in single FFT, defaults to a single FFT
            covering the full duration.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, passing `None` will
            choose based on the window method, default: ``0``.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        Returns
        -------
        psd :  `~gwpy.frequencyseries.FrequencySeries`
            A data series containing the PSD.
        """
        return spectral.psd(
            self,
            spectral.rayleigh,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
        )

    def rayleigh_spectrogram(
        self,
        stride: float,
        fftlength: float | None = None,
        overlap: float = 0,
        window: WindowLike = "hann",
        nproc: int = 1,
        **kwargs,
    ) -> Spectrogram:
        """Calculate the Rayleigh statistic spectrogram of this `TimeSeries`.

        Parameters
        ----------
        stride : `float`
            Number of seconds in single PSD (column of spectrogram).

        fftlength : `float`
            Number of seconds in single FFT.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, passing `None` will
            choose based on the window method, default: ``0``.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        nproc : `int`, optional
            Maximum number of independent frame reading processes, default
            is set to single-process file reading.

        kwargs
            Other keyword arguments are passed to the underlying
            Rayleigh statistic calculation.

        Returns
        -------
        spectrogram : `~gwpy.spectrogram.Spectrogram`
            Time-frequency Rayleigh spectrogram as generated from the
            input time-series.

        See Also
        --------
        TimeSeries.rayleigh
            For details of the statistic calculation.
        """
        specgram = spectral.average_spectrogram(
            self,
            spectral.rayleigh,
            stride,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            nproc=nproc,
            **kwargs,
        )
        specgram.override_unit("")
        return specgram

    def csd_spectrogram(
        self,
        other: TimeSeries,
        stride: float,
        fftlength: float | None = None,
        overlap: float = 0,
        window: WindowLike = "hann",
        nproc: int = 1,
        **kwargs,
    ) -> Spectrogram:
        """Calculate the cross spectral density spectrogram with ``other``.

        Parameters
        ----------
        other : `~gwpy.timeseries.TimeSeries`
            Second time-series for cross spectral density calculation.

        stride : `float`
            Number of seconds in single PSD (column of spectrogram).

        fftlength : `float`
            Number of seconds in single FFT.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        nproc : `int`
            Maximum number of independent frame reading processes, default
            is set to single-process file reading.

        kwargs
            Other keyword arguments are passed to the underlying
            CSD-generation method.

        Returns
        -------
        spectrogram : `~gwpy.spectrogram.Spectrogram`
            Time-frequency cross spectrogram as generated from the
            two input time-series.
        """
        return spectral.average_spectrogram(
            (self, other),
            spectral.csd,
            stride,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            nproc=nproc,
            **kwargs,
        )

    # -- filtering -------------------

    def highpass(
        self,
        frequency: float,
        gpass: float = 2,
        gstop: float = 30,
        fstop: float | None = None,
        type: Literal["fir", "iir"] = "iir",  # noqa: A002
        *,
        filtfilt: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Filter this `TimeSeries` with a high-pass filter.

        Parameters
        ----------
        frequency : `float`
            High-pass corner frequency.

        gpass : `float`
            The maximum loss in the passband (dB).

        gstop : `float`
            The minimum attenuation in the stopband (dB).

        fstop : `float`
            Stop-band edge frequency, defaults to ``frequency * 1.5``.

        type : `str`
            The filter type, either ``'iir'`` or ``'fir'``.

        filtfilt : `bool`, optional
            If `True`, apply the filter using a forward-backward
            filter design, otherwise apply the filter in a single pass.
            Defaults to `True`.

        kwargs
            Other keyword arguments are passed to
            :func:`gwpy.signal.filter_design.highpass`.

        Returns
        -------
        hpseries : `TimeSeries`
            A high-passed version of the input `TimeSeries`.

        See Also
        --------
        gwpy.signal.filter_design.highpass
            For details on the filter design.
        TimeSeries.filter
            For details on how the filter is applied.
        """
        # design filter
        filt = filter_design.highpass(
            frequency,
            self.sample_rate,
            fstop=fstop,
            gpass=gpass,
            gstop=gstop,
            analog=False,
            type=type,
            output="zpk",
            **kwargs,
        )
        # filter_design.highpass returns rad/sample already
        return self._filter(
            "zpk" if type == "iir" else "ba",
            filt,
            filtfilt=filtfilt,
        )

    def lowpass(
        self,
        frequency: float,
        gpass: float = 2,
        gstop: float = 30,
        fstop: float | None = None,
        type: Literal["fir", "iir"] = "iir",  # noqa: A002
        *,
        filtfilt: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Filter this `TimeSeries` with a Butterworth low-pass filter.

        Parameters
        ----------
        frequency : `float`
            Low-pass corner frequency.

        gpass : `float`
            The maximum loss in the passband (dB).

        gstop : `float`
            The minimum attenuation in the stopband (dB).

        fstop : `float`
            Stop-band edge frequency, defaults to `frequency * 1.5`.

        type : `str`
            The filter type, either ``'iir'`` or ``'fir'``.

        filtfilt : `bool`, optional
            If `True`, apply the filter using a forward-backward
            filter design, otherwise apply the filter in a single pass.
            Defaults to `True`.

        kwargs
            Other keyword arguments are passed to
            :func:`gwpy.signal.filter_design.lowpass`.

        Returns
        -------
        lpseries : `TimeSeries`
            A low-passed version of the input `TimeSeries`.

        See Also
        --------
        gwpy.signal.filter_design.lowpass
            For details on the filter design.
        TimeSeries.filter
            For details on how the filter is applied.
        """
        # design filter
        filt = filter_design.lowpass(
            frequency,
            self.sample_rate,
            fstop=fstop,
            gpass=gpass,
            gstop=gstop,
            analog=False,
            type=type,
            output="zpk",
            **kwargs,
        )
        # apply filter, it is already rad/sample
        return self._filter(
            "zpk" if type == "iir" else "ba",
            filt,
            filtfilt=filtfilt,
        )

    def bandpass(
        self,
        flow: float,
        fhigh: float,
        gpass: float = 2,
        gstop: float = 30,
        fstop: tuple[float, float] | None = None,
        type: Literal["fir", "iir"] = "iir",  # noqa: A002
        *,
        filtfilt: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Filter this `TimeSeries` with a band-pass filter.

        Parameters
        ----------
        flow : `float`
            Lower corner frequency of pass band.

        fhigh : `float`
            Upper corner frequency of pass band.

        gpass : `float`
            The maximum loss in the passband (dB).

        gstop : `float`
            The minimum attenuation in the stopband (dB).

        fstop : `tuple` of `float`, optional
            ``(low, high)`` edge-frequencies of stop band.

        type : `str`
            The filter type, either ``'iir'`` or ``'fir'``.

        filtfilt : `bool`, optional
            If `True`, apply the filter using a forward-backward
            filter design, otherwise apply the filter in a single pass.
            Defaults to `True`.

        kwargs
            Other keyword arguments are passed to
            :func:`gwpy.signal.filter_design.bandpass`

        Returns
        -------
        bpseries : `TimeSeries`
            A band-passed version of the input `TimeSeries`.

        See Also
        --------
        gwpy.signal.filter_design.bandpass
            For details on the filter design.
        TimeSeries.filter
            For details on how the filter is applied.
        """
        # design filter
        filt = filter_design.bandpass(
            flow,
            fhigh,
            self.sample_rate,
            fstop=fstop,
            gpass=gpass,
            gstop=gstop,
            analog=False,
            type=type,
            output="zpk",
            **kwargs,
        )
        # apply filter
        return self._filter(
            "zpk" if type == "iir" else "ba",
            filt,
            filtfilt=filtfilt,
        )

    def resample(
        self,
        rate: float,
        window: WindowLike = "hamming",
        ftype: Literal["fir", "iir"] = "fir",
        n: int | None = None,
    ) -> TimeSeries:
        """Resample this Series to a new rate.

        Upsampling, or downsampling by a non-integer factor calls out to
        :func:`scipy.signal.resample`, while integer downsampling calls out
        to :func:`scipy.signal.decimate`.

        Parameters
        ----------
        rate : `float`, `astropy.units.Quantity`
            Rate to which to resample this `Series`.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to signal in the Fourier domain,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats, only used for `ftype='fir'` or irregular downsampling.

        ftype : `str`, optional
            Type of filter, either 'fir' or 'iir', defaults to 'fir'.

        n : `int`, optional
            If `ftype='fir'` the number of taps in the filter, otherwise
            the order of the Chebyshev type I IIR filter.

        Returns
        -------
        Series
            A new Series with the resampling applied, and the same metadata.
        """
        # calculate downsampling factor
        if not isinstance(rate, units.Quantity):
            rate = units.Quantity(rate, 1 / self.xunit)
        downfactor = (self.sample_rate / rate).decompose()
        if downfactor.unit != units.dimensionless_unscaled:
            msg = f"invalid resampling rate '{rate}'"
            raise ValueError(msg)
        downfactor = downfactor.value

        # if the factor is 1 (or extremely close to 1), do nothing
        if math.isclose(downfactor, 1., rel_tol=1e-09, abs_tol=0.):
            warnings.warn(
                f"resample() rate matches current sample_rate ({self.sample_rate}), "
                "returning input data unmodified; please double-check your parameters",
                UserWarning,
                stacklevel=2,
            )
            return self

        def _repack(new: numpy.ndarray) -> TimeSeries:
            """Repack the resampled array to look like the original object."""
            out = cast("Self", new.view(type(self)))
            out.__metadata_finalize__(self)
            out.override_unit(self.unit)
            out.sample_rate = rate
            return out

        # if integer down-sampling, use decimate
        if downfactor.is_integer():
            out = signal.decimate(
                self.value,
                int(downfactor),
                n=n,
                ftype=ftype,
                zero_phase=True,
            )
        # otherwise use Fourier filtering
        else:
            nsamp = int(self.shape[0] / downfactor)
            out = signal.resample(
                self.value,
                nsamp,
                window=window,
            )

        return cast("Self", _repack(out))

    def zpk(
        self,
        zeros: ArrayLike1D,
        poles: ArrayLike1D,
        gain: float,
        *,
        analog: bool = False,
        unit: str = "rad/s",
        normalize_gain: bool = False,
        filtfilt: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Filter this `TimeSeries` by applying a digital zero-pole-gain filter.

        Parameters
        ----------
        zeros : `array-like`
            Zeros of the transfer function.

        poles : `array-like`
            Poles of the transfer function.

        gain : `float`
            System gain.

        analog : `bool`, optional
            Type of filter being applied.
            If ``analog=True`` the zeros/poles/gain will be transformed from
            analogue (s-plane) to digital (z-plane) representation using
            the bilinear transform.

        unit : `str`
            For analogue ZPK filters, the units in which the zeros and poles are
            specified. Either ``'Hz'`` or ``'rad/s'`` (default).

        normalize_gain : `bool`, optional
            Whether to normalize the gain when converting from Hz to rad/s.

            - `False` (default):
              Multiply zeros/poles by -2π but leave gain unchanged.
              This matches the LIGO GDS **'f' plane** convention
              (``plane='f'`` in ``s2z()``).

            - `True`:
              Normalize gain to preserve frequency response magnitude.
              Gain is scaled by :math:`|∏p_i/∏z_i| · (2π)^{(n_p - n_z)}`.
              Use this when your filter was designed with the transfer
              function :math:`H(f) = k·∏(f-z_i)/∏(f-p_i)` in Hz.
              This matches the LIGO GDS **'n' plane** convention
              (``plane='n'`` in ``s2z()``).

            Only used for analogue filters in Hz (``analog=True, unit="Hz"``).

        filtfilt : `bool`, optional
            If `True` (default), apply the filter using a forward-backward
            filter design, otherwise apply the filter in a single pass.

        kwargs
            Other keyword arguments are passed to the filter method.

        Returns
        -------
        timeseries : `TimeSeries`
            The filtered version of the input data.

        See Also
        --------
        TimeSeries.filter
            For details on how a digital ZPK-format filter is applied.
        gwpy.signal.filter_design.prepare_digital_filter
            For details on preparing the digital ZPK filter for application.

        Examples
        --------
        To apply a zpk filter with file poles at 100 Hz, and five zeros at
        1 Hz (giving an overall DC gain of 1e-10)::

        >>> data2 = data.zpk([100]*5, [1]*5, 1e-10)
        """
        zpk = filter_design.prepare_digital_filter(
            (zeros, poles, gain),
            analog=analog,
            sample_rate=self.sample_rate,
            unit=unit,
            normalize_gain=normalize_gain,
            output="zpk",
        )
        return self._filter(
            "zpk",
            zpk,
            filtfilt=filtfilt,
            **kwargs,
        )

    def filter(
        self,
        filt: FilterCompatible,
        *,
        analog: bool = False,
        unit: str = "rad/s",
        normalize_gain: bool = False,
        filtfilt: bool = True,
        **kwargs,
    ) -> Self:
        """Filter this `TimeSeries` with an IIR or FIR filter.

        Parameters
        ----------
        filt : `numpy.ndarray` or `tuple`
            The filter to be applied.
            This can be specified in any of the following forms, with
            the appropriate number of elements in the tuple:

            - `numpy.ndarray` - 1D array of FIR filter coefficients.
            - `tuple[numpy.ndarray, numpy.ndarray]` - numerator/demoinator
              polynomials of the transfer function.
            - `numpy.ndarray` - 2D array of SOS coefficients.
            - `tuple[numpy.ndarray, numpy.ndarray, float]` - zero-pole-gain
              representation.

        filtfilt : `bool`, optional
            Filter forward and backwards to preserve phase,
            default: `False`.

        analog : `bool`, optional
            If `True`, filter coefficients will be converted from Hz
            to Z-domain digital representation, default: `False`.

        inplace : `bool`, optional
            If `True`, this array will be overwritten with the filtered
            version, default: `False`.


        unit : `str`, optional
            For analogue ZPK filters, the units in which the zeros and poles are
            specified. Either ``'Hz'`` or ``'rad/s'`` (default).

        normalize_gain : `bool`, optional
            Whether to normalize the gain when converting from Hz to rad/s.

            - `False` (default):
              Multiply zeros/poles by -2π but leave gain unchanged.
              This matches the LIGO GDS **'f' plane** convention
              (``plane='f'`` in ``s2z()``).

            - `True`:
              Normalize gain to preserve frequency response magnitude.
              Gain is scaled by :math:`|∏p_i/∏z_i| · (2π)^{(n_p - n_z)}`.
              Use this when your filter was designed with the transfer
              function :math:`H(f) = k·∏(f-z_i)/∏(f-p_i)` in Hz.
              This matches the LIGO GDS **'n' plane** convention
              (``plane='n'`` in ``s2z()``).

            Only used for analogue filters in Hz (``analog=True, unit="Hz"``).

        kwargs
            Other keyword arguments are passed to the filter method.

        Returns
        -------
        result : `TimeSeries`
            The filtered version of the input `TimeSeries`.

        Notes
        -----
        IIR filters are converted into digital cascading second-order sections
        before being applied to the data.

        FIR filters are passed directly to :func:`scipy.signal.lfilter` or
        :func:`scipy.signal.filtfilt` without any conversions.

        See Also
        --------
        scipy.signal.sosfilt
            For details on filtering with second-order sections.

        scipy.signal.sosfiltfilt
            For details on forward-backward filtering with second-order sections

        scipy.signal.lfilter
            For details on filtering (without SOS).

        scipy.signal.filtfilt
            For details on forward-backward filtering (without SOS).

        Raises
        ------
        ValueError
            If ``filt`` arguments cannot be interpreted properly.

        Examples
        --------
        We can design an arbitrarily complicated filter using
        :mod:`gwpy.signal.filter_design`

        >>> from gwpy.signal import filter_design
        >>> bp = filter_design.bandpass(50, 250, 4096.)
        >>> notches = [filter_design.notch(f, 4096.) for f in (60, 120, 180)]
        >>> zpk = filter_design.concatenate_zpks(bp, *notches)

        And then can download some data from GWOSC to apply it using
        `TimeSeries.filter`:

        >>> from gwpy.timeseries import TimeSeries
        >>> data = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
        >>> filtered = data.filter(zpk, filtfilt=True)

        We can plot the original signal, and the filtered version, cutting
        off either end of the filtered data to remove filter-edge artefacts

        >>> from gwpy.plot import Plot
        >>> plot = Plot(data, filtered[128:-128], separate=True)
        >>> plot.show()
        """
        # prepare filter (handles parsing, unit conversion,
        # digital conversion, gain extraction)
        filter_ = filter_design.prepare_digital_filter(
            filt,
            analog=analog,
            sample_rate=self.sample_rate,
            unit=unit or "rad/s",
            normalize_gain=normalize_gain,
            output="zpk",
        )

        return self._filter(
            "zpk",
            filter_,
            filtfilt=filtfilt,
            **kwargs,
        )

    def _filter(
        self,
        form: Literal["ba", "sos", "zpk"],
        filter_: BAType | TapsType | ZpkType | SosType,
        *,
        filtfilt: bool = False,
        **kwargs,
    ) -> Self:
        """Apply a prepared filter to this TimeSeries.

        This method just applies a digital filter,
        handling conversion to second-order-sections as necessary,
        and supporting zero-phase ``filtfilt``.

        Parameters
        ----------
        form : `str`
            The filter form, one of ``'zpk'``, ``'ba'``, or ``'sos'``.

        filter_ : filter arguments
            The filter coefficients in the specified form.

        filtfilt : `bool`, optional
            If `True`, apply the filter using a forward-backward
            filter design, otherwise apply the filter in a single pass.
            Defaults to `False`.

        kwargs
            Other keyword arguments are passed to the filter method.

        Returns
        -------
        result : `TimeSeries`
            The filtered time series with gain applied.
        """
        # Transform ZPK to SOS
        sos = None
        if form != "ba":
            filter_ = cast("ZpkType | SosType", filter_)
            sos = filter_design._transform(
                filter_,
                form,
                "sos",
            )

        # perform filter
        kwargs.setdefault("axis", 0)
        if sos is None and filtfilt:
            out = signal.filtfilt(*filter_, self, **kwargs)
        elif sos is None:
            out = signal.lfilter(*filter_, self, **kwargs)
        elif filtfilt:
            out = signal.sosfiltfilt(sos, self, **kwargs)
        else:
            out = signal.sosfilt(sos, self, **kwargs)

        # format as type(self)
        new = cast("Self", out.view(type(self)))
        new.__metadata_finalize__(self)
        new._unit = self.unit
        return new

    def transfer_function(
        self,
        other: TimeSeries,
        fftlength: float | None = None,
        overlap: float | None = None,
        window: WindowLike = "hann",
        average: str = "mean",
        **kwargs,
    ) -> FrequencySeries:
        """Calculate the transfer function between this `TimeSeries` and another.

        This `TimeSeries` is the 'A-channel', serving as the reference
        (denominator) while the other time series is the test (numerator)

        Parameters
        ----------
        other : `TimeSeries`
            `TimeSeries` signal to calculate the transfer function with.

        fftlength : `float`, optional
            Number of seconds in single FFT, defaults to a single FFT
            covering the full duration.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        average : `str`, optional
            FFT-averaging method (default: ``'mean'``) passed to
            underlying csd() and psd() methods.

        kwargs
            Any other keyword arguments accepted by
            :meth:`TimeSeries.csd` or :meth:`TimeSeries.psd`.

        Returns
        -------
        transfer_function : `~gwpy.frequencyseries.FrequencySeries`
            The transfer function `FrequencySeries` of this `TimeSeries`
            with the other.

        Notes
        -----
        If `self` and `other` have difference
        :attr:`TimeSeries.sample_rate` values, the higher sampled
        `TimeSeries` will be down-sampled to match the lower.
        """
        csd = self.csd(
            other,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            average=average,
            **kwargs,
        )
        psd = self.psd(
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            average=average,
            **kwargs,
        )

        # Take the minimum of the frequencyseries csd and psd because the
        # sample rate of different channels might yield different length
        # objects
        size = min(csd.size, psd.size)

        return csd[:size] / psd[:size]

    def coherence(
        self,
        other: TimeSeries,
        fftlength: float | None = None,
        overlap: float | None = None,
        window: WindowLike = "hann",
        **kwargs,
    ) -> FrequencySeries:
        """Calculate the frequency-coherence between this `TimeSeries` and another.

        Parameters
        ----------
        other : `TimeSeries`
            `TimeSeries` signal to calculate coherence with

        fftlength : `float`, optional
            Number of seconds in single FFT, defaults to a single FFT
            covering the full duration.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        kwargs
            Any other keyword arguments accepted by
            :func:`matplotlib.mlab.cohere` except ``NFFT``, ``window``,
            and ``noverlap`` which are superceded by the above keyword
            arguments.

        Returns
        -------
        coherence : `~gwpy.frequencyseries.FrequencySeries`
            The coherence `FrequencySeries` of this `TimeSeries`
            with the other.

        Notes
        -----
        If `self` and `other` have difference
        :attr:`TimeSeries.sample_rate` values, the higher sampled
        `TimeSeries` will be down-sampled to match the lower.

        See Also
        --------
        scipy.signal.coherence
            For details of the coherence calculator.
        """
        return spectral.psd(
            (self, other),
            spectral.coherence,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            **kwargs,
        )

    def auto_coherence(
        self,
        dt: float,
        fftlength: float | None = None,
        overlap: float | None = None,
        window: WindowLike = "hann",
        **kwargs,
    ) -> FrequencySeries:
        """Calculate the coherence between this series and a shifted copy of itself.

        The standard :meth:`TimeSeries.coherence` is calculated between
        the input `TimeSeries` and a :meth:`cropped <TimeSeries.crop>`
        copy of itself. Since the cropped version will be shorter, the
        input series will be shortened to match.

        Parameters
        ----------
        dt : `float`
            Duration (in seconds) of time-shift.

        fftlength : `float`, optional
            Number of seconds in single FFT, defaults to a single FFT
            covering the full duration.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        kwargs
            Any other keyword arguments accepted by
            :func:`matplotlib.mlab.cohere` except ``NFFT``, ``window``,
            and ``noverlap`` which are superceded by the above keyword
            arguments.

        Returns
        -------
        coherence : `~gwpy.frequencyseries.FrequencySeries`
            The coherence `FrequencySeries` of this `TimeSeries`
            with the other.

        Notes
        -----
        The :meth:`TimeSeries.auto_coherence` will perform best when
        ``dt`` is approximately ``fftlength / 2``.

        See Also
        --------
        matplotlib.mlab.cohere
            For details of the coherence calculator.
        """
        # shifting self backwards is the same as forwards
        dt = abs(dt)
        # crop inputs
        self_ = self.crop(self.span[0], self.span[1] - dt)
        other = self.crop(self.span[0] + dt, self.span[1])
        return self_.coherence(
            other,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            **kwargs,
        )

    def coherence_spectrogram(
        self,
        other: TimeSeries,
        stride: float,
        fftlength: float | None = None,
        overlap: float | None = None,
        window: WindowLike = "hann",
        nproc: int = 1,
    ) -> Spectrogram:
        """Calculate the coherence spectrogram between this `TimeSeries` and other.

        Parameters
        ----------
        other : `TimeSeries`
            The second `TimeSeries` in this CSD calculation.

        stride : `float`
            Number of seconds in single PSD (column of spectrogram).

        fftlength : `float`
            Number of seconds in single FFT.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        nproc : `int`
            Number of parallel processes to use when calculating
            individual coherence spectra.

        Returns
        -------
        spectrogram : `~gwpy.spectrogram.Spectrogram`
            Time-frequency coherence spectrogram as generated from the
            input time-series.
        """
        from ..spectrogram.coherence import _from_timeseries_parallel

        return _from_timeseries_parallel(
            self,
            other,
            stride,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            parallel=nproc,
        )

    def rms(
        self,
        stride: float = 1,
    ) -> TimeSeries:
        """Calculate the root-mean-square value of this `TimeSeries` once per stride.

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
        name = f"{self.name} {stride}-second RMS"
        return self.__class__(
            data,
            channel=self.channel,
            t0=self.t0,
            name=name,
            sample_rate=(1/float(stride)),
        )

    def mask(
        self,
        deadtime: SegmentList | None = None,
        flag: str | None = None,
        *,
        query_open_data: bool = False,
        const: float = numpy.nan,
        tpad: float = 0.5,
        inplace: bool = False,
        **kwargs,
    ) -> Self:
        """Mask portions of this `TimeSeries` that fall within a given list of segments.

        Parameters
        ----------
        deadtime : `SegmentList`, optional
            A list of time segments defining the deadtime (i.e., masked
            portions) of the output, will supersede `flag` if given.

        flag : `str`, optional
            The name of a data-quality flag for which to query, required if
            `deadtime` is not given.

        query_open_data : `bool`, optional
            If `True`, will query for publicly released data-quality segments
            through the Gravitational-wave Open Science Center (GWOSC).

        const : `float`, optional
            Constant value with which to mask deadtime data.

        tpad : `float`, optional
            Length of time (in seconds) over which to taper off data at
            mask segment boundaries.

        inplace : `bool`, optional
            If `True`, this array will be overwritten with the masked
            version, otherwise (`False`, default) a modified copy will be
            returned.

        kwargs : `dict`, optional
            Additional keyword arguments to
            `~gwpy.segments.DataQualityFlag.query` or
            `~gwpy.segments.DataQualityFlag.fetch_open_data`,
            see "Notes" below.

        Returns
        -------
        out : `TimeSeries`
            The masked version of this `TimeSeries`.

        Notes
        -----
        If `tpad` is nonzero, the Tukey (tapered cosine) window is used to
        smoothly ramp data down to zero over a timescale `tpad` approaching
        every segment boundary in `deadtime`. However, this does not apply to
        the left or right bounds of the original `TimeSeries`.

        The `deadtime` segment list will always be coalesced and restricted to
        the limits of `self.span`. In particular, when querying a data-quality
        flag, this means the `start` and `end` arguments to
        `~gwpy.segments.DataQualityFlag.query` will effectively be reset and
        therefore need not be given.

        If `flag` is interpreted positively, i.e. if `flag` being active
        corresponds to a "good" state, then its complement in `self.span`
        will be used to define the deadtime for masking.

        See Also
        --------
        gwpy.segments.DataQualityFlag.query
            For the method to query segments of a given data-quality flag.

        gwpy.segments.DataQualityFlag.fetch_open_data
            For the method to query data-quality flags from the GWOSC database.

        scipy.signal.windows.tukey
            For the Tukey (tapered cosine) window used for tapering.
        """
        if deadtime is None and flag is None:
            msg = "either 'deadtime' or 'flag' must be given"
            raise ValueError(msg)

        # Format arguments
        query_method = (
            DataQualityFlag.fetch_open_data if query_open_data
            else DataQualityFlag.query
        )
        (tstart, tend) = cast("tuple[float, float]", self.span)
        span = SegmentList([self.span])
        sample = self.sample_rate.to("Hz").value
        npad = int(tpad * sample)

        # Prepare output TimeSeries
        if inplace:
            out = self
        else:
            out = self.copy()

        # Query the requested data-quality flag
        if deadtime is None:
            start = kwargs.pop("start", tstart)
            end = kwargs.pop("end", tend)
            dqflag = query_method(
                str(flag),
                start,
                end,
                **kwargs,
            )
            deadtime = ~dqflag.active if dqflag.isgood else dqflag.active

        # Identify timestamps and mask out
        deadtime = (deadtime & span).coalesce()
        timestamps = tstart + numpy.arange(self.size) / sample
        (masked, ) = numpy.nonzero([t in deadtime for t in timestamps])
        out.value[masked] = const

        # Create tukey window
        window_full = tukey(2*npad, alpha=1.0)

        # Taper off at segment boundaries, being careful not to taper
        # at either edge of the original TimeSeries, and to cut off
        # the taper window in segments that are too short
        for seg in (span - deadtime):
            k = int((seg[0] - tstart) * sample)
            n = math.ceil(abs(seg) * sample)
            nhalf = min(npad, n)
            taper_left = (k != 0)  # taper left edge if not at start
            taper_right = (k + n != self.size)  # taper right edge if not at end
            if taper_left:
                out[k:k+nhalf] *= window_full[:nhalf]
            if taper_right:
                out[k+n-nhalf:k+n] *= window_full[-nhalf:]
        return out

    @overload
    def demodulate(
        self,
        f: float,
        stride: float = 1,
        *,
        exp: Literal[False] = ...,
        deg: bool = ...,
    ) -> tuple[TimeSeries, TimeSeries]: ...

    @overload
    def demodulate(
        self,
        f: float,
        stride: float = 1,
        *,
        exp: Literal[True],
        deg: bool = ...,
    ) -> TimeSeries: ...

    def demodulate(
        self,
        f: float,
        stride: float = 1,
        *,
        exp: bool = False,
        deg: bool = True,
    ) -> tuple[TimeSeries, TimeSeries] | TimeSeries:
        """Compute the average magnitude and phase of this `TimeSeries`.

        Parameters
        ----------
        f : `float`
            Frequency (Hz) at which to demodulate the signal.

        stride : `float`, optional
            Stride (seconds) between calculations.

        exp : `bool`, optional
            Return the magnitude and phase trends as one `TimeSeries` object
            representing a complex exponential.

        deg : `bool`, optional
            If `exp=False`, calculates the phase in degrees.

        Returns
        -------
        mag, phase : `TimeSeries`
            If `exp=False`, returns a pair of `TimeSeries` objects representing
            magnitude and phase trends with `dt=stride`.

        out : `TimeSeries`
            If `exp=True`, returns a single `TimeSeries` with magnitude and
            phase trends represented as `mag * exp(1j*phase)` with `dt=stride`.

        Examples
        --------
        Demodulation is useful when trying to examine steady sinusoidal
        signals we know to be contained within data. For instance,
        we can download some data from GWOSC to look at trends of the
        amplitude and phase of LIGO Livingston's calibration line at 331.3 Hz:

        >>> from gwpy.timeseries import TimeSeries
        >>> data = TimeSeries.fetch_open_data('L1', 1131350417, 1131357617)

        We can demodulate the `TimeSeries` at 331.3 Hz with a stride of one
        minute:

        >>> amp, phase = data.demodulate(331.3, stride=60)

        We can then plot these trends to visualize fluctuations in the
        amplitude of the calibration line:

        >>> from gwpy.plot import Plot
        >>> plot = Plot(amp)
        >>> ax = plot.gca()
        >>> ax.set_ylabel('Strain Amplitude at 331.3 Hz')
        >>> plot.show()

        See Also
        --------
        TimeSeries.heterodyne
            for the underlying heterodyne detection method
        """
        # stride through the TimeSeries and heterodyne at a single frequency
        phase = (
            2 * numpy.pi * f
            * self.dt.decompose().value
            * numpy.arange(0, self.size)
        )
        out = self.heterodyne(phase, stride=stride, singlesided=True)

        if exp:
            return out

        mag = out.abs()
        phase = type(mag)(numpy.angle(out, deg=deg))
        phase.__array_finalize__(out)
        phase.override_unit("deg" if deg else "rad")
        return (mag, phase)

    def heterodyne(
        self,
        phase: ArrayLike,
        stride: float = 1,
        *,
        singlesided: bool = False,
    ) -> TimeSeries:
        """Compute the heterodyned average magnitude and phase of this `TimeSeries`.

        Parameters
        ----------
        phase : `array_like`
            An array of phase measurements (radians) with which to
            heterodyne the signal.

        stride : `float`, optional
            Stride (seconds) between calculations.

        singlesided : `bool`, optional
            Boolean switch to return single-sided output (i.e., to multiply by
            2 so that the signal is distributed across positive frequencies
            only).

        Returns
        -------
        out : `TimeSeries`
            Magnitude and phase trends, represented as
            `mag * exp(1j*phase)` with `dt=stride`.

        Notes
        -----
        This is similar to the :meth:`~gwpy.timeseries.TimeSeries.demodulate`
        method, but is more general in that it accepts a varying phase
        evolution, rather than a fixed frequency.

        Unlike :meth:`~gwpy.timeseries.TimeSeries.demodulate`, the complex
        output is double-sided by default, so is not multiplied by 2.

        Examples
        --------
        Heterodyning can be useful in analysing quasi-monochromatic signals
        with a known phase evolution, such as continuous-wave signals
        from rapidly rotating neutron stars. These sources radiate at a
        frequency that slowly decreases over time, and is Doppler modulated
        due to the Earth's rotational and orbital motion.

        To see an example of heterodyning in action, we can simulate a signal
        whose phase evolution is described by the frequency and its first
        derivative with respect to time. We can download some O1 era
        LIGO-Livingston data from GWOSC, inject the simulated signal, and
        recover its amplitude.

        >>> from gwpy.timeseries import TimeSeries
        >>> data = TimeSeries.fetch_open_data('L1', 1131350417, 1131354017)

        We now need to set the signal parameters, generate the expected
        phase evolution, and create the signal:

        >>> import numpy
        >>> f0 = 123.456789  # signal frequency (Hz)
        >>> fdot = -9.87654321e-7  # signal frequency derivative (Hz/s)
        >>> fepoch = 1131350417  # phase epoch
        >>> amp = 1.5e-22  # signal amplitude
        >>> phase0 = 0.4  # signal phase at the phase epoch
        >>> times = data.times.value - fepoch
        >>> phase = 2 * numpy.pi * (f0 * times + 0.5 * fdot * times**2)
        >>> signal = TimeSeries(amp * numpy.cos(phase + phase0),
        >>>                     sample_rate=data.sample_rate, t0=data.t0)
        >>> data = data.inject(signal)

        To recover the signal, we can bandpass the injected data around the
        signal frequency, then heterodyne using our phase model with a stride
        of 60 seconds:

        >>> filtdata = data.bandpass(f0 - 0.5, f0 + 0.5)
        >>> het = filtdata.heterodyne(phase, stride=60, singlesided=True)

        We can then plot signal amplitude over time (cropping the first two
        minutes to remove the filter response):

        >>> plot = het.crop(het.x0.value + 180).abs().plot()
        >>> ax = plot.gca()
        >>> ax.set_ylabel("Strain amplitude")
        >>> plot.show()

        See Also
        --------
        TimeSeries.demodulate
            for a method to heterodyne at a fixed frequency
        """
        phase = numpy.asarray(phase)  # make sure phase is a numpy array
        if phase.ndim != 1:
            msg = "Phase is not array_like"
            raise ValueError(msg)
        if len(phase) != len(self):
            msg = "Phase array must be the same length as the TimeSeries"
            raise ValueError(msg)

        stridesamp = int(stride * self.sample_rate.value)
        nsteps = int(self.size // stridesamp)
        # stride through the TimeSeries and heterodyne
        out = type(self)(numpy.zeros(nsteps, dtype=complex))
        out.__array_finalize__(self)
        out.sample_rate = 1 / float(stride)
        for step in range(nsteps):
            istart = int(stridesamp * step)
            iend = istart + stridesamp
            idx = numpy.arange(istart, iend)
            mixed = numpy.exp(-1j * phase[idx]) * self.value[idx]
            out.value[step] = 2 * mixed.mean() if singlesided else mixed.mean()
        return out

    def taper(
        self,
        side: Literal["left", "right", "leftright"] = "leftright",
        duration: float | None = None,
        nsamples: int | None = None,
    ) -> Self:
        """Taper the ends of this `TimeSeries` smoothly to zero.

        Parameters
        ----------
        side : `str`, optional
            The side of the `TimeSeries` to taper, must be one of `'left'`,
            `'right'`, or `'leftright'`.

        duration : `float`, optional
            The duration of time to taper, will override `nsamples`
            if both are provided as arguments.

        nsamples : `int`, optional
            The number of samples to taper, will be overridden by `duration`
            if both are provided as arguments.

        Returns
        -------
        out : `TimeSeries`
            A copy of `self` tapered at one or both ends.

        Raises
        ------
        ValueError
            If `side` is not one of ``('left', 'right', 'leftright')``.

        Examples
        --------
        To see the effect of the Tukey (tapered cosine) window, we can taper a
        sinusoidal `TimeSeries` at both ends:

        >>> import numpy
        >>> from gwpy.timeseries import TimeSeries
        >>> t = numpy.linspace(0, 1, 2048)
        >>> series = TimeSeries(numpy.cos(10.5*numpy.pi*t), times=t)
        >>> tapered = series.taper()

        We can plot it to see how the ends now vary smoothly from 0 to 1:

        >>> from gwpy.plot import Plot
        >>> plot = Plot(series, tapered, separate=True, sharex=True)
        >>> plot.show()

        Notes
        -----
        The :meth:`TimeSeries.taper` automatically tapers from the second
        stationary point (local maximum or minimum) on the specified side
        of the input. However, the method will never taper more than half
        the full width of the `TimeSeries`, and will fail if there are no
        stationary points.

        See :func:`scipy.signal.windows.tukey` for the Tukey (tapered cosine)
        window used for tapering, and see :func:`scipy.signal.get_window`
        for other common window formats.
        """
        # check window properties
        if side not in ("left", "right", "leftright"):
            msg = "side must be one of 'left', 'right', or 'leftright'"
            raise ValueError(msg)

        if duration:
            nsamples = int(duration * self.sample_rate.to("Hz").value)

        out = self.copy()
        # if a duration or number of samples is not specified, automatically
        # identify the second stationary point away from each boundary,
        # else default to half the TimeSeries width
        nleft, nright = 0, 0
        if not nsamples:
            mini, = signal.argrelmin(out.value)
            maxi, = signal.argrelmax(out.value)
        if "left" in side:
            nleft = nsamples or max(mini[0], maxi[0])
            nleft = min(nleft, int(self.size/2))
        if "right" in side:
            nright = nsamples or out.size - min(mini[-1], maxi[-1])
            nright = min(nright, int(self.size/2))

        # Create tukey window and apply the appropriate portions
        nmax = max(nleft, nright)
        if nmax > 0:
            window_full = tukey(2 * nmax, alpha=1.0)
            if "left" in side and nleft > 0:
                out.value[:nleft] *= window_full[:nleft]
            if "right" in side and nright > 0:
                out.value[-nright:] *= window_full[-nright:]
        return out

    def whiten(
        self,
        fftlength: float | None = None,
        overlap: float = 0,
        method: str = DEFAULT_FFT_METHOD,
        window: WindowLike = "hann",
        detrend: Literal["linear", "constant"] = "constant",
        asd: FrequencySeries | None = None,
        fduration: float = 2,
        highpass: float | None = None,
        **kwargs,
    ) -> Self:
        """Whiten this `TimeSeries` using inverse spectrum truncation.

        Parameters
        ----------
        fftlength : `float`, optional
            FFT integration length (in seconds) for ASD estimation.

        overlap : `float`, optional
            Number of seconds of overlap between FFTs, defaults to the
            recommended overlap for the given window (if given), or 0.

        method : `str`, optional
            FFT-averaging method (default: ``'median'``).

        window : `str`, `numpy.ndarray`, optional
            Window function to apply to timeseries prior to FFT,
            see :func:`scipy.signal.get_window` for details on acceptable
            formats.

        detrend : `str`, optional
            Type of detrending to do before FFT (see `~TimeSeries.detrend`
            for more details).

        asd : `~gwpy.frequencyseries.FrequencySeries`, optional
            The amplitude spectral density using which to whiten the data,
            overrides other ASD arguments, default: `None`.

        fduration : `float`, optional
            Duration (in seconds) of the time-domain FIR whitening filter,
            must be no longer than `fftlength`, default: 2 seconds.

        highpass : `float`, optional
            Highpass corner frequency (in Hz) of the FIR whitening filter.

        kwargs
            Other keyword arguments are passed to the `TimeSeries.asd`
            method to estimate the amplitude spectral density
            `FrequencySeries` of this `TimeSeries`.

        Returns
        -------
        out : `TimeSeries`
            A whitened version of the input data with zero mean and unit variance.

        See Also
        --------
        TimeSeries.asd
            For details on the ASD calculation.

        TimeSeries.convolve
            For details on convolution with the overlap-save method.

        gwpy.signal.filter_design.fir_from_transfer
            For FIR filter design through spectrum truncation.

        Notes
        -----
        The accepted ``method`` arguments are:

        - ``'bartlett'`` : a mean average of non-overlapping periodograms
        - ``'median'`` : a median average of overlapping periodograms
        - ``'welch'`` : a mean average of overlapping periodograms

        The ``window`` argument is used in ASD estimation, FIR filter design,
        and in preventing spectral leakage in the output.

        Due to filter settle-in, a segment of length ``0.5*fduration`` will be
        corrupted at the beginning and end of the output. See
        `~TimeSeries.convolve` for more details.

        The input is detrended and the output normalised such that, if the
        input is stationary and Gaussian, then the output will have zero mean
        and unit variance.

        For more on inverse spectrum truncation, see arXiv:gr-qc/0509116.
        """
        # compute the ASD
        fftlength = fftlength if fftlength else _fft_length_default(self.dt)
        if asd is None:
            asd = self.asd(
                fftlength,
                overlap=overlap,
                method=method,
                window=window,
                **kwargs,
            )
        asd = asd.interpolate(1./self.duration.decompose().value)
        # design whitening filter, with highpass if requested
        ncorner = int(highpass / asd.df.decompose().value) if highpass else 0
        if isinstance(window, str | tuple):
            ntaps = int((fduration * self.sample_rate).decompose().value)
        else:
            window = numpy.asarray(window)
            ntaps = len(window)
        tdw = filter_design.fir_from_transfer(
            1/asd.value,
            ntaps=ntaps,
            window=window,
            ncorner=ncorner,
        )
        # condition the input data and apply the whitening filter
        in_ = self.copy().detrend(detrend)
        out = in_.convolve(tdw, window=window)
        return out * numpy.sqrt(2 * in_.dt.decompose().value)

    def find_gates(
        self,
        tzero: float = 1.0,
        *,
        whiten: bool = True,
        threshold: float = 50.0,
        cluster_window: float = 0.5,
        **whiten_kwargs,
    ) -> SegmentList:
        """Identify points that should be gates using a provided threshold.

        This method identifies points in the `TimeSeries` that exceed a
        provided threshold, and returns a list of segments that should be
        gated. The gating points are clustered within a provided time window.

        This method is useful for identifying high amplitude peaks in the
        data that should be removed or masked out.

        Parameters
        ----------
        tzero : `int`, optional
            Half-width time duration (seconds) in which the timeseries is
            set to zero.

        whiten : `bool`, optional
            If True, data will be whitened before gating points are discovered,
            use of this option is highly recommended.

        threshold : `float`, optional
            Amplitude threshold, if the data exceeds this value a gating window
            will be placed.

        cluster_window : `float`, optional
            Time duration (seconds) over which gating points will be clustered.

        whiten_kwargs
            Other keyword arguments will be passed to the `TimeSeries.whiten`
            method if it is being used when discovering gating points.

        Returns
        -------
        out : `~gwpy.segments.SegmentList`
            A list of segments that should be gated based on the
            provided parameters.

        See Also
        --------
        TimeSeries.gate
            For a method that applies the identified gates.
        """
        # Find points to gate based on a threshold
        sample = self.sample_rate.to("Hz").value
        data = self.whiten(**whiten_kwargs) if whiten else self
        window_samples = cluster_window * sample
        gates = signal.find_peaks(
            abs(data.value),
            height=threshold,
            distance=window_samples,
        )[0]
        # represent gates as time segments
        return SegmentList([Segment(
            self.t0.value + (k / sample) - tzero,
            self.t0.value + (k / sample) + tzero,
        ) for k in gates]).coalesce()

    def gate(
        self,
        tzero: float = 1.0,
        tpad: float = 0.5,
        *,
        whiten: bool = True,
        threshold: float = 50.0,
        cluster_window: float = 0.5,
        **whiten_kwargs,
    ) -> Self:
        """Remove high amplitude peaks from data using inverse Tukey window.

        Points will be discovered automatically using a provided threshold
        and clustered within a provided time window.

        Parameters
        ----------
        tzero : `int`, optional
            Half-width time duration (seconds) in which the timeseries is
            set to zero.

        tpad : `int`, optional
            Half-width time duration (seconds) in which the Tukey window
            is tapered.

        whiten : `bool`, optional
            If True, data will be whitened before gating points are discovered,
            use of this option is highly recommended.

        threshold : `float`, optional
            Amplitude threshold, if the data exceeds this value a gating window
            will be placed.

        cluster_window : `float`, optional
            Time duration (seconds) over which gating points will be clustered.

        whiten_kwargs
            Other keyword arguments will be passed to the `TimeSeries.whiten`
            method if it is being used when discovering gating points.

        Returns
        -------
        out : `~gwpy.timeseries.TimeSeries`
            A copy of the original `TimeSeries` that has had gating windows applied.

        Examples
        --------
        Read data into a `TimeSeries`

        >>> from gwpy.timeseries import TimeSeries
        >>> data = TimeSeries.fetch_open_data('H1', 1135148571, 1135148771)

        Apply gating using custom arguments

        >>> gated = data.gate(
        ...     tzero=1.0,
        ...     tpad=1.0,
        ...     threshold=10.0,
        ...     fftlength=4,
        ...     overlap=2,
        ...     method='median',
        ... )

        Plot the original data and the gated data, whiten both for
        visualization purposes

        >>> overlay = data.whiten(4,2,method="median").plot(
        ...     dpi=150,
        ...     label="Ungated",
        ...     color="dodgerblue",
        ...     zorder=2,
        ... )
        >>> ax = overlay.gca()
        >>> ax.plot(
        ...     gated.whiten(4, 2, method="median"),
        ...     label="Gated",
        ...     color="orange",
        ...     zorder=3,
        ... )
        >>> ax.set_xlim(1135148661, 1135148681)
        >>> ax.legend()
        >>> overlay.show()

        See Also
        --------
        TimeSeries.mask
            For the method that masks out unwanted data.

        TimeSeries.find_gates
            For the method that identifies gating points.

        TimeSeries.whiten
            For the whitening filter used to identify gating points.
        """
        gates = self.find_gates(
            tzero=tzero,
            whiten=whiten,
            threshold=threshold,
            cluster_window=cluster_window,
            **whiten_kwargs,
        )
        return self.mask(deadtime=gates, const=0, tpad=tpad)

    def convolve(
        self,
        fir: numpy.ndarray,
        window: WindowLike = "hann",
    ) -> Self:
        """Convolve this `TimeSeries` with an FIR filter using the overlap-save method.

        Parameters
        ----------
        fir : `numpy.ndarray`
            The time domain filter to convolve with.

        window : `str`, optional
            Window function to apply to boundaries, default: ``'hann'``
            see :func:`scipy.signal.get_window` for details on acceptable formats.

        Returns
        -------
        out : `TimeSeries`
            The result of the convolution.

        See Also
        --------
        scipy.signal.fftconvolve
            For details on the convolution scheme used here.

        TimeSeries.filter
            For an alternative method designed for short filters.

        Notes
        -----
        The output `TimeSeries` is the same length and has the same timestamps
        as the input.

        Due to filter settle-in, a segment half the length of `fir` will be
        corrupted at the left and right boundaries. To prevent spectral leakage
        these segments will be windowed before convolving.
        """
        pad = int(numpy.ceil(fir.size/2))
        nfft = min(8*fir.size, self.size)
        # condition the input data
        in_ = self.copy()
        window = get_window(window, fir.size)
        in_.value[:pad] *= window[:pad]
        in_.value[-pad:] *= window[-pad:]
        # if FFT length is long enough, perform only one convolution
        if nfft >= self.size/2:
            conv = signal.fftconvolve(in_.value, fir, mode="same")
        # else use the overlap-save algorithm
        else:
            nstep = nfft - 2*pad
            conv = numpy.zeros(self.size)
            # handle first chunk separately
            conv[:nfft-pad] = signal.fftconvolve(
                in_.value[:nfft],
                fir,
                mode="same",
            )[:nfft-pad]
            # process chunks of length nstep
            k = nfft - pad
            while k < self.size - nfft + pad:
                yk = signal.fftconvolve(
                    in_.value[k-pad:k+nstep+pad],
                    fir,
                    mode="same",
                )
                conv[k:k+yk.size-2*pad] = yk[pad:-pad]
                k += nstep
            # handle last chunk separately
            conv[-nfft+pad:] = signal.fftconvolve(
                in_.value[-nfft:],
                fir,
                mode="same",
            )[-nfft+pad:]
        out = type(self)(conv)
        out.__array_finalize__(self)
        return out

    def correlate(
        self,
        mfilter: TimeSeries,
        window: WindowLike = "hann",
        detrend: Literal["linear", "constant"] = "linear",
        *,
        whiten: bool = False,
        wduration: float = 2,
        highpass: float | None = None,
        **asd_kw,
    ) -> TimeSeries:
        """Cross-correlate this `TimeSeries` with another signal.

        Parameters
        ----------
        mfilter : `TimeSeries`
            the time domain signal to correlate with

        window : `str`, optional
            window function to apply to timeseries prior to FFT,
            default: ``'hann'``
            see :func:`scipy.signal.get_window` for details on acceptable
            formats

        detrend : `str`, optional
            type of detrending to do before FFT (see `~TimeSeries.detrend`
            for more details), default: ``'linear'``

        whiten : `bool`, optional
            boolean switch to enable (`True`) or disable (`False`) data
            whitening, default: `False`

        wduration : `float`, optional
            duration (in seconds) of the time-domain FIR whitening filter,
            only used if `whiten=True`, defaults to 2 seconds

        highpass : `float`, optional
            highpass corner frequency (in Hz) of the FIR whitening filter,
            only used if `whiten=True`, default: `None`

        **asd_kw
            keyword arguments to pass to `TimeSeries.asd` to generate
            an ASD, only used if `whiten=True`

        Returns
        -------
        snr : `TimeSeries`
            the correlated signal-to-noise ratio (SNR) timeseries

        See Also
        --------
        TimeSeries.asd
            for details on the ASD calculation
        TimeSeries.convolve
            for details on convolution with the overlap-save method

        Notes
        -----
        The `window` argument is used in ASD estimation, whitening, and
        preventing spectral leakage in the output. It is not used to condition
        the matched-filter, which should be windowed before passing to this
        method.

        Due to filter settle-in, a segment half the length of `mfilter` will be
        corrupted at the beginning and end of the output. See
        `~TimeSeries.convolve` for more details.

        The input and matched-filter will be detrended, and the output will be
        normalised so that the SNR measures number of standard deviations from
        the expected mean.
        """
        self.is_compatible(mfilter)
        # condition data
        if whiten is True:
            fftlength = asd_kw.pop(
                "fftlength",
                _fft_length_default(self.dt),
            )
            overlap = asd_kw.pop("overlap", None)
            if overlap is None and not isinstance(window, str):
                warnings.warn(
                    "custom window given without overlap, defaulting to zero, "
                    "please set overlap=0 explicitly if this is intended",
                    category=UserWarning,
                    stacklevel=2,
                )
                overlap = 0
            elif overlap is None:
                window = cast("str", window)
                overlap = recommended_overlap(window) * fftlength
            asd = self.asd(fftlength, overlap, window=window, **asd_kw)
            # pad the matched-filter to prevent corruption
            npad = int(wduration * mfilter.sample_rate.decompose().value / 2)
            mfilter = mfilter.pad(npad)
            # whiten (with errors on division by zero)
            with numpy.errstate(all="raise"):
                in_ = self.whiten(
                    window=window,
                    fduration=wduration,
                    asd=asd,
                    highpass=highpass,
                    detrend=detrend,
                )
                mfilter = mfilter.whiten(
                    window=window,
                    fduration=wduration,
                    asd=asd,
                    highpass=highpass,
                    detrend=detrend,
                )[npad:-npad]
        else:
            in_ = self.detrend(detrend)
            mfilter = mfilter.detrend(detrend)
        # compute matched-filter SNR and normalise
        stdev = numpy.sqrt((mfilter.value**2).sum())
        snr = in_.convolve(mfilter[::-1], window=window) / stdev
        snr.__array_finalize__(self)
        return snr

    def detrend(
        self,
        detrend: Literal["constant", "linear"] = "constant",
    ) -> Self:
        """Remove the trend from this `TimeSeries`.

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
        data = cast("Self", signal.detrend(self.value, type=detrend).view(type(self)))
        data.__metadata_finalize__(self)
        data._unit = self.unit  # noqa: SLF001
        return data

    def notch(
        self,
        frequency: QuantityLike,
        type: Literal["iir"] = "iir",  # noqa: A002
        *,
        filtfilt: bool = True,
        **kwargs,
    ) -> Self:
        """Notch out a frequency in this `TimeSeries`.

        Parameters
        ----------
        frequency : `float`, `~astropy.units.Quantity`
            Frequency (default in Hertz) at which to apply the notch.

        type : `str`, optional
            Type of filter to apply, currently only 'iir' is supported.

        filtfilt : `bool`, optional
            Whether to apply zero-phase filtering (default is True).

        kwargs
            Other keyword arguments to pass to `gwpy.signal.filter_design.notch`.

        Returns
        -------
        notched : `TimeSeries`
           A notch-filtered copy of the input `TimeSeries`.

        See Also
        --------
        TimeSeries.filter
            For details on the filtering method.
        gwpy.signal.filter_design.notch
            For details on the IIR filter design method.
        """
        zpk = filter_design.notch(
            frequency,
            self.sample_rate.value,
            type=type,
            output="zpk",
            **kwargs,
        )
        return self._filter(
            "zpk",
            zpk,
            filtfilt=filtfilt,
        )

    def q_gram(
        self,
        qrange: tuple[float, float] = qtransform.DEFAULT_QRANGE,
        frange: tuple[float, float] = qtransform.DEFAULT_FRANGE,
        mismatch: float = qtransform.DEFAULT_MISMATCH,
        snrthresh: float = 5.5,
        **kwargs,
    ) -> EventTable:
        """Scan a `TimeSeries` using the multi-Q transform.

        Parameters
        ----------
        qrange : `tuple` of `float`, optional
            ``(low, high)`` range of Qs to scan.

        frange : `tuple` of `float`, optional
            ``(low, high)`` range of frequencies to scan.

        mismatch : `float`, optional
            Maximum allowed fractional mismatch between neighbouring tiles.

        snrthresh : `float`, optional
            Lower inclusive threshold on individual tile SNR to keep in the table.

        kwargs
            Other keyword arguments to be passed to :meth:`QTiling.transform`,
            including ``'epoch'`` and ``'search'``.

        Returns
        -------
        qgram : `EventTable`
            A table of time-frequency tiles on the most significant `QPlane`.

        See Also
        --------
        TimeSeries.q_transform
            For a method to interpolate the raw Q-transform over a regularly
            gridded spectrogram.

        gwpy.signal.qtransform
            For code and documentation on how the Q-transform is implemented.

        gwpy.table.EventTable.tile
            To render this `EventTable` as a collection of polygons.

        Notes
        -----
        Only tiles with signal energy greater than or equal to
        `snrthresh ** 2 / 2` will be stored in the output `EventTable`. The
        table columns are ``'time'``, ``'duration'``, ``'frequency'``,
        ``'bandwidth'``, and ``'energy'``.
        """
        qscan, _ = qtransform.q_scan(
            self,
            mismatch=mismatch,
            qrange=qrange,
            frange=frange,
            **kwargs,
        )
        return qscan.table(snrthresh=snrthresh)

    def q_transform(
        self,
        qrange: tuple[float, float] = qtransform.DEFAULT_QRANGE,
        frange: tuple[float, float] = qtransform.DEFAULT_FRANGE,
        gps: float | None = None,
        search: float = 0.5,
        tres: str | float = "<default>",
        fres: str | float = "<default>",
        *,
        logf: bool = False,
        norm: str = "median",
        mismatch: float = qtransform.DEFAULT_MISMATCH,
        outseg: Segment | None = None,
        whiten: bool | FrequencySeries = True,
        fduration: float = 2,
        highpass: float | None = None,
        **asd_kw,
    ) -> Spectrogram:
        """Compute the multi-Q transform and return an interpolated spectrogram.

        By default, this method returns a high-resolution spectrogram in
        both time and frequency, which can result in a large memory
        footprint. If you know that you only need a subset of the output
        for, say, a figure, consider using ``outseg`` and the other
        keyword arguments to restrict the size of the returned data.

        Parameters
        ----------
        qrange : `tuple` of `float`, optional
            ``(low, high)`` range of Qs to scan.

        frange : `tuple` of `float`, optional
            ``(log, high)`` range of frequencies to scan.

        gps : `float`, optional
            Central time of interest for determine loudest Q-plane.

        search : `float`, optional
            Window around `gps` in which to find peak energies, only
            used if `gps` is given.

        tres : `float`, optional
            Desired time resolution (seconds) of output `Spectrogram`,
            default is ``abs(outseg) / 1000``.

        fres : `float`, `int`, `None`, optional
            Desired frequency resolution (Hertz) of output `Spectrogram`,
            or, if ``logf=True``, the number of frequency samples;
            give `None` to skip this step and return the original resolution,
            default is 0.5 Hz or 500 frequency samples.

        logf : `bool`, optional
            If `True`, use logarithmically sampled frequencies in the output
            (using ``fres`` as the number of frequency samples),
            otherwise use linearly sampled frequencies of the specified resolution.

        norm : `bool`, `str`, optional
            If `True` normalise the returned Q-transform output by the median;
            if `False` do not normalize; if a string, specify
            how to normalize the output, e.g. ``'mean'`` or ``'median'``.

        mismatch : `float`
            Maximum allowed fractional mismatch between neighbouring tiles.

        outseg : `~gwpy.segments.Segment`, optional
            GPS `[start, stop)` segment for output `Spectrogram`,
            default is the full duration of the input.

        whiten : `bool`, `~gwpy.frequencyseries.FrequencySeries`, optional
            If `True`, whiten the data before computing the Q-transform;
            if `False`, do not whiten the data; if a `FrequencySeries`,
            use that as the amplitude spectral density (ASD) to whiten the data.

        fduration : `float`, optional
            Duration (in seconds) of the time-domain FIR whitening filter,
            only used if `whiten` is not `False`, defaults to 2 seconds.

        highpass : `float`, optional
            Highpass corner frequency (in Hz) of the FIR whitening filter,
            used only if `whiten` is not `False`, default: `None`.

        asd_kw
            Keyword arguments to pass to `TimeSeries.asd` to generate
            an ASD to use when whitening the data.

        Returns
        -------
        out : `~gwpy.spectrogram.Spectrogram`
            Output `Spectrogram` of normalised Q energy.

        See Also
        --------
        TimeSeries.asd
            For documentation on acceptable ``**asd_kw``.

        TimeSeries.whiten
            For documentation on how the whitening is done.

        gwpy.signal.qtransform
            For code and documentation on how the Q-transform is implemented.

        Notes
        -----
        This method will return a `Spectrogram` of dtype ``float32`` if
        ``norm`` is given, and ``float64`` otherwise.

        To optimize plot rendering with `~matplotlib.axes.Axes.pcolormesh`,
        the output `~gwpy.spectrogram.Spectrogram` can be given a log-sampled
        frequency axis by passing `logf=True` at runtime. The `fres` argument
        is then the number of points on the frequency axis. Note, this is
        incompatible with `~matplotlib.axes.Axes.imshow`.

        It is also highly recommended to use the `outseg` keyword argument
        when only a small window around a given GPS time is of interest. This
        will speed up this method a little, but can greatly speed up
        rendering the resulting `Spectrogram` using `pcolormesh`.

        If you aren't going to use `pcolormesh` in the end, don't worry.

        Examples
        --------
        >>> from numpy.random import normal
        >>> from scipy.signal import gausspulse
        >>> from gwpy.timeseries import TimeSeries

        Generate a `TimeSeries` containing Gaussian noise sampled at 4096 Hz,
        centred on GPS time 0, with a sine-Gaussian pulse ('glitch') at
        500 Hz:

        >>> noise = TimeSeries(
        ...     normal(loc=1, size=4096*4),
        ...     sample_rate=4096,
        ...     epoch=-2,
        ... )
        >>> glitch = TimeSeries(
        ...     gausspulse(noise.times.value, fc=500) * 4,
        ...     sample_rate=4096,
        ... )
        >>> data = noise + glitch

        Compute and plot the Q-transform of these data:

        >>> q = data.q_transform()
        >>> plot = q.plot()
        >>> ax = plot.gca()
        >>> ax.set_xlim(-.2, .2)
        >>> ax.set_epoch(0)
        >>> plot.show()
        """
        from ..frequencyseries import FrequencySeries

        # condition data
        if whiten is True:  # generate ASD dynamically
            window = asd_kw.pop("window", "hann")
            fftlength = asd_kw.pop("fftlength", _fft_length_default(self.dt))
            overlap = asd_kw.pop("overlap", None)
            if overlap is None and fftlength == self.duration.value:
                asd_kw["method"] = DEFAULT_FFT_METHOD
                overlap = 0
            elif overlap is None:
                overlap = recommended_overlap(window) * fftlength
            whiten = self.asd(
                fftlength,
                overlap,
                window=window,
                **asd_kw,
            )

        if isinstance(whiten, FrequencySeries):
            # apply whitening (with error on division by zero)
            with numpy.errstate(all="raise"):
                data = self.whiten(
                    asd=whiten,
                    fduration=fduration,
                    highpass=highpass,
                )
        else:
            data = self

        # determine search window
        search_seg: Segment | None = None
        if gps is not None and search is not None:
            search_seg = Segment(gps-search/2, gps+search/2) & self.span
        qgram, _ = qtransform.q_scan(
            data,
            frange=frange,
            qrange=qrange,
            norm=norm,
            mismatch=mismatch,
            search=search_seg,
        )
        return qgram.interpolate(
            tres=tres,
            fres=fres,
            logf=logf,
            outseg=outseg,
        )


@as_series_dict_class(TimeSeries)
class TimeSeriesDict(TimeSeriesBaseDict):
    """A dictionary of `TimeSeries` objects."""

    __doc__ = TimeSeriesBaseDict.__doc__.replace("TimeSeriesBase", "TimeSeries")  # type: ignore[union-attr]
    EntryClass = TimeSeries

    read = UnifiedReadWriteMethod(TimeSeriesDictRead)
    write = UnifiedReadWriteMethod(TimeSeriesDictWrite)
    get = UnifiedReadWriteMethod(TimeSeriesDictGet)


class TimeSeriesList(TimeSeriesBaseList):
    """A list of `TimeSeries` objects."""

    __doc__ = TimeSeriesBaseList.__doc__.replace("TimeSeriesBase", "TimeSeries")  # type: ignore[union-attr]
    EntryClass = TimeSeries
