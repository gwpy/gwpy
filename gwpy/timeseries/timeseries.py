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
from math import (ceil, pi)
from multiprocessing import (Process, Queue as ProcessQueue)

import numpy
from numpy import fft as npfft
from scipy import signal

from astropy import units

from ..io import (reader, writer)
from ..segments import Segment
from ..utils import with_import
from ..utils.docstring import interpolate_docstring
from ..utils.compat import OrderedDict
from .core import (TimeSeriesBase, TimeSeriesBaseDict, TimeSeriesBaseList,
                   as_series_dict_class)
from .filter import create_notch

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@interpolate_docstring
class TimeSeries(TimeSeriesBase):
    """A time-domain data array

    Parameters
    ----------
    %(Array1)s

    %(time-axis)s

    %(Array2)s

    Examples
    --------
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

    .. rubric:: Key Methods

    .. autosummary::

        ~TimeSeries.fetch
        ~TimeSeries.read
        ~TimeSeries.write
        ~TimeSeries.plot

    """
    read = classmethod(interpolate_docstring(reader(
        doc="""Read data into a `TimeSeries`

        Parameters
        ----------
        %(timeseries-read1)s

        %(timeseries-read2)s

        Notes
        -----""")))

    write = writer(
        doc="""Write this `TimeSeries` to a file

        Parameters
        ----------
        outfile : `str`
            path of output file

        Notes
        -----
        """)

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
        out : :class:`~gwpy.frequencyseries.FrequencySeries`
            the normalised, complex-valued FFT `FrequencySeries`.

        See Also
        --------
        :mod:`scipy.fftpack` for the definition of the DFT and conventions
        used.

        Notes
        -----
        This method, in constrast to the :meth:`numpy.fft.rfft` method
        it calls, applies the necessary normalisation such that the
        amplitude of the output :class:`~gwpy.frequencyseries.FrequencySeries` is
        correct.
        """
        from ..frequencyseries import FrequencySeries
        if nfft is None:
            nfft = self.size
        dft = npfft.rfft(self.value, n=nfft) / nfft
        dft[1:] *= 2.0
        new = FrequencySeries(dft, epoch=self.epoch, channel=self.channel,
                              unit=self.unit)
        try:
            new.frequencies = npfft.rfftfreq(nfft, d=self.dx.value)
        except AttributeError:
            new.frequencies = numpy.arange(0, new.size) / (nfft * self.dx.value)
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
        out : complex-valued :class:`~gwpy.frequencyseries.FrequencySeries`
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

    def psd(self, fftlength=None, overlap=None, method='welch', **kwargs):
        """Calculate the PSD `FrequencySeries` for this `TimeSeries`.

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
        plan : :lal:`REAL8FFTPlan`, optional
            LAL FFT plan to use when generating average spectrum,
            substitute type 'REAL8' as appropriate.

        Returns
        -------
        psd :  :class:`~gwpy.frequencyseries.FrequencySeries`
            a data series containing the PSD.

        Notes
        -----
        The available methods are:

        """
        from ..frequencyseries.registry import get_method
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
        return psd_

    def asd(self, fftlength=None, overlap=None, method='welch', **kwargs):
        """Calculate the ASD `FrequencySeries` of this `TimeSeries`.

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
        plan : :lal:`REAL8FFTPlan`, optional
            LAL FFT plan to use when generating average spectrum,
            substitute type 'REAL8' as appropriate.

        Returns
        -------
        psd :  :class:`~gwpy.frequencyseries.FrequencySeries`
            a data series containing the PSD.

        Notes
        -----
        The available methods are:

        """
        asd_ = self.psd(method=method, fftlength=fftlength,
                        overlap=overlap, **kwargs) ** (1/2.)
        return asd_

    def csd(self, other, fftlength=None, overlap=None, **kwargs):
        """Calculate the CSD `FrequencySeries` for two `TimeSeries`

        Parameters
        ----------
        fftlength : `float`, default: :attr:`TimeSeries.duration`
            number of seconds in single FFT
        overlap : `float`, optional, default: `None`
            number of seconds of overlap between FFTs, defaults to that of
            the relevant method.
        window : `timeseries.Window`, optional
            window function to apply to timeseries prior to FFT
        plan : :lal:`REAL8FFTPlan`, optional
            LAL FFT plan to use when generating average spectrum,
            substitute type 'REAL8' as appropriate.

        Returns
        -------
        csd :  :class:`~gwpy.frequencyseries.FrequencySeries`
            a data series containing the CSD.
        """

        from ..frequencyseries.registry import get_method
        # get method
        scaling = kwargs.get('scaling', 'density')
        method_func = get_method('csd', scaling=scaling)
        # type-cast arguments
        if fftlength is None:
            fftlength = self.duration
        nfft = int((fftlength * self.sample_rate).decompose().value)
        if overlap is not None:
            kwargs['noverlap'] = int(
                (overlap * self.sample_rate).decompose().value)
        # calculate and return spectrum
        csd_ = method_func(self, other, nfft, **kwargs)
        return csd_

    def spectrogram(self, stride, fftlength=None, overlap=0,
                    method='welch', window=None, nproc=1, **kwargs):
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
        timeseries : `TimeSeries`
            input time-series to process.

        stride : `float`
            number of seconds in single PSD (column of spectrogram).

        fftlength : `float`
            number of seconds in single FFT.

        overlap : `int`, optional, default: 0
            number of seconds between FFTs.

        method : `str`, optional, default: 'welch'
            average spectrum method.

        window : `str`, `numpy.ndarray`, optional, default: `None`
            window function to apply to timeseries prior to FFT,
            see `scipy.signal.get_window` for details on acceptable
            formats

        plan : :lal:`REAL8FFTPlan`, optional
            LAL FFT plan to use when generating average spectrum,
            substitute type 'REAL8' as appropriate. This is only accepted
            if you select `method` as one of 'median-mean', 'median', or
            'lal-welch'

        nproc : `int`, default: ``1``
            number of CPUs to use in parallel processing of FFTs

        cross : `TimeSeries`
            optional keyword argument
            time-series for calculating CSD spectrogram
            if None, then calculates PSD spectrogram

        Returns
        -------
        spectrogram : `~gwpy.spectrogram.Spectrogram`
            time-frequency power spectrogram as generated from the
            input time-series.
        """
        from ..frequencyseries.utils import (
            safe_import, scale_timeseries_units)
        from ..frequencyseries.registry import get_method
        from ..spectrogram import (Spectrogram, SpectrogramList)

        # format FFT parameters
        if fftlength is None:
            fftlength = stride
        if overlap is None:
            overlap = 0
        fftlength = units.Quantity(fftlength, 's').value
        overlap = units.Quantity(overlap, 's').value
        cross = kwargs.pop('cross', None)

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
        noverlap = int((overlap * self.sample_rate).decompose().value)
        nsteps = int(self.size // nsamp)
        nproc = min(nsteps, nproc)
        noverlap2 = int(noverlap // 2.)

        # generate window and plan if needed
        method_func = get_method(method)
        if method_func.__module__.endswith('lal_') and cross is None:
            safe_import('lal', method)
            from ..frequencyseries.lal_ import (generate_lal_fft_plan,
                                                generate_lal_window)
            if kwargs.get('window', None) is None:
                kwargs['window'] = generate_lal_window(nfft, dtype=self.dtype)
            if kwargs.get('plan', None) is None:
                kwargs['plan'] = generate_lal_fft_plan(nfft, dtype=self.dtype)
        else:
            if window is None:
                window = 'hanning'
            if isinstance(window, str) or type(window) is tuple:
                window = signal.get_window(window, nfft)
            kwargs['window'] = window

        # set up single process Spectrogram generation
        def _from_timeseries(ts, cts, epoch=None):
            """Generate a `Spectrogram` from a `TimeSeries`.
            """
            # calculate specgram parameters
            dt = stride
            df = 1 / fftlength

            # get size of spectrogram
            nsteps_ = int(ts.size // nsamp)
            nfreqs = int(fftlength * ts.sample_rate.value // 2 + 1)

            # generate output spectrogram
            unit = scale_timeseries_units(
                ts.unit, kwargs.get('scaling', 'density'))
            dtype = numpy.float64 if cts is None else complex
            if epoch is None:
                epoch = ts.epoch
            out = Spectrogram(numpy.zeros((nsteps_, nfreqs)), dtype=dtype,
                              unit=unit, channel=ts.channel, epoch=epoch,
                              f0=0, df=df, dt=dt, copy=False)

            if not nsteps_:
                return out

            # stride through TimeSeries, calculating PSDs or CSDs
            if cts is not None and method not in (None, 'welch'):
                warn("Cannot calculate cross spectral density using "
                     "the %r method. Using 'welch' instead..." % method)
            for step in range(nsteps_):
                # find step TimeSeries with overlap
                idx = max(0, nsamp * step - noverlap2)
                idx_end = min(ts.size, idx + nsamp + noverlap)
                stepseries = ts[idx:idx_end]
                if cts is None:
                    stepsd = stepseries.psd(fftlength=fftlength,
                                            overlap=overlap,
                                            method=method, **kwargs)
                else:
                    otherstepseries = cts[idx:idx_end]
                    stepsd = stepseries.csd(otherstepseries,
                                            fftlength=fftlength,
                                            overlap=overlap, **kwargs)
                out.value[step, :] = stepsd.value
            return out

        # single-process return
        if nsteps == 0 or nproc == 1:
            return _from_timeseries(self, cross)

        # wrap spectrogram generator
        def _specgram(q, *args, **kwargs):
            try:
                q.put(_from_timeseries(*args, **kwargs))
            except Exception as e:
                q.put(e)

        # otherwise build process list
        stepperproc = int(ceil(nsteps / nproc))
        nsampperproc = stepperproc * nsamp
        queue = ProcessQueue(nproc)
        processlist = []
        for i in range(nproc):
            # index of un-overlapped time-series and epoch
            a = i * nsampperproc
            t = self.x0 + self.dx * a
            # overlapped indices for FFT
            ao = max(0, a - noverlap2)
            bo = min(self.size, a + nsampperproc + noverlap)
            tsamp = self[ao:bo]
            if cross is None:
                csamp = None
            else:
                csamp = cross[ao:bo]
            # process this chunk
            process = Process(target=_specgram, args=(queue, tsamp, csamp),
                              kwargs={'epoch': t})
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
        from ..frequencyseries import scale_timeseries_units
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
        nsteps = 1 + int((self.size - nstride) / nstride)  # number of columns
        nfreqs = int(nfft / 2 + 1)  # number of rows
        unit = scale_timeseries_units(self.unit, scaling)
        dt = nstride * self.dt
        tmp = numpy.zeros((nsteps, nfreqs), dtype=self.dtype)
        out = Spectrogram(numpy.zeros((nsteps, nfreqs), dtype=self.dtype),
                          epoch=self.epoch, channel=self.channel,
                          name=self.name, unit=unit, dt=dt, f0=0,
                          df=1/fftlength)

        # get window
        if window is None:
            window = 'boxcar'
        if isinstance(window, (str, tuple)):
            window = signal.get_window(window, nfft)

        # calculate overlapping periodograms
        for i in xrange(nsteps):
            idx = i * nstride
            # don't proceed past end of data, causes artefacts
            if idx+nfft > self.size:
                break
            ts = self.value[idx:idx+nfft]
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
            out.value[i, :] = numpy.average(tmp[x0:x1], axis=0, weights=w)

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
        """Calculate the Rayleigh `FrequencySeries` for this `TimeSeries`.

        Parameters
        ----------
        fftlength : `float`, default: :attr:`TimeSeries.duration`
            number of seconds in single FFT
        overlap : `float`, optional, default: `None`
            number of seconds of overlap between FFTs, defaults to that of
            the relevant method.
        window : `timeseries.Window`, optional
            window function to apply to timeseries prior to FFT
        plan : :lal:`REAL8FFTPlan`, optional
            LAL FFT plan to use when generating average spectrum,
            substitute type 'REAL8' as appropriate.

        Returns
        -------
        psd :  :class:`~gwpy.frequencyseries.FrequencySeries`
            a data series containing the PSD.

        Notes
        -----
        The available methods are:

        """
        from ..frequencyseries.registry import get_method
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
        plan : :lal:`REAL8FFTPlan`, optional
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
        rspecgram.override_unit('')
        return rspecgram

    def csd_spectrogram(self, other, stride, fftlength=None, overlap=0,
                        window=None, nproc=1, **kwargs):
        """Calculate the cross spectral density spectrogram of this
           `TimeSeries` with 'other'.

        Parameters
        ----------
        timeseries : :class:`~gwpy.timeseries.core.TimeSeries`
            input time-series to process.
        other : :class:`~gwpy.timeseries.core.TimeSeries`
            second time-series for cross spectral density calculation
        stride : `float`
            number of seconds in single PSD (column of spectrogram).
        fftlength : `float`
            number of seconds in single FFT.
        overlap : `int`, optiona, default: fftlength
            number of seconds between FFTs.
        window : `timeseries.window.Window`, optional, default: `None`
            window function to apply to timeseries prior to FFT.
        plan : :lal:`REAL8FFTPlan`, optional
            LAL FFT plan to use when generating average spectrum,
            substitute type 'REAL8' as appropriate.
        nproc : `int`, default: ``1``
            maximum number of independent frame reading processes, default
            is set to single-process file reading.

        Returns
        -------
        spectrogram : :class:`~gwpy.spectrogram.core.Spectrogram`
            time-frequency cross spectrogram as generated from the
            two input time-series.
        """
        cspecgram = self.spectrogram(stride, fftlength=fftlength,
                                     overlap=overlap, window=window,
                                     cross=other, nproc=nproc, **kwargs)
        return cspecgram

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
            new = signal.decimate(self.value, factor, numtaps-1,
                                  ftype='fir').view(self.__class__)
        # otherwise use Fourier filtering
        else:
            nsamp = int(self.shape[0] * self.dx.value * rate)
            new = signal.resample(self.value, nsamp,
                                  window=window).view(self.__class__)
        new.__dict__ = self.copy_metadata()
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
        new.__dict__ = self.copy_metadata()
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
        coherence : :class:`~gwpy.frequencyseries.FrequencySeries`
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
            kwargs['window'] = window
        coh, f = mlab.cohere(self_.value, other.value, NFFT=fftlength,
                             Fs=sampling, noverlap=overlap, **kwargs)
        out = coh.view(FrequencySeries)
        out.xindex = f
        out.epoch = self.epoch
        out.name = 'Coherence between %s and %s' % (self.name, other.name)
        out.unit = 'coherence'
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
        coherence : :class:`~gwpy.frequencyseries.FrequencySeries`
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
            rms_ = numpy.sqrt(numpy.mean(numpy.abs(stepseries.value)**2))
            data[step] = rms_
        name = '%s %.2f-second RMS' % (self.name, stride)
        return self.__class__(data, channel=self.channel, epoch=self.epoch,
                              name=name, sample_rate=(1/float(stride)))

    def whiten(self, fftlength, overlap=0, method='welch', window='hanning',
               detrend='constant', asd=None, **kwargs):
        """White this `TimeSeries` against its own ASD

        Parameters
        ----------
        fftlength : `float`
            number of seconds in single FFT

        overlap : `float`, optional, default: 0
            numbers of seconds by which to overlap neighbouring FFTs,
            by default, no overlap is used.

        method : `str`, optional, default: `welch`
            average spectrum method

        window : `str`, :class:`numpy.ndarray`
            name of the window function to use, or an array of length
            ``fftlength * TimeSeries.sample_rate`` to use as the window.

        asd : `~gwpy.frequencyseries.FrequencySeries`
            the amplitude-spectral density using which to whiten the data

        **kwargs
            other keyword arguments are passed to the `TimeSeries.asd`
            method to estimate the amplitude spectral density
            `FrequencySeries` of this `TimeSeries.

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
        """
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
        out = type(self)(numpy.zeros(nsteps * nstride + noverlap))
        out.__dict__ = self.copy_metadata()
        del out.times
        # loop over ffts and whiten each one
        for i in range(nsteps):
             i0 = i * nstride
             i1 = i0 + nfft
             in_ = self[i0:i1].detrend(detrend) * window
             out.value[i0:i1] += npfft.irfft(in_.fft().value * invasd)
        return out

    def detrend(self, detrend='constant'):
        """Remove the trend from this `TimeSeries`

        This method just wraps :meth:`scipy.signal.detrend` to return
        an object of the same type as the input.

        Parameters
        ----------
        detrend : `str`, optional, default: `constant`
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
        data.__dict__ = self.copy_metadata()
        return data

    def plot(self, **kwargs):
        """Plot the data for this TimeSeries.
        """
        from ..plotter import TimeSeriesPlot
        return TimeSeriesPlot(self, **kwargs)

    def notch(self, frequency, type='iir', **kwargs):
        """Notch out a frequency in a `TimeSeries`

        Parameters
        ----------
        frequency : `float`, `~astropy.units.Quantity`
            frequency (default in Hertz) at which to apply the notch
        type : `str`, optional, default: 'iir'
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
        zpk = create_notch(frequency, self.sample_rate.value,
                           type=type, **kwargs)
        return self.filter(*zpk)

    def q_transform(self, qrange=(4, 64), frange=(0, numpy.inf),
                    gps=None, search=.5, tres=.001, fres=.5, outseg=None,
                    **psdkwargs):
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

        outseg : `~gwpy.segments.Segment`, optional
            GPS `[start, stop)` segment for output `Spectrogram`

        **psdkwargs
            keyword arguments to pass to `TimeSeries.psd` when whitening
            the input data

        Returns
        -------
        specgram : `~gwpy.spectrogram.Spectrogram`
            output `Spectrogram` of normalised Q energy

        Notes
        -----
        It is highly recommended to use the `outseg` keyword argument when
        only a small window around a given GPS time is of interest. This
        will speed up this method a little, but can greatly speed up
        rendering the resulting `~gwpy.spectrogram.Spectrogram` using
        `~matplotlib.axes.Axes.pcolormesh`.

        If you aren't going to use `pcolormesh` in the end, don't worry.

        See Also
        --------
        TimeSeries.psd
            for documentation on acceptable `**psdkwargs`
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
        """
        from scipy.interpolate import (interp2d, InterpolatedUnivariateSpline)
        from ..spectrogram import Spectrogram
        from ..signal.qtransform import QTiling

        if outseg is None:
            outseg = self.span

        # generate tiling
        planes = QTiling(abs(self.span), self.sample_rate.value,
                         qrange=qrange, frange=frange)

        # condition data
        psdkw = {
            'method': 'median-mean',
            'fftlength': 2,
            'overlap': 1,
        }
        psdkw.update(psdkwargs)
        fftlength = psdkw.pop('fftlength')
        overlap = psdkw.pop('overlap')
        asd = self.asd(fftlength, overlap, **psdkw)
        wdata = self.whiten(fftlength, overlap, asd=asd)
        fdata = wdata.fft().value

        # set up results
        peakq = None
        peakenergy = 0

        # Q-transform data for each `(Q, frequency)` tile
        for plane in planes:
            f, normenergies = plane.transform(fdata, normalized=True,
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
                    frequencies = f

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
        # XXX: this is done because duncan doesn't like interpolated images,
        #      because they don't support log scaling
        if fres is None:  # unless user tells us not to
            return out
        else:
            interp = interp2d(out.times.value, frequencies, out.value.T,
                              kind='cubic')
            f2 = numpy.arange(planes.frange[0], planes.frange[1], fres)
            new = Spectrogram(interp(out.times.value, f2 + fres/2.).T,
                              x0=outseg[0], dx=tres,
                              f0=planes.frange[0], df=fres)
            new.q = peakq
            return new


@as_series_dict_class(TimeSeries)
class TimeSeriesDict(TimeSeriesBaseDict):
    __doc__ = TimeSeriesBaseDict.__doc__.replace('TimeSeriesBase',
                                                 'TimeSeries')
    EntryClass = TimeSeries
    read = classmethod(reader(doc=TimeSeriesBaseDict.read.__doc__))


class TimeSeriesList(TimeSeriesBaseList):
    __doc__ = TimeSeriesBaseDict.__doc__.replace('TimeSeriesBase',
                                                 'TimeSeries')
    EntryClass = TimeSeries
