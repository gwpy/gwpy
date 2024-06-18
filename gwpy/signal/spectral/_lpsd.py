"""
Wrapper for LPSD method, see https://gitlab.com/uhh-gwd/lpsd

| Martin Hewitson <martin.hewitson@aeiDOTmpg.de>
(original LPSD code)
| Christian Darsow-Fromm <cdarsowf[at]physnet.uni-hamburg.de>
(python wrapper for LPSD)
| Artem Basalaev <artem[dot]basalaev[at]physik.uni-hamburg.de>
(this implementation)
"""

import warnings
import inspect
import numpy

from ...frequencyseries import FrequencySeries
from ...window import get_window
from . import _registry as fft_registry
from ._utils import scale_timeseries_unit


def lpsd(*args, **kwargs):
    """Calculate the CSD of a `TimeSeries` using LPSD method.

    Parameters
    ----------
    args : list
        timeseries: TimeSeries
            input time series (if only one specified, calculates PSD)
        other: TimeSeries, optional
            second input time series
        nfft: int
            number of samples per FFT, user-specified value ignored;
            calculated by the algorithm instead

    kwargs : dict
        fftlength : float, optional
            number of seconds in single FFT. User-specified value ignored,
            algorithm calculates optimal segment lengths.
        overlap : float, optional
            number of seconds of overlap between FFTs.
        window : str, optional
            Window function to apply to timeseries prior to FFT.
            Possible values: 'hann', 'hanning', 'ham',
            'hamming', 'bartlett', 'blackman', 'kaiser'.
            Defaults to 'kaiser'.
        additional arguments are passed to :class:lpsd.lcsd

    Returns
    -------
    fs: FrequencySeries
        resulting CSD
    """

    import pandas
    from lpsd import lcsd
    from lpsd._lcsd import LCSD

    try:
        timeseries, nfft = args
        other = timeseries
    except ValueError:
        timeseries, other, nfft = args

    if len(timeseries.value) != len(other.value):
        raise ValueError("Time series must have "
                         "the same length (number of samples)!")

    # No fftlength for LPSD method.
    if kwargs.get("fftlength") or nfft != len(timeseries.value):
        raise ValueError(
            "fftlength/nfft arguments are "
            "not supported by LPSD averaging method; "
            "segment lengths are calculated by the algorithm."
        )

    # Convert inputs to pandas.DataFrame
    df = pandas.DataFrame()
    df["x1"] = timeseries.value
    df["x2"] = other.value
    df.index = timeseries.times.value

    # clean up kwargs: get ones that are allowed for LPSD
    lpsd_kwargs, overlap, window_function = \
        _parse_kwargs(timeseries.duration, kwargs)

    csd = lcsd(
        df,
        overlap=overlap,
        window_function=window_function,
        **lpsd_kwargs
    )

    # generate FrequencySeries and return
    unit = scale_timeseries_unit(
        timeseries.unit,
        kwargs.pop("scaling", "density"),
    )
    return FrequencySeries(
        csd["psd"].values,
        unit=unit,
        frequencies=csd.index.values,
        name=timeseries.name,
        epoch=timeseries.epoch,
        channel=timeseries.channel,
    )


def _parse_window(window):
    """Parse `window` argument for LPSD algorithm.

    Parameters
    ----------
    window : str
        Window function as string
        (specifying np.array not supported, defaults to `kaiser`)

    Returns
    -------
    window: func
        Window as a numpy function
    """
    window = "kaiser" if window is None else window

    if not isinstance(window, str):
        warnings.warn(
            "Specifying window as an array "
            "for LPSD averaging method is not supported,"
            " defaulting to 'kaiser' window"
        )
        window = "kaiser"

    try:
        window_function = get_window(window)
    except KeyError as exc:
        raise KeyError(
            "Window " + window + "is not supported for LPSD averaging method"
        ) from exc

    return window_function


def _parse_kwargs(total_duration, kwargs):
    """Parse arguments for LPSD algorithm.
    Convert overlap s -> %, window to numpy function,
    and remove kwargs not accepted by lpsd code.

    Parameters
    ----------
    total_duration : float
        Total duration of time series in seconds
    kwargs : dict
        Rest of the arguments

    Returns
    -------
    lpsd_kwargs: dict
        Cleaned-up arguments
    overlap: float
        Overlap in percentage
    window: func
        Window as a numpy function

    """
    # convert overlap given in number of seconds to percentage
    overlap = _parse_overlap(kwargs.pop("overlap", 0), total_duration)

    # convert window to numpy function
    window_function = _parse_window(kwargs.pop("window_", None))
    # clean up default value from kwargs
    if "window" in kwargs:
        kwargs.pop("window")

    allowed_kwargs = inspect.getfullargspec(LCSD.__init__).args
    lpsd_kwargs = kwargs.copy()
    for k in kwargs:
        if k not in allowed_kwargs:
            lpsd_kwargs.pop(k)

    return lpsd_kwargs, overlap, window_function


def _parse_overlap(overlap, total_duration):
    """Parse `overlap` argument for LPSD algorithm.
    Convert value in seconds to percentage

    Parameters
    ----------
    total_duration : float
        Total duration of time series in seconds

    Returns
    -------
    overlap: float
        Overlap in percentage
    """
    if overlap == 0:
        return 0
    if overlap > 0:
        if overlap > total_duration:
            raise ValueError(
                "Specified overlap (in seconds) "
                "exceeds total time series duration!"
            )
        return overlap / total_duration


fft_registry.register_method(lpsd)


def lpsd_coherence(timeseries, other, fftlength=None,
                   overlap=None, window='kaiser', **kwargs):
    """Calculate the frequency-coherence between this `TimeSeries`
    and another using LPSD method.

    Parameters
    ----------
    timeseries : `TimeSeries`
        `TimeSeries` signal

    other : `TimeSeries`
        `TimeSeries` signal to calculate coherence with

    fftlength : `float`, optional
        number of seconds in single FFT. User-specified value ignored,
        algorithm calculates optimal segment lengths.

    overlap : `float`, optional
        number of seconds of overlap between FFTs.

    window : `str`, `numpy.ndarray`, optional
        Window function to apply to timeseries prior to FFT.
        See :func:`scipy.signal.get_window`
        Defaults to 'kaiser'.

    **kwargs
        any other keyword arguments accepted by :class:lpsd.lcsd

    Returns
    -------
    coherence : `~gwpy.frequencyseries.FrequencySeries`
        the coherence `FrequencySeries` of this `TimeSeries`
        with the other
    """

    csd = spectral.psd(
        (timeseries, other),
        method_func=spectral.lpsd,
        fftlength=fftlength,
        overlap=overlap,
        window=window,
        **kwargs,
    )
    psd1 = spectral.psd(
        timeseries,
        method_func=spectral.lpsd,
        fftlength=fftlength,
        overlap=overlap,
        window=window,
        **kwargs,
    )
    psd2 = spectral.psd(
        other,
        method_func=spectral.lpsd,
        fftlength=fftlength,
        overlap=overlap,
        window=window,
        **kwargs,
    )
    coherence = numpy.abs(csd) ** 2 / psd1 / psd2
    coherence.name = f"Coherence between {self.name} and {other.name}"
    coherence.override_unit("coherence")
    return coherence


fft_registry.register_method(lpsd_coherence)
