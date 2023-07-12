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
import pandas
import numpy

from lpsd import lcsd
from lpsd._lcsd import LCSD

from ...frequencyseries import FrequencySeries
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
    overlap, window_function = _parse_kwargs(timeseries.duration, kwargs)
    allowed_kwargs = inspect.getfullargspec(LCSD.__init__).args
    lpsd_kwargs = kwargs.copy()
    for k in kwargs:
        if k not in allowed_kwargs:
            lpsd_kwargs.pop(k)
    csd = lcsd(df, overlap=overlap, window_function=window_function, **lpsd_kwargs)

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


def _parse_kwargs(total_duration, kwargs):

    # convert overlap given in number of seconds to percentage
    overlap = kwargs.pop("overlap", 0)
    if overlap > 0:
        if overlap > total_duration:
            raise ValueError(
                "Specified overlap (in seconds) "
                "exceeds total time series duration!"
            )
        overlap = overlap / total_duration

    # convert window to numpy function
    window = kwargs.pop("window_", None)
    window = "kaiser" if window is None else window
    # clean up default value from kwargs
    if "window" in kwargs:
        kwargs.pop("window")
    if not isinstance(window, str):
        warnings.warn(
            "Specifying window as an array "
            "for LPSD averaging method is not supported,"
            " defaulting to 'kaiser' window"
        )
        window = "kaiser"

    window_to_func = {
        "kaiser": numpy.kaiser,
        "hann": numpy.hanning,
        "hanning": numpy.hanning,
        "hamm": numpy.hamming,
        "hamming": numpy.hamming,
        "bartlett": numpy.bartlett,
        "blackman": numpy.blackman,
    }
    try:
        window_function = window_to_func[window]
    except KeyError as exc:
        raise KeyError(
            "Window " + window + "is not supported for LPSD averaging method"
        ) from exc

    return overlap, window_function

fft_registry.register_method(lpsd)
