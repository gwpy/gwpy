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

"""Read/write WAV files using `scipy.signal.wavfile`."""

from __future__ import annotations

import struct
import wave
from typing import TYPE_CHECKING

import numpy
from scipy.io import wavfile

from .. import TimeSeries

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

    from ...time import SupportsToGps

WAV_SIGNATURE = ("RIFF", "WAVE")


def read(
    fobj: str | Path | IO,
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
    **kwargs,
) -> TimeSeries:
    """Read a WAV file into a `TimeSeries`.

    Parameters
    ----------
    fobj : `file`, `str`
        Open file-like object or filename to read from.

    start : `float`, `astropy.units.Quantity`, optional
        The desired start point of the X-axis, defaults to
        the start point of the incoming series.

    end : `float`, `astropy.units.Quantity`, optional
        The desired end point of the X-axis, defaults to
        the end point of the incoming series.

    **kwargs
        All keyword arguments are passed onto :func:`scipy.io.wavfile.read`.

    See Also
    --------
    scipy.io.wavfile.read
        For details on how the WAV file is actually read.

    Examples
    --------
    >>> from gwpy.timeseries import TimeSeries
    >>> t = TimeSeries.read('test.wav')
    """
    fsamp, arr = wavfile.read(fobj, **kwargs)
    return TimeSeries(arr, sample_rate=fsamp).crop(
        start=start,
        end=end,
    )


def write(
    series: TimeSeries,
    output: str | IO,
    scale: float | None = None,
) -> None:
    """Write a `TimeSeries` to a WAV file.

    Parameters
    ----------
    series : `TimeSeries`
        the series to write

    output : `file`, `str`
        the file object or filename to write to

    scale : `float`, optional
        the factor to apply to scale the data to (-1.0, 1.0),
        pass `scale=1` to not apply any scale, otherwise
        the data will be auto-scaled

    See Also
    --------
    scipy.io.wavfile.write
        for details on how the WAV file is actually written

    Examples
    --------
    >>> from gwpy.timeseries import TimeSeries
    >>> t = TimeSeries([1, 2, 3, 4, 5])
    >>> t = TimeSeries.write('test.wav')
    """
    fsamp = int(series.sample_rate.decompose().value)
    if scale is None:
        scale = 1 / numpy.abs(series.value).max()
    data = (series.value * scale).astype("float32")
    return wavfile.write(output, fsamp, data)


# -- identify ------------------------

def _is_wav(
    origin: str,
    filepath: str,
    fileobj: IO,
    *args: IO,
    **kwargs,
) -> bool:
    """Identify a file as WAV.

    See `astropy.io.registry` for details on how this function is used.
    """
    # if given an open, readable file object, inspect the magic signature
    if origin == "read" and fileobj is not None:
        loc = fileobj.tell()
        fileobj.seek(0)
        try:
            riff, _, fmt = struct.unpack("<4sI4s", fileobj.read(12))
            if isinstance(riff, bytes):
                riff = riff.decode("utf-8")
                fmt = fmt.decode("utf-8")
            return riff == WAV_SIGNATURE[0] and fmt == WAV_SIGNATURE[1]
        except (
            UnicodeDecodeError,
            struct.error,
        ):
            return False
        finally:
            fileobj.seek(loc)

    # check the file extension
    if filepath is not None:
        return filepath.endswith((".wav", ".wave"))

    # attempt to read the file as .wav
    kwargs.pop("start", None)
    kwargs.pop("end", None)
    try:
        with wave.open(*args, **kwargs):
            return True
    except (
        AttributeError,  # not a file-like object
        TypeError,  # wave.open failed for some other reason
        wave.Error,  # scipy failed to read this as a WAV file
    ):
        return False


# -- register ------------------------

registry = TimeSeries.read.registry
registry.register_reader("wav", TimeSeries, read)
registry.register_writer("wav", TimeSeries, write)
registry.register_identifier("wav", TimeSeries, _is_wav)
