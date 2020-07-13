# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Read/write WAV files using `scipy.signal.wavfile`
"""

import struct
import wave

import numpy

from scipy.io import wavfile

from .. import TimeSeries
from ...io import registry as io_registry

WAV_SIGNATURE = ('RIFF', 'WAVE')


def read(fobj, **kwargs):
    """Read a WAV file into a `TimeSeries`

    Parameters
    ----------
    fobj : `file`, `str`
        open file-like object or filename to read from

    **kwargs
        all keyword arguments are passed onto :func:`scipy.io.wavfile.read`

    See also
    --------
    scipy.io.wavfile.read
        for details on how the WAV file is actually read

    Examples
    --------
    >>> from gwpy.timeseries import TimeSeries
    >>> t = TimeSeries.read('test.wav')
    """
    fsamp, arr = wavfile.read(fobj, **kwargs)
    return TimeSeries(arr, sample_rate=fsamp)


def write(series, output, scale=None):
    """Write a `TimeSeries` to a WAV file

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

    See also
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
    data = (series.value * scale).astype('float32')
    return wavfile.write(output, fsamp, data)


def is_wav(origin, filepath, fileobj, *args, **kwargs):
    """Identify a file as WAV

    See `astropy.io.registry` for details on how this function is used.
    """
    # pylint: disable=unused-argument
    if origin == 'read' and fileobj is not None:
        loc = fileobj.tell()
        fileobj.seek(0)
        try:
            riff, _, fmt = struct.unpack('<4sI4s', fileobj.read(12))
            if isinstance(riff, bytes):
                riff = riff.decode('utf-8')
                fmt = fmt.decode('utf-8')
            return riff == WAV_SIGNATURE[0] and fmt == WAV_SIGNATURE[1]
        except (UnicodeDecodeError, struct.error):
            return False
        finally:
            fileobj.seek(loc)
    elif filepath is not None:
        return filepath.endswith(('.wav', '.wave'))
    else:
        try:
            wave.open(args[0])
        except (wave.Error, AttributeError):
            return False
        else:
            return True


io_registry.register_reader('wav', TimeSeries, read)
io_registry.register_writer('wav', TimeSeries, write)
io_registry.register_identifier('wav', TimeSeries, is_wav)
