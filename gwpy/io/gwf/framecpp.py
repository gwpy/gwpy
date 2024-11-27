# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-)
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

"""GWF I/O utilities for frameCPP.
"""

import warnings
from enum import IntEnum

import numpy
from LDAStools import frameCPP

from ...segments import Segment
from ...time import (
    LIGOTimeGPS,
    to_gps,
)
from ...utils.enum import NumpyTypeEnum
from ..utils import file_path
from .core import _series_name

_FrVect = frameCPP.FrVect

# -- detectors -----------------------

DetectorLocation = IntEnum(
    "DetectorLocation",
    {key[18:]: val for key, val in vars(frameCPP).items() if
     key.startswith("DETECTOR_LOCATION_")},
)


# -- type mapping --------------------

class FrVectType(IntEnum, NumpyTypeEnum):
    INT8 = _FrVect.FR_VECT_C
    INT16 = _FrVect.FR_VECT_2S
    INT32 = _FrVect.FR_VECT_4S
    INT64 = _FrVect.FR_VECT_8S
    FLOAT32 = _FrVect.FR_VECT_4R
    FLOAT64 = _FrVect.FR_VECT_8R
    COMPLEX64 = _FrVect.FR_VECT_8C
    COMPLEX128 = _FrVect.FR_VECT_16C
    BYTES = _FrVect.FR_VECT_STRING
    UINT8 = _FrVect.FR_VECT_1U
    UINT16 = _FrVect.FR_VECT_2U
    UINT32 = _FrVect.FR_VECT_4U
    UINT64 = _FrVect.FR_VECT_8U


# -- compression types ---------------

try:
    _FrVect.ZERO_SUPPRESS
except AttributeError:  # python-ldas-tools-framecpp < 3.0.0
    class Compression(IntEnum):
        RAW = _FrVect.RAW
        GZIP = _FrVect.GZIP
        DIFF_GZIP = _FrVect.DIFF_GZIP
        ZERO_SUPPRESS_WORD_2 = _FrVect.ZERO_SUPPRESS_WORD_2
        ZERO_SUPPRESS_WORD_4 = _FrVect.ZERO_SUPPRESS_WORD_4
        ZERO_SUPPRESS_WORD_8 = _FrVect.ZERO_SUPPRESS_WORD_8
        ZERO_SUPPRESS_OTHERWISE_GZIP = _FrVect.ZERO_SUPPRESS_OTHERWISE_GZIP
else:
    class Compression(IntEnum):
        RAW = _FrVect.RAW
        BIGENDIAN_RAW = _FrVect.BIGENDIAN_RAW
        LITTLEENDIAN_RAW = _FrVect.LITTLEENDIAN_RAW
        GZIP = _FrVect.GZIP
        BIGENDIAN_GZIP = _FrVect.BIGENDIAN_GZIP
        LITTLEENDIAN_GZIP = _FrVect.LITTLEENDIAN_GZIP
        DIFF_GZIP = _FrVect.DIFF_GZIP
        BIGENDIAN_DIFF_GZIP = _FrVect.BIGENDIAN_DIFF_GZIP
        LITTLEENDIAN_DIFF_GZIP = _FrVect.LITTLEENDIAN_DIFF_GZIP
        ZERO_SUPPRESS = _FrVect.ZERO_SUPPRESS
        BIGENDIAN_ZERO_SUPPRESS = _FrVect.BIGENDIAN_ZERO_SUPPRESS
        LITTLEENDIAN_ZERO_SUPPRESS = _FrVect.LITTLEENDIAN_ZERO_SUPPRESS
        ZERO_SUPPRESS_OTHERWISE_GZIP = _FrVect.ZERO_SUPPRESS_OTHERWISE_GZIP

# compression level is '6' for all GZip compressions, otherwise 0 (none)
DefaultCompressionLevel = IntEnum(
    "DefaultCompressionLevel",
    {k: 6 if "GZIP" in k else 0 for k in Compression.__members__},
)


# -- Proc data types -----------------

class FrProcDataType(IntEnum):
    UNKNOWN = frameCPP.FrProcData.UNKNOWN_TYPE
    TIME_SERIES = frameCPP.FrProcData.TIME_SERIES
    FREQUENCY_SERIES = frameCPP.FrProcData.FREQUENCY_SERIES
    OTHER_1D_SERIES_DATA = frameCPP.FrProcData.OTHER_1D_SERIES_DATA
    TIME_FREQUENCY = frameCPP.FrProcData.TIME_FREQUENCY
    WAVELETS = frameCPP.FrProcData.WAVELETS
    MULTI_DIMENSIONAL = frameCPP.FrProcData.MULTI_DIMENSIONAL


class FrProcDataSubType(IntEnum):
    UNKNOWN = frameCPP.FrProcData.UNKNOWN_SUB_TYPE
    DFT = frameCPP.FrProcData.DFT
    AMPLITUDE_SPECTRAL_DENSITY = frameCPP.FrProcData.AMPLITUDE_SPECTRAL_DENSITY
    POWER_SPECTRAL_DENSITY = frameCPP.FrProcData.POWER_SPECTRAL_DENSITY
    CROSS_SPECTRAL_DENSITY = frameCPP.FrProcData.CROSS_SPECTRAL_DENSITY
    COHERENCE = frameCPP.FrProcData.COHERENCE
    TRANSFER_FUNCTION = frameCPP.FrProcData.TRANSFER_FUNCTION


# -- I/O functions -------------------

def open_gwf(filename, mode='r'):
    """Open a filename for reading or writing GWF format data

    Parameters
    ----------
    filename : `str`
        the path to read from, or write to

    mode : `str`, optional
        either ``'r'`` (read) or ``'w'`` (write)

    Returns
    -------
    `LDAStools.frameCPP.IFrameFStream`
        the input frame stream (if `mode='r'`), or
    `LDAStools.frameCPP.IFrameFStream`
        the output frame stream (if `mode='w'`)
    """
    if mode not in ('r', 'w'):
        raise ValueError("mode must be either 'r' or 'w'")
    filename = file_path(filename)
    if mode == 'r':
        return frameCPP.IFrameFStream(str(filename))
    return frameCPP.OFrameFStream(str(filename))


def write_frames(filename, frames, compression='GZIP', compression_level=None):
    """Write a list of frame objects to a file

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    filename : `str`
        path to write into

    frames : `list` of `LDAStools.frameCPP.FrameH`
        list of frames to write into file

    compression : `int`, `str`, optional
        name of compresion algorithm to use, or its endian-appropriate
        ID, choose from

        - ``'RAW'``
        - ``'GZIP'``
        - ``'DIFF_GZIP'``
        - ``'ZERO_SUPPRESS'``
        - ``'ZERO_SUPPRESS_OTHERWISE_GZIP'``

    compression_level : `int`, optional
        compression level for given method, default is ``6`` for GZIP-based
        methods, otherwise ``0``
    """
    # handle compression arguments
    if not isinstance(compression, int):
        compression = Compression[compression]
    if compression_level is None:
        compression_level = DefaultCompressionLevel[compression.name]

    # open stream
    stream = open_gwf(filename, 'w')

    # write frames one-by-one
    if isinstance(frames, frameCPP.FrameH):
        frames = [frames]
    for frame in frames:
        stream.WriteFrame(frame, int(compression), int(compression_level))
    # stream auto-closes (apparently)


def create_frame(time=0, duration=None, name='gwpy', run=-1, ifos=None):
    """Create a new :class:`~LDAStools.frameCPP.FrameH`

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    time : `float`, optional
        frame start time in GPS seconds

    duration : `float`, optional
        frame length in seconds

    name : `str`, optional
        name of project or other experiment description

    run : `int`, optional
        run number (number < 0 reserved for simulated data); monotonic for
        experimental runs

    ifos : `list`, optional
        list of interferometer prefices (e.g. ``'L1'``) associated with this
        frame

    Returns
    -------
    frame : :class:`~LDAStools.frameCPP.FrameH`
        the newly created frame header
    """
    # create frame
    frame = frameCPP.FrameH()

    # add timing
    gps = to_gps(time)
    gps = frameCPP.GPSTime(gps.gpsSeconds, gps.gpsNanoSeconds)
    frame.SetGTime(gps)
    if duration is not None:
        frame.SetDt(float(duration))

    # add FrDetectors
    for prefix in ifos or []:
        frame.AppendFrDetector(
            frameCPP.GetDetector(DetectorLocation[prefix], gps),
        )

    # add descriptions
    frame.SetName(name)
    frame.SetRun(run)

    return frame


def create_fradcdata(series, frame_epoch=0,
                     channelgroup=0, channelid=0, nbits=16):
    """Create a `~frameCPP.FrAdcData` from a `~gwpy.types.Series`

    .. note::

       Currently this method is restricted to 1-dimensional arrays.

    Parameters
    ----------
    series : `~gwpy.types.Series`
        the input data array to store

    frame_epoch : `float`, `int`, optional
        the GPS start epoch of the `Frame` that will contain this
        data structure

    Returns
    -------
    frdata : `~frameCPP.FrAdcData`
        the newly created data structure

    Notes
    -----
    See Table 10 (§4.3.2.4) of LIGO-T970130 for more details
    """
    # assert correct type
    if not series.xunit.is_equivalent('s') or series.ndim != 1:
        raise TypeError("only 1-dimensional timeseries data can be "
                        "written as FrAdcData")

    frdata = frameCPP.FrAdcData(
        _series_name(series),
        channelgroup,
        channelid,
        nbits,
        (1 / series.dx.to('s')).value
    )
    frdata.SetTimeOffset(
        float(to_gps(series.x0.value) - to_gps(frame_epoch)),
    )
    return frdata


def _get_series_trange(series):
    if series.xunit.is_equivalent('s'):
        return abs(series.xspan)
    return 0


def _get_series_frange(series):
    if series.xunit.is_equivalent('Hz'):  # FrequencySeries
        return abs(series.xspan)
    elif series.ndim == 2 and series.yunit.is_equivalent('Hz'):  # Spectrogram
        return abs(series.yspan)
    return 0


def create_frprocdata(series, frame_epoch=0, comment=None,
                      type=None, subtype=None, trange=None,
                      fshift=0, phase=0, frange=None, bandwidth=0):
    """Create a `~frameCPP.FrAdcData` from a `~gwpy.types.Series`

    .. note::

       Currently this method is restricted to 1-dimensional arrays.

    Parameters
    ----------
    series : `~gwpy.types.Series`
        the input data array to store

    frame_epoch : `float`, `int`, optional
        the GPS start epoch of the `Frame` that will contain this
        data structure

    comment : `str`, optional
        comment

    type : `int`, `str`, optional
        type of data object

    subtype : `int`, `str`, optional
        subtype for f-Series

    trange : `float`, optional
        duration of sampled data

    fshift : `float`, optional
        frequency in the original data that corresponds to 0 Hz in the
        heterodyned series

    phase : `float`, optional
        phase of the heterodyning signal at start of dataset

    frange : `float`, optional
        frequency range

    bandwidth : `float, optional
        reoslution bandwidth

    Returns
    -------
    frdata : `~frameCPP.FrAdcData`
        the newly created data structure

    Notes
    -----
    See Table 17 (§4.3.2.11) of LIGO-T970130 for more details
    """
    # format auxiliary data
    if trange is None:
        trange = _get_series_trange(series)
    if frange is None:
        frange = _get_series_frange(series)

    return frameCPP.FrProcData(
        _series_name(series),
        str(comment or series.name),
        _get_frprocdata_type(series, type),
        _get_frprocdata_subtype(series, subtype),
        float(to_gps(series.x0.value) - to_gps(frame_epoch)),
        trange,
        fshift,
        phase,
        frange,
        bandwidth,
    )


def create_frsimdata(series, frame_epoch=0, comment=None, fshift=0, phase=0):
    """Create a `~frameCPP.FrAdcData` from a `~gwpy.types.Series`

    .. note::

       Currently this method is restricted to 1-dimensional arrays.

    Parameters
    ----------
    series : `~gwpy.types.Series`
        the input data array to store

    frame_epoch : `float`, `int`, optional
        the GPS start epoch of the `Frame` that will contain this
        data structure

    fshift : `float`, optional
        frequency in the original data that corresponds to 0 Hz in the
        heterodyned series

    phase : `float`, optional
        phase of the heterodyning signal at start of dataset

    Returns
    -------
    frdata : `~frameCPP.FrSimData`
        the newly created data structure

    Notes
    -----
    See Table 20 (§4.3.2.14) of LIGO-T970130 for more details
    """
    # assert correct type
    if not series.xunit.is_equivalent('s'):
        raise TypeError("only timeseries data can be written as FrSimData")

    return frameCPP.FrSimData(
        _series_name(series),
        str(comment or series.name),
        (1 / series.dx.to('s')).value,
        float(to_gps(series.x0.value) - to_gps(frame_epoch)),
        fshift,
        phase,
    )


def create_frvect(series):
    """Create a `~frameCPP.FrVect` from a `~gwpy.types.Series`

    .. note::

       Currently this method is restricted to 1-dimensional arrays.

    Parameters
    ----------
    series : `~gwpy.types.Series`
        the input data array to store

    Returns
    -------
    frvect : `~frameCPP.FrVect`
        the newly created data vector
    """
    # create dimensions
    dims = frameCPP.Dimension(
        series.shape[0],  # num elements
        series.dx.value,  # step size
        str(series.dx.unit),  # unit
        0,  # starting value
    )

    # create FrVect
    vect = frameCPP.FrVect(
        _series_name(series),  # name
        int(FrVectType.find(series.dtype)),  # data type enum
        series.ndim,  # num dimensions
        dims,  # dimension object
        str(series.unit),  # unit
    )

    # populate FrVect
    vect.GetDataArray()[:] = numpy.require(series.value, requirements=['C'])

    return vect


# -- utilities -----------------------

def _iter_channels(framefile):
    """Yields the name and type of each channel in a GWF file TOC

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    framefile : `str`, `LDAStools.frameCPP.IFrameFStream`
        path of GWF file, or open file stream, to read
    """
    if not isinstance(framefile, frameCPP.IFrameFStream):
        framefile = open_gwf(framefile, 'r')
    toc = framefile.GetTOC()
    for typename in ('Sim', 'Proc', 'ADC'):
        typen = typename.lower()
        for name in getattr(toc, f"Get{typename}")():
            yield name, typen


def _channel_segments(path, channel, warn=True):
    """Yields the segments containing data for ``channel`` in this GWF path
    """
    stream = open_gwf(path)
    # get segments for frames
    toc = stream.GetTOC()
    secs = toc.GetGTimeS()
    nano = toc.GetGTimeN()
    dur = toc.GetDt()

    readers = [getattr(stream, f"ReadFr{type_.title()}Data") for
               type_ in ("proc", "sim", "adc")]

    # for each segment, try and read the data for this channel
    for i, (s, ns, dt) in enumerate(zip(secs, nano, dur)):
        for read in readers:
            try:
                read(i, channel)
            except (IndexError, ValueError):
                continue
            readers = [read]  # use this one from now on
            epoch = LIGOTimeGPS(s, ns)
            yield Segment(epoch, epoch + dt)
            break
        else:  # none of the readers worked for this channel, warn
            if warn:
                warnings.warn(
                    f"'{channel}' not found in frame {i} of {path}",
                )


def _get_type(type_, enum):
    """Handle a type string, or just return an `int`

    Only to be called in relation to FrProcDataType and FrProcDataSubType
    """
    if isinstance(type_, int):
        return type_
    return enum[str(type_).upper()]


def _get_frprocdata_type(series, type_):
    """Determine the appropriate `FrProcDataType` for this series

    Notes
    -----
    See Table 17 (§4.3.2.11) of LIGO-T970130 for more details
    """
    if type_ is not None:  # format user value
        return _get_type(type_, FrProcDataType)

    if series.ndim == 1 and series.xunit.is_equivalent("s"):
        type_ = FrProcDataType.TIME_SERIES
    elif series.ndim == 1 and series.xunit.is_equivalent("Hz"):
        type_ = FrProcDataType.FREQUENCY_SERIES
    elif series.ndim == 1:
        type_ = FrProcDataType.OTHER_1D_SERIES_DATA
    elif (
            series.ndim == 2
            and series.xunit.is_equivalent("s")
            and series.yunit.is_equivalent("Hz")
    ):
        type_ = FrProcDataType.TIME_FREQUENCY
    elif series.ndim > 2:
        type_ = FrProcDataType.MULTI_DIMENSIONAL
    else:
        type_ = FrProcDataType.UNKNOWN

    return type_


def _get_frprocdata_subtype(series, subtype):
    """Determine the appropriate `FrProcDataSubType` for this series

    Notes
    -----
    See Table 17 (§4.3.2.11) of LIGO-T970130 for more details
    """
    if subtype is not None:  # format user value
        return _get_type(subtype, FrProcDataSubType)

    if series.unit == 'coherence':
        return FrProcDataSubType.COHERENCE
    return FrProcDataSubType.UNKNOWN
