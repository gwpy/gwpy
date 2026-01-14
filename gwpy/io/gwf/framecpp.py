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

"""GWF I/O utilities for frameCPP."""

from __future__ import annotations

import warnings
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    overload,
)

import numpy
from LDAStools import frameCPP

from ...segments import Segment
from ...time import (
    LIGOTimeGPS,
    to_gps,
)
from ...utils.enum import NumpyTypeEnum
from ..utils import file_path
from .core import (
    FRDATA_TYPES,
    _series_name,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Iterator,
    )
    from pathlib import Path
    from typing import (
        IO,
        Literal,
    )

    from ...time import SupportsToGps
    from ...types import Series

_FrVect = frameCPP.FrVect

# -- detectors -----------------------

DETECTORS: dict[str, int] = {
    key[18:]: val for key, val in vars(frameCPP).items()
    if key.startswith("DETECTOR_LOCATION_")
}
DetectorLocation = IntEnum("DetectorLocation", DETECTORS)


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


@overload
def open_gwf(
    gwf: str | Path | IO | frameCPP.IFrameFStream | frameCPP.OFrameFStream,
    mode: Literal["r"],
) -> frameCPP.IFrameFStream:
    ...

@overload
def open_gwf(
    gwf: str | Path | IO | frameCPP.IFrameFStream | frameCPP.OFrameFStream,
    mode: Literal["w"],
) -> frameCPP.OFrameFStream:
    ...

def open_gwf(
    gwf: str | Path | IO | frameCPP.IFrameFStream | frameCPP.OFrameFStream,
    mode: Literal["r", "w"] = "r",
) -> frameCPP.IFrameFStream | frameCPP.OFrameFStream:
    """Open a stream for reading or writing GWF format data.

    Parameters
    ----------
    gwf : `str`, `pathlib.Path`, `file`, or open frameCPP stream
        The path to read from, or write to. Already open GWF streams
        are returned unmodified, if the type matches the mode.

    mode : `str`, optional
        The mode with which to open the file, either `r` or `w`.

    Returns
    -------
    stream : `LDAStools.frameCPP.IFrameFStream`, or `LDAStools.frameCPP.OFrameFStream`
        The file stream for reading or writing.
    """
    # check mode
    if mode not in ("r", "w"):
        msg = "mode must be either 'r' or 'w'"
        raise ValueError(msg)
    # match open file stream
    if mode == "r" and isinstance(gwf, frameCPP.IFrameFStream):
        return gwf
    if mode == "w" and isinstance(gwf, frameCPP.OFrameFStream):
        return gwf
    # open a new stream
    filename = file_path(gwf)
    if mode == "r":
        return frameCPP.IFrameFStream(filename)
    return frameCPP.OFrameFStream(filename)


def write_frames(
    gwf: str | Path | IO,
    frames: Iterable[frameCPP.FrameH],
    compression: int | str | None = None,
    compression_level: int | None = None,
):
    """Write a list of frame objects to a file.

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
        - ``'GZIP'`` (default)
        - ``'DIFF_GZIP'``
        - ``'ZERO_SUPPRESS'``
        - ``'ZERO_SUPPRESS_OTHERWISE_GZIP'``

    compression_level : `int`, optional
        compression level for given method, default is ``6`` for GZIP-based
        methods, otherwise ``0``
    """
    # handle compression arguments
    comp: Compression
    if compression is None:
        comp = Compression.GZIP
    elif isinstance(compression, int):
        comp = Compression(compression)
    else:
        comp = Compression[compression]
    if compression_level is None:
        compression_level = DefaultCompressionLevel[comp.name].value

    # open stream for writing
    stream = open_gwf(gwf, "w")

    # write frames one-by-one
    if isinstance(frames, frameCPP.FrameH):
        frames = [frames]
    for frame in frames:
        stream.WriteFrame(frame, int(comp.value), int(compression_level))
    # stream auto-closes (apparently)


def create_frame(
    time: SupportsToGps = 0,
    duration: float | None = None,
    name: str = "gwpy",
    run: int = -1,
    ifos: Iterable[str] | None = None,
) -> frameCPP.FrameH:
    """Create a new :class:`~LDAStools.frameCPP.FrameH`.

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    time : `float`, optional
        Frame start time in GPS seconds.

    duration : `float`, optional
        Frame length in seconds.

    name : `str`, optional
        Name of project or other experiment description.

    run : `int`, optional
        Run number (number < 0 reserved for simulated data); monotonic for
        experimental runs.

    ifos : `list`, optional
        List of interferometer prefices (e.g. ``'L1'``) associated with this
        frame.

    Returns
    -------
    frame : :class:`~LDAStools.frameCPP.FrameH`
        The newly created frame header.
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


def create_fradcdata(
    series: Series,
    frame_epoch: SupportsToGps = 0,
    channelgroup: int = 0,
    channelid: int = 0,
    nbits: int = 16,
) -> frameCPP.FrAdcData:
    """Create a `~frameCPP.FrAdcData` from a `~gwpy.types.Series`.

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
    See Table 10 (§4.3.2.4) of :dcc:`LIGO-T970130` for more details.
    """
    # assert correct type
    if not series.xunit.is_equivalent("s") or series.ndim != 1:
        msg = "only 1-dimensional timeseries data can be written as FrAdcData"
        raise TypeError(msg)

    frdata = frameCPP.FrAdcData(
        _series_name(series),
        channelgroup,
        channelid,
        nbits,
        (1 / series.dx.to("s")).value,
    )
    frdata.SetTimeOffset(
        float(to_gps(series.x0.value) - to_gps(frame_epoch)),
    )
    return frdata


def _get_series_trange(series: Series) -> float:
    if series.xunit.is_equivalent("s"):
        return abs(series.xspan)
    return 0


def _get_series_frange(series: Series) -> float:
    if series.xunit.is_equivalent("Hz"):  # FrequencySeries
        return abs(series.xspan)
    if series.ndim == 2 and series.yunit.is_equivalent("Hz"):  # Spectrogram
        return abs(series.yspan)
    return 0


def create_frprocdata(
    series: Series,
    frame_epoch: SupportsToGps = 0,
    comment: str | None = None,
    type: int | str | None = None,
    subtype: int | str | None = None,
    trange: float | None = None,
    fshift: float = 0,
    phase: float = 0,
    frange: float | None = None,
    bandwidth: float = 0,
) -> frameCPP.FrProcData:
    """Create a `~frameCPP.FrProcData` from a `~gwpy.types.Series`.

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
    See Table 17 (§4.3.2.11) of :dcc:`LIGO-T970130` for more details.
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


def create_frsimdata(
    series: Series,
    frame_epoch: SupportsToGps = 0,
    comment: str | None = None,
    fshift: float = 0,
    phase: float = 0,
) -> frameCPP.FrSimData:
    """Create a `~frameCPP.FrSimData` from a `~gwpy.types.Series`.

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
    See Table 20 (§4.3.2.14) of :dcc:`LIGO-T970130` for more details.
    """
    # assert correct type
    if not series.xunit.is_equivalent("s"):
        msg = "only timeseries data can be written as FrSimData"
        raise TypeError(msg)

    return frameCPP.FrSimData(
        _series_name(series),
        str(comment or series.name),
        (1 / series.dx.to("s")).value,
        float(to_gps(series.x0.value) - to_gps(frame_epoch)),
        fshift,
        phase,
    )


def create_frvect(series: Series) -> frameCPP.FrVect:
    """Create a `~frameCPP.FrVect` from a `~gwpy.types.Series`.

    .. note::

       Currently this method is restricted to 1-dimensional arrays.

    Parameters
    ----------
    series : `~gwpy.types.Series`
        The input data array to store.

    Returns
    -------
    frvect : :class:`LDASTools.frameCPP.FrVect`
        The newly created data vector.
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
        int(FrVectType.find(series.dtype).value),  # data type enum
        series.ndim,  # num dimensions
        dims,  # dimension object
        str(series.unit),  # unit
    )

    # populate FrVect
    vect.GetDataArray()[:] = numpy.require(series.value, requirements=["C"])

    return vect


# -- utilities -----------------------

@overload
def _iter_toc(
    gwf: str | Path | IO | frameCPP.IFrameFStream,
    type: str | None,
    count: Literal[True],
) -> Iterator[int]:
    ...

@overload
def _iter_toc(
    gwf: str | Path | IO | frameCPP.IFrameFStream,
    type: str | None,
    count: Literal[False],
) -> Iterator[tuple[str, str]]:
    ...

def _iter_toc(
    gwf: str | Path | IO | frameCPP.IFrameFStream,
    type: str | None = None,
    count: bool = False,
) -> Iterator[tuple[str, str] | int]:
    """Yields the name and type of each channel in a GWF file TOC.

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    gwf : `str`, `LDAStools.frameCPP.IFrameFStream`
        Path of GWF file, or open file stream, to read.

    type : `str`, optional
        Only yield items of the given type (case insensitive).
        Default is all types 'adc', 'proc', 'or 'sim'.

    Yields
    ------
    name, type : `str`, `str`
        The data channel name and type (one of 'adc', 'proc', or 'sim').
    """
    stream = open_gwf(gwf, mode="r")
    toc = stream.GetTOC()
    if type is None:
        types = FRDATA_TYPES
    else:
        types = (type,)
    for typename in map(str.lower, types):
        if typename == "adc":
            getter = toc.GetADC
        else:
            getter = getattr(toc, f"Get{typename.title()}")
        if count:  # just count, don't unpack
            yield len(getter())
            continue
        for name in getter():
            yield name, typename


def _count_toc(
    gwf: str | Path | IO | frameCPP.IFrameFStream,
    type: str | None = None,
) -> int:
    """Yield the names and types of channels listed in the TOC for a GWF file.

    Parameters
    ----------
    gwf : `str`, `pathlib.Path`, `file`, `lalframe.FrameUFrFile`
        Path of GWF file, or open file stream, to read.

    Yields
    ------
    name : `str`
        The name of a channel
    type : `str`
        The ``FrVect`` type of the channel, one of ``"sim"``, ``"proc"``,
        or ``"adc"``.
    """
    return sum(_iter_toc(gwf, type=type, count=True))


def _channel_segments(
    gwf: str | Path | IO | frameCPP.IFrameFStream,
    channel: str,
    warn: bool = True,
) -> Iterator[Segment]:
    """Yields the segments containing data for ``channel`` in this GWF path."""
    stream = open_gwf(gwf, mode="r")

    # get segments for frames
    toc = stream.GetTOC()
    secs = toc.GetGTimeS()
    nano = toc.GetGTimeN()
    dur = toc.GetDt()

    readers: tuple[Callable, ...] = tuple(
        getattr(stream, f"ReadFr{type_.title()}Data")
        for type_ in FRDATA_TYPES
    )

    # for each segment, try and read the data for this channel
    for i, (s, ns, dt) in enumerate(zip(secs, nano, dur, strict=True)):
        for read in readers:
            try:
                read(i, channel)
            except (IndexError, ValueError):
                continue
            readers = (read,)  # use this one from now on
            epoch = LIGOTimeGPS(s, ns)
            yield Segment(epoch, epoch + dt)
            break
        else:  # none of the readers worked for this channel, warn
            if warn:
                warnings.warn(
                    f"'{channel}' not found in frame {i} of {gwf}",
                    stacklevel=2,
                )


def _get_type(
    type_: int | str,
    enum: type[IntEnum],
) -> int:
    """Handle a type string, or just return an `int`.

    Only to be called in relation to FrProcDataType and FrProcDataSubType
    """
    if isinstance(type_, int):
        return type_
    return enum[str(type_).upper()].value


def _get_frprocdata_type(
    series: Series,
    type_: int | str | None,
) -> int:
    """Determine the appropriate `FrProcDataType` for this series.

    Notes
    -----
    See Table 17 (§4.3.2.11) of :dcc:`LIGO-T970130` for more details.
    """
    if type_ is not None:  # format user value
        return _get_type(type_, FrProcDataType)

    if series.ndim == 1:
        if series.xunit.is_equivalent("s"):
            return FrProcDataType.TIME_SERIES.value
        if series.xunit.is_equivalent("Hz"):
            return FrProcDataType.FREQUENCY_SERIES.value
        return FrProcDataType.OTHER_1D_SERIES_DATA.value
    if (
        series.ndim == 2
        and series.xunit.is_equivalent("s")
        and series.yunit.is_equivalent("Hz")
    ):
        return FrProcDataType.TIME_FREQUENCY.value
    if series.ndim > 2:
        return FrProcDataType.MULTI_DIMENSIONAL.value

    return FrProcDataType.UNKNOWN.value


def _get_frprocdata_subtype(
    series: Series,
    subtype: int | str | None,
) -> int:
    """Determine the appropriate `FrProcDataSubType` for this series.

    Notes
    -----
    See Table 17 (§4.3.2.11) of :dcc:`LIGO-T970130` for more details.
    """
    if subtype is not None:  # format user value
        return _get_type(subtype, FrProcDataSubType)

    if series.unit == "coherence":
        return FrProcDataSubType.COHERENCE.value
    return FrProcDataSubType.UNKNOWN.value
