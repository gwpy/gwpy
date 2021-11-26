# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""I/O utilities for GWF files using the lalframe or frameCPP APIs
"""

import warnings

import numpy

from ..segments import (Segment, SegmentList)
from ..time import (to_gps, LIGOTimeGPS)
from .cache import read_cache
from .utils import file_path

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# first 4 bytes of any valid GWF file (see LIGO-T970130 §4.3.1)
GWF_SIGNATURE = b'IGWD'


# -- i/o ----------------------------------------------------------------------

def identify_gwf(origin, filepath, fileobj, *args, **kwargs):
    """Identify a filename or file object as GWF

    This function is overloaded in that it will also identify a cache file
    as 'gwf' if the first entry in the cache contains a GWF file extension
    """
    # pylint: disable=unused-argument

    # try and read file descriptor
    if fileobj is not None:
        loc = fileobj.tell()
        fileobj.seek(0)
        try:
            if fileobj.read(4) == GWF_SIGNATURE:
                return True
        finally:
            fileobj.seek(loc)
    if filepath is not None:
        if filepath.endswith('.gwf'):
            return True
        if filepath.endswith(('.lcf', '.cache')):
            try:
                cache = read_cache(filepath)
            except IOError:
                return False
            else:
                if cache[0].path.endswith('.gwf'):
                    return True


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
    from LDAStools import frameCPP
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
        - ``'ZERO_SUPPRESS_WORD_2'``
        - ``'ZERO_SUPPRESS_WORD_4'``
        - ``'ZERO_SUPPRESS_WORD_8'``
        - ``'ZERO_SUPPRESS_OTHERWISE_GZIP'``

    compression_level : `int`, optional
        compression level for given method, default is ``6`` for GZIP-based
        methods, otherwise ``0``
    """
    from LDAStools import frameCPP
    from ._framecpp import (Compression, DefaultCompressionLevel)

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
    from LDAStools import frameCPP
    from ._framecpp import DetectorLocation

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
    from LDAStools import frameCPP

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
    from LDAStools import frameCPP

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
    from LDAStools import frameCPP

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
    from LDAStools import frameCPP
    from ._framecpp import FrVectType

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


# -- utilities ----------------------------------------------------------------

def num_channels(framefile):
    """Find the total number of channels in this framefile

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    framefile : `str`
        path to GWF-format file on disk

    Returns
    -------
    n : `int`
        the total number of channels found in the table of contents for this
        file
    """
    return len(get_channel_names(framefile))


def get_channel_type(channel, framefile):
    """Find the channel type in a given GWF file

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    channel : `str`, `~gwpy.detector.Channel`
        name of data channel to find

    framefile : `str`
        path of GWF file in which to search

    Returns
    -------
    ctype : `str`
        the type of the channel ('adc', 'sim', or 'proc')

    Raises
    ------
    ValueError
        if the channel is not found in the table-of-contents
    """
    channel = str(channel)
    for name, type_ in _iter_channels(framefile):
        if channel == name:
            return type_
    raise ValueError(
        f"'{channel}' not found in table-of-contents for {framefile}",
    )


def channel_in_frame(channel, framefile):
    """Determine whether a channel is stored in this framefile

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    channel : `str`
        name of channel to find

    framefile : `str`
        path of GWF file to test

    Returns
    -------
    inframe : `bool`
        whether this channel is included in the table of contents for
        the given framefile
    """
    channel = str(channel)
    for name in iter_channel_names(framefile):
        if channel == name:
            return True
    return False


def iter_channel_names(framefile):
    """Iterate over the names of channels found in a GWF file

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    framefile : `str`
        path of GWF file to read

    Returns
    -------
    channels : `generator`
        an iterator that will loop over the names of channels as read from
        the table of contents of the given GWF file
    """
    for name, _ in _iter_channels(framefile):
        yield name


def get_channel_names(framefile):
    """Return a list of all channel names found in a GWF file

    This method just returns

    >>> list(iter_channel_names(framefile))

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    framefile : `str`
        path of GWF file to read

    Returns
    -------
    channels : `list` of `str`
        a `list` of channel names as read from the table of contents of
        the given GWF file
    """
    return list(iter_channel_names(framefile))


def _iter_channels(framefile):
    """Yields the name and type of each channel in a GWF file TOC

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    framefile : `str`, `LDAStools.frameCPP.IFrameFStream`
        path of GWF file, or open file stream, to read
    """
    from LDAStools import frameCPP
    if not isinstance(framefile, frameCPP.IFrameFStream):
        framefile = open_gwf(framefile, 'r')
    toc = framefile.GetTOC()
    for typename in ('Sim', 'Proc', 'ADC'):
        typen = typename.lower()
        for name in getattr(toc, f"Get{typename}")():
            yield name, typen


def data_segments(paths, channel, warn=True):
    """Returns the segments containing data for a channel

    **Requires:** |LDAStools.frameCPP|_

    A frame is considered to contain data if a valid FrData structure
    (of any type) exists for the channel in that frame.  No checks
    are directly made against the underlying FrVect structures.

    Parameters
    ----------
    paths : `list` of `str`
        a list of GWF file paths

    channel : `str`
        the name to check in each frame

    warn : `bool`, optional
        emit a `UserWarning` when a channel is not found in a frame

    Returns
    -------
    segments : `~gwpy.segments.SegmentList`
        the list of segments containing data
    """
    segments = SegmentList()
    for path in paths:
        segments.extend(_gwf_channel_segments(path, channel, warn=warn))
    return segments.coalesce()


def _gwf_channel_segments(path, channel, warn=True):
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
    from ._framecpp import FrProcDataType

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
    from ._framecpp import FrProcDataSubType

    if subtype is not None:  # format user value
        return _get_type(subtype, FrProcDataSubType)

    if series.unit == 'coherence':
        return FrProcDataSubType.COHERENCE
    return FrProcDataSubType.UNKNOWN


def _series_name(series):
    """Returns the 'name' of a `Series` that should be written to GWF

    This is basically `series.name or str(series.channel) or ""`

    Parameters
    ----------
    series : `gwpy.types.Series`
        the input series that will be written

    Returns
    -------
    name : `str`
        the name to use when storing this series
    """
    return (
        series.name
        or str(series.channel or "")
        or None
    )
