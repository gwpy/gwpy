# Copyright (C) Cardiff University (2024-)
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

"""Core I/O utilities for GWF files.
"""

from ...segments import SegmentList
from ..cache import read_cache
from .backend import get_backend_function

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# first 4 bytes of any valid GWF file (see LIGO-T970130 §4.3.1)
GWF_SIGNATURE = b'IGWD'


# -- i/o -----------------------------

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
            except OSError:
                return False
            else:
                if cache[0].path.endswith('.gwf'):
                    return True


# -- utilities -----------------------

def num_channels(framefile, backend=None):
    """Find the total number of channels in this framefile.

    Requires a GWF backend library.

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
    return len(get_channel_names(framefile, backend=backend))


def get_channel_type(channel, framefile, backend=None):
    """Find the channel type in a given GWF file

    Requires a GWF backend library.

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
    _iter_channels = get_backend_function("_iter_channels", backend=backend)
    channel = str(channel)
    for name, type_ in _iter_channels(framefile):
        if channel == name:
            return type_
    raise ValueError(
        f"'{channel}' not found in table-of-contents for {framefile}",
    )


def channel_in_frame(channel, framefile, backend=None):
    """Determine whether a channel is stored in this framefile

    Requires a GWF backend library.

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
    for name in iter_channel_names(framefile, backend=backend):
        if channel == name:
            return True
    return False


def iter_channel_names(framefile, backend=None):
    """Iterate over the names of channels found in a GWF file

    Requires a GWF backend library.

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
    _iter_channels = get_backend_function(
        "_iter_channels",
        backend=backend,
    )
    for name, _ in _iter_channels(framefile):
        yield name


def get_channel_names(framefile, backend=None):
    """Return a list of all channel names found in a GWF file

    This method just returns

    >>> list(iter_channel_names(framefile))

    Requires a GWF backend library.

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
    return list(iter_channel_names(framefile, backend=backend))


def data_segments(paths, channel, warn=True, backend=None):
    """Returns the segments containing data for a channel

    Requires a GWF backend library.

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
    _channel_segments = get_backend_function(
        "_channel_segments",
        backend=backend,
    )
    segments = SegmentList()
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        segments.extend(_channel_segments(path, channel, warn=warn))
    return segments.coalesce()


def _get_type(type_, enum):
    """Handle a type string, or just return an `int`

    Only to be called in relation to FrProcDataType and FrProcDataSubType
    """
    if isinstance(type_, int):
        return type_
    return enum[str(type_).upper()]


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
