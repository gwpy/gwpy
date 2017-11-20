# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
#
# This file is part of GWpy.

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

import six

from ..time import to_gps
from ..utils import shell
from .cache import read_cache

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# first 4 bytes of any valid GWF file (see LIGO-T970130 §4.3.1)
if six.PY2:
    GWF_SIGNATURE = 'IGWD'
else:
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


def write_frames(filename, frames, compression=257, compression_level=6):
    """Write a list of frame objects to a file

    **Requires:** |LDAStools.frameCPP|_

    Parameters
    ----------
    filename : `str`
        path to write into

    frames : `list` of `LDAStools.frameCPP.FrameH`
        list of frames to write into file

    compression : `int`, optional
        enum value for compression scheme, default is ``GZIP``

    compression_level : `int`, optional
        compression level for given scheme
    """
    from LDAStools import frameCPP
    # open stream
    stream = frameCPP.OFrameFStream(filename)
    # write frames one-by-one
    if isinstance(frames, frameCPP.FrameH):
        frames = [frames]
    for frame in frames:
        stream.WriteFrame(frame, compression, compression_level)
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
        idx = getattr(frameCPP, 'DETECTOR_LOCATION_%s' % prefix)
        frame.AppendFrDetector(frameCPP.GetDetector(idx, gps))

    # add descriptions
    frame.SetName(name)
    frame.SetRun(run)

    return frame


# -- utilities ----------------------------------------------------------------

def num_channels(framefile):
    """Find the total number of channels in this framefile

    Parameters
    ----------
    framefile : `str`
        path to GWF-format file on disk

    Returns
    -------
    n : `int`
        the total number of channels found in the table of contents for this
        file

    Notes
    -----
    This method requires LALFrame or FrameL to run
    """
    return len(get_channel_names(framefile))


def get_channel_type(channel, framefile):
    """Find the channel type in a given frame file

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
    import lalframe

    name = str(channel)
    # read frame and table of contents
    frfile = lalframe.FrameUFrFileOpen(framefile, "r")
    frtoc = lalframe.FrameUFrTOCRead(frfile)
    for type_ in ['Sim', 'Proc', 'Adc']:
        query = getattr(lalframe, 'FrameUFrTOCQuery%sName' % type_)
        i = 0
        while True:
            try:
                chan = query(frtoc, i)
            except RuntimeError:
                break
            else:
                if chan == name:
                    return type_.lower()
            i += 1
    raise ValueError("%s not found in table-of-contents for %s"
                     % (name, framefile))


def channel_in_frame(channel, framefile):
    """Determine whether a channel is stored in this framefile

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

    Parameters
    ----------
    framefile : `str`
        path of frame file to read

    Returns
    -------
    channels : `generator`
        an iterator that will loop over the names of channels as read from
        the table of contents of the given GWF file
    """
    try:
        out = shell.call(['FrChannels', framefile])[0]
    except (OSError, shell.CalledProcessError):
        import lalframe
        # read frame and table of contents
        frfile = lalframe.FrameUFrFileOpen(framefile, "r")
        frtoc = lalframe.FrameUFrTOCRead(frfile)
        for ctype in ['Sim', 'Proc', 'Adc']:
            query = getattr(lalframe, 'FrameUFrTOCQuery%sName' % ctype)
            i = 0
            while True:
                try:
                    yield query(frtoc, i)
                except RuntimeError:
                    break
                i += 1
    else:
        for line in iter(out.splitlines()):
            chan = line.split(None, 1)[0]
            if isinstance(chan, bytes):
                yield chan.decode('utf-8')
            else:
                yield chan


def get_channel_names(framefile):
    """Return a list of all channel names found in a GWF file

    This method just returns

    >>> list(iter_channel_names(framefile))

    Parameters
    ----------
    framefile : `str`
        path of frame file to read

    Returns
    -------
    channels : `list` of `str`
        a `list` of channel names as read from the table of contents of
        the given GWF file
    """
    return list(iter_channel_names(framefile))
