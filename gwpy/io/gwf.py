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

"""I/O utilities for GWF files using the lalframe API
"""

from ..time import to_gps
from ..utils import (shell, with_import)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@with_import('lalframe')
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


@with_import('lalframe')
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
    name = str(channel)
    # read frame and table of contents
    frfile = lalframe.FrameUFrFileOpen(framefile, "r")
    frtoc = lalframe.FrameUFrTOCRead(frfile)
    for type_ in ['Sim', 'Proc', 'Adc']:
        query = getattr(lalframe, 'FrameUFrTOCQuery%sName' % type_)
        i = 0
        while True:
            try:
                c = query(frtoc, i)
            except RuntimeError:
                break
            else:
                if c == name:
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
            yield line.split(b' ', 1)[0].decode('utf-8')


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
