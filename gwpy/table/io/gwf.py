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

"""Read events from GWF FrEvent structures into a Table
"""

from astropy.table import Table
from astropy.io import registry as io_registry

from ...table import EventTable
from ...time import LIGOTimeGPS
from ...io import gwf as io_gwf
from ...io.cache import FILE_LIKE
from ..filter import parse_column_filters

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- read ---------------------------------------------------------------------


def _columns_from_frevent(frevent):
    """Get list of column names from frevent
    """
    params = dict(frevent.GetParam())
    return (['time', 'amplitude', 'probability', 'timeBefore', 'timeAfter',
             'comment'] + list(params.keys()))


def _row_from_frevent(frevent, columns, selection):
    """Generate a table row from an FrEvent

    Filtering (``selection``) is done here, rather than in the table reader,
    to enable filtering on columns that aren't being returned.
    """
    # read params
    params = dict(frevent.GetParam())
    params['time'] = float(LIGOTimeGPS(*frevent.GetGTime()))
    params['amplitude'] = frevent.GetAmplitude()
    params['probability'] = frevent.GetProbability()
    params['timeBefore'] = frevent.GetTimeBefore()
    params['timeAfter'] = frevent.GetTimeAfter()
    params['comment'] = frevent.GetComment()
    # filter
    if not all(op_(params[c], t) for c, op_, t in selection):
        return None
    # return event as list
    return [params[c] for c in columns]


def table_from_gwf(filename, name, columns=None, selection=None):
    """Read a Table from FrEvent structures in a GWF file (or files)

    Parameters
    ----------
    filename : `str`
        path of GWF file to read

    name : `str`
        name associated with the `FrEvent` structures

    columns : `list` of `str`
        list of column names to read

    selection : `str`, `list` of `str`
        one or more column selection strings to apply, e.g. ``'snr>6'``
    """
    # open frame file
    if isinstance(filename, FILE_LIKE):
        filename = filename.name
    stream = io_gwf.open_gwf(filename)

    # parse selections and map to column indices
    if selection is None:
        selection = []
    selection = parse_column_filters(selection)

    # read events row by row
    data = []
    i = 0
    while True:
        try:
            frevent = stream.ReadFrEvent(i, name)
        except IndexError:
            break
        i += 1
        # read first event to get column names
        if columns is None:
            columns = _columns_from_frevent(frevent)
        # read row with filter
        row = _row_from_frevent(frevent, columns, selection)
        if row is not None:  # if passed selection
            data.append(row)

    return Table(rows=data, names=columns)


# -- write --------------------------------------------------------------------

def table_to_gwf(table, filename, name, **kwargs):
    """Create a new `~frameCPP.FrameH` and fill it with data

    Parameters
    ----------
    table : `~astropy.table.Table`
        the data to write

    filename : `str`
        the name of the file to write into

    **kwargs
        other keyword arguments (see below for references)

    See also
    --------
    gwpy.io.gwf.create_frame
    gwpy.io.gwf.write_frames
        for documentation of keyword arguments
    """
    from LDAStools.frameCPP import (FrEvent, GPSTime)

    # create frame
    write_kw = {key: kwargs.pop(key) for
                key in ('compression', 'compression_level') if key in kwargs}
    frame = io_gwf.create_frame(name=name, **kwargs)

    # append row by row
    names = table.dtype.names
    for row in table:
        rowd = dict((n, row[n]) for n in names)
        gps = LIGOTimeGPS(rowd.pop('time', 0))
        frame.AppendFrEvent(FrEvent(
            str(name),
            str(rowd.pop('comment', '')),
            str(rowd.pop('inputs', '')),
            GPSTime(gps.gpsSeconds, gps.gpsNanoSeconds),
            float(rowd.pop('timeBefore', 0)),
            float(rowd.pop('timeAfter', 0)),
            int(rowd.pop('eventStatus', 0)),
            float(rowd.pop('amplitude', 0)),
            float(rowd.pop('probability', -1)),
            str(rowd.pop('statistics', '')),
            list(rowd.items()),  # remaining params as tuple
        ))

    # write frame to file
    io_gwf.write_frames(filename, [frame], **write_kw)


# -- registration -------------------------------------------------------------

for table_class in (Table, EventTable):
    io_registry.register_reader('gwf', table_class, table_from_gwf)
    io_registry.register_writer('gwf', table_class, table_to_gwf)
    io_registry.register_identifier('gwf', table_class, io_gwf.identify_gwf)
