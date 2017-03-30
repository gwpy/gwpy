# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
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

from astropy.table import (Table, Row, vstack as vstack_tables)
from astropy.io import registry as io_registry

from ...table import EventTable
from ...time import LIGOTimeGPS
from ...io import gwf as io_gwf
from ...io.cache import FILE_LIKE
from ...io.utils import identify_factory

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- read ---------------------------------------------------------------------


def get_columns_from_frevent(frevent):
    """Get list of column names from frevent
    """
    params = dict(frevent.GetParam())
    return (['time', 'amplitude', 'probability', 'timeBefore', 'timeAfter',
             'comment'] + list(params.keys()))


def row_from_frevent(frevent, columns=None, row_class=Row):
    """Generate a table row from an FrEvent
    """
    if columns is None:
        columns = get_columns_from_frevent(frevent)
    params = dict(frevent.GetParam())
    params['time'] = float(LIGOTimeGPS(*frevent.GetGTime()))
    params['amplitude'] = frevent.GetAmplitude()
    params['probability'] = frevent.GetProbability()
    params['timeBefore'] = frevent.GetTimeBefore()
    params['timeAfter'] = frevent.GetTimeAfter()
    params['comment'] = frevent.GetComment()
    return [params[c] for c in columns]


def table_from_gwf(filename, name, columns=None):
    """Read a Table from FrEvent structures in a GWF file (or files)

    Parameters
    ----------
    filename : `str`, `list` of `str`
        path of GWF file (or files) to read

    name : `str`
        name associated with the `FrEvent` structures

    columns : `
    """
    # read list of files
    if isinstance(filename, list):
        return vstack_tables([table_from_gwf(f, name, columns=columns) for
                              f in filename])

    # -- read single file from here --

    from LDAStools import frameCPP

    # open frame file
    if isinstance(filename, FILE_LIKE):
        filename = filename.name
    stream = frameCPP.IFrameFStream(filename)

    # read events row by row
    data = []
    i = 0
    while True:
        try:
            frevent = stream.ReadFrEvent(i, name)
        except IndexError as e:
            break
        if columns is None:  # read first event to get column names
            columns = get_columns_from_frevent(frevent)
        data.append(row_from_frevent(frevent, columns=columns))
        i += 1

    return Table(rows=data, names=columns)


# -- write --------------------------------------------------------------------

def table_to_gwf(table, filename, name, start=0, duration=None,
                 run=0, ifos=[], compression=257, compression_level=6):
    """Create a new `~frameCPP.FrameH` and fill it with data
    """
    from LDAStools.frameCPP import (FrEvent, GPSTime)

    # create frame
    frame = io_gwf.create_frame(time=start, duration=duration, name=name,
                                run=run, ifos=ifos)

    # append row by row
    names = table.dtype.names
    for row in table:
        rowd = dict((n, row[n]) for n in names)
        t = LIGOTimeGPS(rowd.pop('time', 0))
        frame.AppendFrEvent(FrEvent(
            str(name),
            str(rowd.pop('comment', '')),
            str(rowd.pop('inputs', '')),
            GPSTime(t.gpsSeconds, t.gpsNanoSeconds),
            float(rowd.pop('timeBefore', 0)),
            float(rowd.pop('timeAfter', 0)),
            int(rowd.pop('eventStatus', 0)),
            float(rowd.pop('amplitude', 0)),
            float(rowd.pop('probability', -1)),
            str(rowd.pop('statistics', '')),
            list(rowd.items()),  # remaining params as tuple
        ))

    # write frame to file
    io_gwf.write_frames(filename, [frame], compression=compression,
                        compression_level=compression_level)


# -- registration -------------------------------------------------------------

for table_class in (Table, EventTable):
    io_registry.register_reader('gwf', table_class, table_from_gwf)
    io_registry.register_writer('gwf', table_class, table_to_gwf)
    io_registry.register_identifier('gwf', table_class, io_gwf.identify_gwf)
