# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Input/output methods for tabular data.

Access to HACR triggers requires local access to the MySQL database. Users
can set the IP address of the server, and the username and password for
connections in the following environment variables

- ``HACR_DATABASE_SERVER``
- ``HACR_DATABASE_USER``
- ``HACR_DATABASE_PASSWD``

These can also be given directly to the connection function as keyword
arguments
"""

import datetime
import os.path
from dateutil.relativedelta import relativedelta

import numpy
from numpy.lib import recfunctions

from ...segments import Segment
from ...time import (to_gps, from_gps)
from ...utils.deps import with_import
from .. import EventTable

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# get HACR environment variables
HACR_DATABASE_SERVER = os.getenv('HACR_DATABASE_SERVER', None)
HACR_DATABASE_USER = os.getenv('HACR_DATABASE_USER', None)
HACR_DATABASE_PASSWD = os.getenv('HACR_DATABASE_PASSWD', None)

HACR_COLUMNS = [
    'gps_start',
    'gps_offset',
    'freq_central',
    'bandwidth',
    'duration',
    'num_pixels',
    'snr',
    'totPower',
]


def get_database_names(start, end):
    # convert to datetimes
    start = from_gps(to_gps(start))
    end = from_gps(to_gps(end))
    # loop over months
    dbs = []
    d = start
    dt = relativedelta(months=1)
    while d < end:
        dbs.append('geo%s' % d.strftime('%Y%m'))
        d += dt
    return dbs


def get_hacr_channels(db=None, gps=None, connection=None,
                      **conectkwargs):
    """Return the names of all channels present in the given HACR database
    """
    # connect if needed
    if connection is None:
        if gps is None:
            gps = from_gps('now')
        if db is None:
            db = get_database_names(gps, gps)[0]
        connection = connect(db=db, **conectkwargs)
    # query
    out = query("select channel from job where monitorName = 'chacr'")
    return [r[0] for r in out]


def get_hacr_triggers(channel, start, end, columns=HACR_COLUMNS, pid=None,
                      monitor='chacr', **connectkwargs):
    """Fetch a table of HACR triggers in the given interval
    """
    if columns is None:
        columns = HACR_COLUMNS
    columns = list(columns)
    data = []
    span = Segment(map(to_gps, (start, end)))

    # allow user to specify 'time'
    ucolumns = list(columns)
    try:
        columns.pop(columns.index('time'))
    except ValueError:
        addtime = False
    else:
        # make sure that we query for 'gps_start' and 'gps_offset'
        # optionally popping those columns out before returning
        # (if they weren't given in the first place)
        addtime = True
        popstart = False
        popoffset = False
        if 'gps_start' not in columns:
            columns.append('gps_start')
            popstart = True
        if 'gps_offset' not in columns:
            columns.append('gps_offset')
            popoffset = True

    # get database names and loop over each on
    databases = get_database_names(start, end)
    rows = []
    for db in databases:
        conn = connect(db, **connectkwargs)
        cursor = conn.cursor()
        # find process ID(s) for this channel
        pids = query("select process_id, gps_start, gps_stop "
                     "from job where monitorName = %r and channel = %r"
                     % (monitor, str(channel)), connection=conn)
        for p, s, e in pids:
            # validate this process id
            if pid is not None and int(p) != int(pid):
                continue
            tspan = Segment(float(s), float(e))
            if not tspan.intersects(span):
                continue
            # execute trigger query
            q = ('select %s from mhacr where process_id = %d and '
                 'gps_start > %s and gps_start < %d order by gps_start asc'
                 % (', '.join(columns), int(p), span[0], span[1]))
            n = cursor.execute(q)
            if n == 0:
                continue
            # get new events, convert to recarray, and append to table
            rows.extend(cursor.fetchall())
            dtype = [(c, type(x)) for c, x in zip(columns, new[0])]
    return EventTable(rows=rows, names=columns)


def add_time_column(table, name='time', pop_start=True, pop_offset=True):
    """Append a column named 'time' by combining the gps_start and _offset

    Parameters
    ----------
    table : `EventTable`
        table of events to modify
    name : `str`, optional
        name of field to append, default: 'time'
    pop_start: `bool`, optional
        remove the 'gps_start' field when finished, default: `True`
    pop_offset: `bool`, optional
        remove the 'gps_offset' field when finished, default: `True`

    Returns
    -------
    mod : `recarray`, matches type of input
        a modified version of the input table with the new time field
    """
    type_ = type(table)
    t = table['gps_start'] + table['gps_offset']
    drop = []
    if pop_start:
        drop.append('gps_start')
    if pop_offset:
        drop.append('gps_offset')
    if drop:
        table = recfunctions.rec_drop_fields(table, drop)
    return recfunctions.rec_append_fields(table, [name], [t]).view(type_)


# -- utilities ----------------------------------------------------------------

@with_import('MySQLdb')
def connect(db, host=HACR_DATABASE_SERVER, user=HACR_DATABASE_USER,
            passwd=HACR_DATABASE_PASSWD):
    """Connect to the given MySQL database
    """
    return MySQLdb.connect(host=host, user=user, passwd=passwd, db=db)


def query(querystr, connection=None, **connectkwargs):
    """Execute a query of the given MySQL database
    """
    if connection is None:
        connection = connect(**connectkwargs)
    cursor = connection.cursor()
    cursor.execute(querystr)
    return cursor.fetchall()
