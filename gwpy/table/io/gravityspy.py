# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017-2020)
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

Access to GravitySpy and O1GlitchClassification triggers requires access
to a PostgresSQL database. Users can set the username and password for
connections in the following environment variables

- ``GRAVITYSPY_DATABASE_USER``
- ``GRAVITYSPY_DATABASE_PASSWORD``

These can be found https://secrets.ligo.org/secrets/144/. The description
is the username and thesecret is the password.
"""

import os

from .sql import fetch
from .fetch import register_fetcher
from .. import GravitySpyTable
from .. import EventTable

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'


def get_gravityspy_triggers(tablename, engine=None, **kwargs):
    """Fetch data into an `GravitySpyTable`

    Parameters
    ----------
    table : `str`,
        The name of table you are attempting to receive triggers
        from.

    selection
        other filters you would like to supply
        underlying reader method for the given format

    .. note::

       For now it will attempt to automatically connect you
       to a specific DB. In the future, this may be an input
       argument.

    Returns
    -------
    table : `GravitySpyTable`
    """
    from sqlalchemy.engine import create_engine
    from sqlalchemy.exc import ProgrammingError

    # connect if needed
    if engine is None:
        conn_kw = {}
        for key in ('db', 'host', 'user', 'passwd'):
            try:
                conn_kw[key] = kwargs.pop(key)
            except KeyError:
                pass
        engine = create_engine(get_connection_str(**conn_kw))

    try:
        return GravitySpyTable(fetch(engine, tablename, **kwargs))
    except ProgrammingError as exc:
        if 'relation "%s" does not exist' % tablename in str(exc):
            msg = exc.args[0]
            msg = msg.replace(
                'does not exist',
                'does not exist, the following tablenames are '
                'acceptable:\n    %s\n' % '\n    '.join(engine.table_names()))
            exc.args = (msg,)
        raise


# -- utilities ----------------------------------------------------------------

def get_connection_str(db='gravityspy',
                       host='gravityspyplus.ciera.northwestern.edu',
                       user=None,
                       passwd=None):
    """Create string to pass to create_engine

    Parameters
    ----------
    db : `str`, default: ``gravityspy``
        The name of the SQL database your connecting to.

    host : `str`, default: ``gravityspyplus.ciera.northwestern.edu``
        The name of the server the database you are connecting to
        lives on.

    user : `str`, default: `None`
        Your username for authentication to this database.

    passwd : `str`, default: `None`
        Your password for authentication to this database.

    .. note::

       `user` and `passwd` should be given together, otherwise they will be
       ignored and values will be resolved from the
       ``GRAVITYSPY_DATABASE_USER`` and ``GRAVITYSPY_DATABASE_PASSWD``
       environment variables.

    Returns
    -------
    conn_string : `str`
        A SQLAlchemy engine compliant connection string
    """
    if (not user) or (not passwd):
        user = os.getenv('GRAVITYSPY_DATABASE_USER', None)
        passwd = os.getenv('GRAVITYSPY_DATABASE_PASSWD', None)

    if (not user) or (not passwd):
        raise ValueError('Remember to either pass '
                         'or export GRAVITYSPY_DATABASE_USER '
                         'and export GRAVITYSPY_DATABASE_PASSWD in order '
                         'to access the Gravity Spy Data: '
                         'https://secrets.ligo.org/secrets/144/'
                         ' description is username and secret is password.')

    return 'postgresql://{0}:{1}@{2}:5432/{3}'.format(user, passwd, host, db)


register_fetcher('gravityspy', EventTable, get_gravityspy_triggers,
                 usage="tablename")
