# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017)
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

from astropy.table import Table

from .fetch import register_fetcher
from .. import GravitySpyTable
from .. import EventTable

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'


def get_gravityspy_triggers(tablename, engine=None,
                            **connectkwargs):
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
    import pandas as pd
    from sqlalchemy.engine import create_engine

    # connect if needed
    if engine is None:
        connectionStr = connectStr(**connectkwargs)
        engine = create_engine(connectionStr)

    try:
        tab = pd.read_sql(tablename, engine)
    except Exception as e:
        raise ValueError('I am sorry could not retrive triggers\
            from that table. The following our acceptible \
            table names {0}'.format(engine.table_names()))

    tab = Table.from_pandas(tab)

    # and return
    return GravitySpyTable(tab.filled())

# -- utilities ----------------------------------------------------------------


def connectStr(db='gravityspy', host='gravityspy.ciera.northwestern.edu',
               user=os.getenv('GRAVITYSPY_DATABASE_USER', None),
               passwd=os.getenv('GRAVITYSPY_DATABASE_PASSWD', None)):
    """Create string to pass to create_engine
    """

    if (not user) or (not passwd):
        raise ValueError('Remember to either pass '
                         'or export GRAVITYSPY_DATABASE_USER '
                         'and export GRAVITYSPY_DATABASE_PASSWD in order '
                         'to access the Gravity Spy Data: '
                         'https://secrets.ligo.org/secrets/144/'
                         ' description is username and secret is password.')

    connectionString = 'postgresql://{0}:{1}@{2}:5432/{3}'.format(
        user, passwd, host, db)

    return connectionString


register_fetcher('gravityspy', EventTable, get_gravityspy_triggers,
                 usage="tablename")
