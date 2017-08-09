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

- ``QUEST_SQL_USER``
- ``QUEST_SQL_PASSWORD``

These can be found https://secrets.ligo.org/secrets/144/. The description
is the username and thesecret is the password.
"""
import os

from .. import GravitySpyTable
from .. import EventTable
from astropy.table import Table
from .fetch import register_fetcher

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'

# get GravitySpy environment variables
QUEST_SQL_USER = os.getenv('QUEST_SQL_USER', None)
QUEST_SQL_PASSWORD = os.getenv('QUEST_SQL_PASSWORD', None)


def get_gravityspy_triggers(tablename, selection=None, **kwargs):
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

    Notes"""
    try:
        import pandas as pd
    except ImportError as e:
        e.args = ('pandas is required to download triggers',)
        raise

    try:
        from sqlalchemy.engine import create_engine
    except ImportError as e:
        e.args = ('sqlalchemy is required to download triggers',)
        raise

    if (not QUEST_SQL_USER) or (not QUEST_SQL_PASSWORD):
        raise ValueError('Remember to set export QUEST_SQL_USER \
            and export QUEST_SQL_PASSWORD in order to access the \
            Gravity Spy Data: https://secrets.ligo.org/secrets/144/\
             description is username and secret is password.')

    engine = create_engine('postgresql://{0}:{1}\
        @gravityspy.ciera.northwestern.edu:5432/gravityspy'
        .format(os.environ['QUEST_SQL_USER'], 
            os.environ['QUEST_SQL_PASSWORD']))

    try:
        # parse selections and map to column indices
        if selection is None:
            selectionstr = 'SELECT * FROM \"{0}\"'.format(tablename)
        else:
            selectionstr = 'SELECT * FROM \"{0}\"'.format(tablename)
        tab = pd.read_sql(selectionstr, engine)
    except Exception as e:
        raise ValueError('I am sorry could not retrive triggers\
            from that table. The following our acceptible \
            table names {0}'.format(engine.table_names()))

    tab = Table.from_pandas(tab)

    # and return
    return GravitySpyTable(tab.filled())


register_fetcher('gravityspy', EventTable, get_gravityspy_triggers,
                 usage="tablename")
