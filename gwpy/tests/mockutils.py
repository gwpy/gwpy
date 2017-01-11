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

"""Mock objects for GWpy tests
"""

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- DQSEGDB calls ------------------------------------------------------------

def mock_query_times(result, deactivated=False,
                     active_indicates_ifo_badness=False, **kwargs):
    """Build a mock of `dqsegdb.apicalls.dqsegdbQueryTimes` for testing
    """

    def query_times(protocol, server, ifo, name, version, request, start, end):
        flag = '%s:%s:%d' % (ifo, name, version)
        return {
            'ifo': ifo,
            'name': name,
            'version': version,
            'known': list(map(tuple, result[flag].known)),
            'active': list(map(tuple, result[flag].active)),
            'query_information': {},
            'metadata': kwargs,
        }, 'BOGUS_QUERY_STRING'

    return query_times


def mock_dqsegdb_cascaded_query(result, deactivated=False,
                                active_indicates_ifo_badness=False, **kwargs):
    """Build a mock of `dqsegdb.apicalls.dqsegdbCascadedQuery` for testing
    """

    def cascaded_query(protocol, server, ifo, name, request, start, end):
        # this is a bit hacky, but it's just for tests
        flag = [x for x in result if
                x.rsplit(':', 1)[0] == '%s:%s' % (ifo, name)][0]
        version = int(flag.split(':')[-1])
        return (
            {'known': list(map(tuple, result[flag].known)),
             'active': list(map(tuple, result[flag].active)),
             'ifo': ifo,
             'name': 'RESULT',
             'version': 1},
            [{'ifo': ifo,
             'name': name,
             'version': version,
             'known': list(map(tuple, result[flag].known)),
             'active': list(map(tuple, result[flag].active)),
             'query_information': {},
             'metadata': kwargs}],
            {},
        )

    return cascaded_query
