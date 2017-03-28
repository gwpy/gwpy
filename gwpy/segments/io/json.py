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

"""Read/write segments and flags from DQSEGDB-format JSON
"""

from __future__ import absolute_import

import json

from six import string_types

from ...io import registry
from ...io.utils import identify_factory
from .. import DataQualityFlag

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- read ---------------------------------------------------------------------

def read_json_flag(f):
    """Read a `DataQualityFlag` from a segments-web.ligo.org JSON file
    """
    if isinstance(f, string_types):
        f = open(f, 'r')

    with f:
        txt = f.read()
        if isinstance(txt, bytes):
            txt = txt.decode('utf-8')
        data = json.loads(txt)
    name = '{ifo}:{name}:{version}'.format(**data)
    out = DataQualityFlag(name, active=data['active'],
                          known=data['known'])
    try:  # parse 'metadata'
        out.description = data['metadata'].get('flag_description', None)
    except KeyError:  # no metadata available, but that's ok
        pass
    else:
        out.isgood = not data['metadata'].get(
            'active_indicates_ifo_badness', False)

    return out


# -- write --------------------------------------------------------------------

def write_json_flag(flag, f, **kwargs):
    """Write a `DataQualityFlag` to a JSON file

    Parameters
    ----------
    flag : `DataQualityFlag`
        data to write

    f : `str`, `file`
        target file (or filename) to write

    **kwargs
        other keyword arguments to pass to :func:`json.dump`

    See also
    --------
    json.dump
        for details on acceptable keyword arguments
    """
    # build json packet
    data = {}
    data['ifo'] = flag.ifo
    data['name'] = flag.tag
    data['version'] = flag.version
    data['active'] = flag.active
    data['known'] = flag.known
    data['metadata'] = {}
    data['metadata']['active_indicates_ifo_badness'] = not flag.isgood
    data['metadata']['flag_description'] = flag.description

    if isinstance(f, string_types):
        f = open(f, 'w')

    with f:
        json.dump(data, f, **kwargs)


# -- identify -----------------------------------------------------------------

identify_json = identify_factory('json')

# -- register -----------------------------------------------------------------

registry.register_reader('json', DataQualityFlag, read_json_flag)
registry.register_writer('json', DataQualityFlag, write_json_flag)
registry.register_identifier('json', DataQualityFlag, identify_json)
