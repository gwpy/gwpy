# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

"""Interface to the LIGO Channel Information System
"""

import json

from six.moves.urllib.error import HTTPError

import numpy

from .. import (Channel, ChannelList)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

CIS_API_URL = 'https://cis.ligo.org/api/channel'
CIS_DATA_TYPE = {
    1: numpy.int16,
    2: numpy.int32,
    3: numpy.int64,
    4: numpy.float32,
    5: numpy.float64,
    6: numpy.complex64,
    7: numpy.uint32,
}


def query(name, use_kerberos=None, debug=False):
    """Query the Channel Information System for details on the given
    channel name

    Parameters
    ----------
    name : `~gwpy.detector.Channel`, or `str`
        Name of the channel of interest

    Returns
    -------
    channel : `~gwpy.detector.Channel`
        Channel with all details as acquired from the CIS
    """
    url = '%s/?q=%s' % (CIS_API_URL, name)
    more = True
    out = ChannelList()
    while more:
        reply = _get(url, use_kerberos=use_kerberos, debug=debug)
        try:
            out.extend(map(parse_json, reply[u'results']))
        except KeyError:
            pass
        except TypeError:  # reply is a list
            out.extend(map(parse_json, reply))
            break
        more = 'next' in reply and reply['next'] is not None
        if more:
            url = reply['next']
        else:
            break
    out.sort(key=lambda c: c.name)
    return out


def _get(url, use_kerberos=None, debug=False):
    """Perform a GET query against the CIS
    """
    from ligo.org import request

    # perform query
    try:
        response = request(url, debug=debug, use_kerberos=use_kerberos)
    except HTTPError:
        raise ValueError("Channel not found at URL %s "
                         "Information System. Please double check the "
                         "name and try again." % url)

    if isinstance(response, bytes):
        response = response.decode('utf-8')
    return json.loads(response)


def parse_json(data):
    """Parse the input data dict into a `Channel`.

    Parameters
    ----------
    data : `dict`
        input data from CIS json query

    Returns
    -------
    c : `Channel`
        a `Channel` built from the data
    """
    sample_rate = data['datarate']
    unit = data['units']
    dtype = CIS_DATA_TYPE[data['datatype']]
    model = data['source']
    url = data['displayurl']
    return Channel(data['name'], sample_rate=sample_rate, unit=unit,
                   dtype=dtype, model=model, url=url)
