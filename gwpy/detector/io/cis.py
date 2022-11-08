# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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


def query(name, kerberos=None, **kwargs):
    """Query the Channel Information System for details on the given
    channel name

    Parameters
    ----------
    name : `~gwpy.detector.Channel`, or `str`
        Name of the channel of interest

    kerberos : `bool`, optional
        use an existing Kerberos ticket as the authentication credential,
        default behaviour will check for credentials and request username
        and password if none are found (`None`)

    kwargs
        other keyword arguments are passed directly to
        :func:`ciecplib.get`

    Returns
    -------
    channel : `~gwpy.detector.Channel`
        Channel with all details as acquired from the CIS
    """
    url = f"{CIS_API_URL}/?q={name}"
    more = True
    out = ChannelList()
    while more:
        reply = _get(url, kerberos=kerberos, **kwargs)
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


def _get(url, kerberos=None, idp="login.ligo.org", **kwargs):
    """Perform a GET query against the CIS
    """
    from ciecplib import get as get_cis
    response = get_cis(url, endpoint=idp, kerberos=kerberos, **kwargs)
    response.raise_for_status()
    return response.json()


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
