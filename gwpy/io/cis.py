# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Interface to the LIGO Channel Information System

This module is based on the 
"""

import json
import numpy
from urllib2 import HTTPError

from glue.auth.saml import HTTPNegotiateAuthHandler

from .. import version
from ..detector import (Channel, ChannelList)

from . import auth

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

CIS_API_URL = 'https://cis.ligo.org/api/channel'
CIS_DATA_TYPE = {4: numpy.float32}


def query(name, debug=False):
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
        reply = _get(url, debug=debug)
        try:
           out.extend(map(parse_json, reply[u'results']))
        except KeyError:
           pass
        more = reply.has_key('next') and reply['next'] is not None
        if more:
            url = reply['next']
        else:
            break
    out.sort(key=lambda c: c.name)
    return out


def _get(url, debug=False):
    """Perform a GET query against the CIS
    """
    try:
        response = auth.request(url, debug=debug)
    except HTTPError:
        raise ValueError("Channel named '%s' not found in Channel "
                         "Information System. Please double check the "
                         "name and try again." % name)
    return json.loads(response.read())

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
    name = data['name']
    sample_rate = data['datarate']
    unit = data['units']
    dtype = CIS_DATA_TYPE[data['datatype']]
    model = data['source']
    url = data['displayurl']
    return Channel(data['name'], sample_rate=sample_rate, unit=unit,
                   dtype=dtype, model=model)
