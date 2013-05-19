# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Interface to the LIGO Channel Information System

This module is based on the 
"""

import json
import numpy
from urllib2 import HTTPError

from glue.auth.saml import HTTPNegotiateAuthHandler

from .. import version
from ..detector import Channel

from . import auth

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

CIS_API_URL = 'https://cis.ligo.org/api/channel/%s'
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
    url = CIS_API_URL % (name)
    try:
        response = auth.request(url, debug=debug)
    except HTTPError:
        raise ValueError("Channel named '%s' not found in Channel "
                         "Information System. Please double check the "
                         "name and try again." % name)
    channel_data = json.loads(response.read())
    name = channel_data['name']
    sample_rate = channel_data['datarate']
    unit = channel_data['units']
    dtype = CIS_DATA_TYPE[channel_data['datatype']]
    model = channel_data['source']
    url = channel_data['displayurl']
    return Channel(channel_data['name'], sample_rate=sample_rate, unit=unit,
                   dtype=dtype, model=model)
