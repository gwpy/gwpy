# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Interface to the LIGO Channel Information System

This module is based on the 
"""

import json

from glue.auth.saml import HTTPNegotiateAuthHandler

from .. import version
from ..channels import Channel

from . import auth

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

CIS_CHANNEL_URL = 'https://cis.ligo.org/api/channel/%s'


def query(name, debug=False):
    """Query the Channel Information System for details on the given
    channel name

    Parameters
    ----------
    name : `~gwpy.channels.Channel`, or `str`
        Name of the channel of interest

    Returns
    -------
    channel : `~gwpy.channels.Channel`
        Channel with all details as acquired from the CIS
    """
    url = CIS_CHANNEL_URL % (name)
    response = auth.request(url, debug=debug)
    channel_data = json.loads(response.read())
    name = channel_data['name']
    sample_rate = channel_data['datarate']
    unit = channel_data['units']
    type_ = channel_data['datatype']
    model = channel_data['source']
    url = channel_data['displayurl']
    return Channel(channel_data['name'], sample_rate=sample_rate, unit=unit,
                   type=type_, model=model, url=url)
