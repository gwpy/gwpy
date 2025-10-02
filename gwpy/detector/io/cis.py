# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Interface to the LIGO Channel Information System."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    TypedDict,
)

import numpy

from .. import (
    Channel,
    ChannelList,
)

if TYPE_CHECKING:
    from typing import NotRequired

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# -- CIS definitions -----------------

CIS_API_URL = "https://cis.ligo.org/api/channel"
CIS_DATA_TYPE = {
    1: numpy.int16,
    2: numpy.int32,
    3: numpy.int64,
    4: numpy.float32,
    5: numpy.float64,
    6: numpy.complex64,
    7: numpy.uint32,
}


class CisChannelResponse(TypedDict):
    """Response from the CIS for one channel."""

    datarate: float
    datatype: int
    displayurl: str
    name: str
    source: str
    units: str


class CisResponse(TypedDict):
    """Response from a CIS API query."""

    # list of channels
    results: list[CisChannelResponse]
    # pager
    next: NotRequired[str]


# -- I/O routines --------------------

def query(
    name: str | Channel,
    *,
    kerberos: bool | None = None,
    **kwargs,
) -> ChannelList:
    """Query the Channel Information System for details on a channel.

    Parameters
    ----------
    name : `~gwpy.detector.Channel`, `str`
        Name of the channel of interest.

    kerberos : `bool`, `None`, optional
        Use an existing Kerberos ticket as the authentication credential.
        Default behaviour (`kerberos=None`) is to check for credentials
        and request username and passowrd (interactively) if none are foudn.

    kwargs
        Other keyword arguments are passed directly to :func:`ciecplib.get`.

    Returns
    -------
    channel : `~gwpy.detector.Channel`
        Channel with all details as acquired from the CIS
    """
    out = ChannelList()
    url: str | None = f"{CIS_API_URL}/?q={name}"
    while url:
        reply = _get(url, kerberos=kerberos, **kwargs)

        # list result
        if not isinstance(reply, dict):
            out.extend(map(_parse_json, reply))
            break

        # (paged) result
        out.extend(map(_parse_json, reply["results"]))
        url = reply.get("next", None)

    out.sort(key=lambda c: c.name)
    return out


def _get(
    url: str,
    *,
    kerberos: bool | None = None,
    idp: str = "login.ligo.org",
    **kwargs,
) -> CisResponse | list[CisChannelResponse]:
    """Perform a GET query against the CIS."""
    from ciecplib import get as get_cis
    response = get_cis(url, endpoint=idp, kerberos=kerberos, **kwargs)
    response.raise_for_status()
    return response.json()


def _parse_json(
    data: CisChannelResponse,
) -> Channel:
    """Parse the input data dict into a `Channel`.

    Parameters
    ----------
    data : `dict`
        Input data from CIS json query.

    Returns
    -------
    c : `Channel`
        A `Channel` built from the data.
    """
    sample_rate = data["datarate"]
    unit = data["units"]
    dtype = CIS_DATA_TYPE[data["datatype"]]
    model = data["source"]
    url = data["displayurl"]
    return Channel(
        data["name"],
        sample_rate=sample_rate,
        unit=unit,
        dtype=dtype,
        model=model,
        url=url,
    )
