# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""I/O utilities for LIGO Open Science Center data

For more details, see https://losc.ligo.org
"""

import json
from collections import namedtuple
from operator import attrgetter

from astropy.utils.data import get_readable_fileobj

from ..segments import (Segment, SegmentList)
from ..time import to_gps

# default URL
LOSC_URL = 'https://losc.ligo.org'


# -- JSON handling ------------------------------------------------------------

def fetch_json(url, verbose=False):
    """Fetch JSON data from a remote URL

    Parameters
    ----------
    url : `str`
        the remote URL to fetch

    verbose : `bool`, optional
        display verbose download progress, default: `False`

    Returns
    ------
    json : `object`
        the data fetched from ``url`` as parsed by :func:`json.loads`

    See also
    --------
    json.loads
        for details of the JSON parsing

    Examples
    --------
    >>> from gwpy.io.losc import fetch_json
    >>> fetch_json('https://losc.ligo.org/archive/1126257414/1126261510/json/')
    """
    with get_readable_fileobj(url, show_progress=verbose) as response:
        data = response.read()
        try:
            return json.loads(data)
        except ValueError as exc:
            exc.args = ("Failed to parse LOSC JSON from %r: %s"
                        % (url, str(exc)),)
            raise


# -- API calls ----------------------------------------------------------------

def fetch_dataset_json(gpsstart, gpsend, host=LOSC_URL):
    """Returns the JSON metadata for all datasets matching the GPS interval

    Parameters
    ----------
    gpsstart : `int`
        the GPS start of the desired interval

    gpsend : `int`
        the GPS end of the desired interval

    host : `str`, optional
        the URL of the LOSC host to query, defaults to losc.ligo.org

    Returns
    -------
    json
        the JSON data retrieved from LOSC and returned by `json.loads`
    """
    url = '{}/archive/{:d}/{:d}/json/'.format(host, gpsstart, gpsend)
    return fetch_json(url)


def fetch_event_json(event, host=LOSC_URL):
    """Returns the JSON metadata for the given event
    """
    url = '{}/archive/{}/json/'.format(host, event)
    return fetch_json(url)


def fetch_run_json(run, detector, gpsstart, gpsend, host=LOSC_URL):
    """Returns the JSON metadata for the given science run parameters

    Parameters
    ----------
    run : `str`
        the name of the science run, e.g. ``'O1'``

    detector : `str`
        the prefix of the GW detector, e.g. ``'L1'``

    gpsstart : `int`
        the GPS start of the desired interval

    gpsend : `int`
        the GPS end of the desired interval

    host : `str`, optional
        the URL of the LOSC host to query, defaults to losc.ligo.org

    Returns
    -------
    json
        the JSON data retrieved from LOSC and returned by `json.loads`
    """
    url = '{}/archive/links/{}/{}/{:d}/{:d}/json/'.format(
        host, run, detector, gpsstart, gpsend)
    return fetch_json(url)


# -- utilities ----------------------------------------------------------------

def event_gps(event, host=LOSC_URL):
    """Returns the GPS time of an open-data event

    Parameters
    ----------
    event : `str`
        the name of the event to query

    host : `str`, optional
        the URL of the LOSC host to query, defaults to losc.ligo.org

    Returns
    -------
    gps : `float`
        the GPS time of this event
    """
    return fetch_event_json(event, host=host)['GPS']


def event_segment(event, host=LOSC_URL, **match):
    """Returns the GPS segment covered by a LOSC event dataset

    Parameters
    ----------
    event : `str`
        the name of the event to query

    host : `str`, optional
        the URL of the LOSC host to query, defaults to losc.ligo.org

    **match : metadata requirements, optional
        restrictions for matched files, e.g. ``detector='L1'``, all
        keys must be present in JSON metadata for all files

    Returns
    -------
    segment : `~gwpy.segments.Segment`
        the GPS ``[start, end)`` segment matched for this event

    Examples
    --------
    >>> from gwpy.io.losc import event_segment
    >>> event_segment('GW150914')
    [1126257414 ... 1126261510)
    """
    jdata = fetch_event_json(event, host=host)
    seg = None
    for fmeta in sieve_urls(jdata['strain'], **match):
        start = fmeta['GPSstart']
        end = start + fmeta['duration']
        fseg = Segment(start, end)
        if seg is None:
            seg = fseg
        else:
            seg |= fseg
    if seg is None:
        raise ValueError("No files matched for event {}".format(event))
    return seg


def sieve_urls(urllist, **match):
    """Sieve a list of LOSC URL metadata dicts based on key, value pairs

    This method simply matches keys from the ``match`` keywords with those
    found in the JSON dicts for a file URL returned by the LOSC API.
    """
    for urlmeta in urllist:
        if any(match[key] != urlmeta[key] for key in match):
            continue
        yield urlmeta


# -- epoch discovery ----------------------------------------------------------

def find_datasets(start, end, detector=None, strict=False, host=LOSC_URL):
    """Returns the datasets that could contain data for the given interval

    Parameters
    ----------
    start : `int`
        the GPS start time of the interval

    end : `int`
        the GPS end time of the interval

    detector : `str`, optional
        the observatory prefix for the detector of interest, e.g. ``'L1'``

    strict : `bool`, optional
        return only those datasets with 100% coverage of the interval,
        default: `False` (return any dataset overlapping the interval)

    host : `str`, optional
        the URL of the LOSC host to query, defaults to losc.ligo.org

    Returns
    -------
    datasets : `tuple`
        a `tuple` containing the name of each dataset that covers the
        requested interval

    Examples
    --------
    >>> from gwpy.io.losc import find_datasets
    >>> find_datasets(934000000, 934100000)
    (u'S6', u'tenyear')
    >>> find_datasets(1126257414, 1126261510)
    (u'GW150914', u'tenyear')
    """
    start = to_gps(start).gpsSeconds
    end = to_gps(end).gpsSeconds
    span = Segment(start, end)
    jdata = fetch_dataset_json(start, end, host=host)

    Dataset = namedtuple('Dataset', ('name', 'span', 'gap'))

    # extract epochs
    epochs = []
    for epochtype in jdata:
        for epoch, metadata in jdata[epochtype].items():
            if detector and detector not in metadata['detectors']:
                continue

            # get dataset segment
            if epochtype == 'events' and detector:
                epochseg = event_segment(epoch, detector=detector, host=host)
            elif epochtype == 'events':
                epochseg = event_segment(epoch, host=host)
            else:
                epochseg = Segment(metadata['GPSstart'], metadata['GPSend'])

            # match segment against request
            try:
                coverage = span & epochseg
            except (ValueError, TypeError):
                continue
            gap = abs(SegmentList([span]) - SegmentList([epochseg]))
            if strict and gap:
                continue
            epochs.append(Dataset(epoch, abs(epochseg), gap))

    # sort epochs by coverage
    return list(zip(*sorted(epochs, key=attrgetter('gap', 'span'))))[0]
