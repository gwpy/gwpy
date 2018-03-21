# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018)
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

"""I/O routines for accessing LIGO segments through LOSC
"""

from astropy.utils.data import get_readable_fileobj

from ...io.losc import (LOSC_URL, find_datasets)
from ...segments import (Segment, SegmentList)
from ...time import (LIGOTimeGPS, to_gps)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def timeline_url(flag, start, end, host=LOSC_URL):
    """Construct the LOSC API URL to the given flag and GPS [start, stop).

    This method is mainly designed to support the :func:`get_segments`
    function in the same module.

    Parameters
    ----------
    flag : `str`
        the name of the flag to query

    start : `int`, `str`
        the GPS start time (or parseable date string) to query

    end : `int`, `str`
        the GPS end time (or parseable date string) to query

    host : `str`, optional
        URL of LOSC host, default: ``'losc.ligo.org'``

    Returns
    -------
    url : `str`
        the LOSC URL that will return matching timeline segments

    Examples
    --------
    >>> from gwpy.segments.io.losc import timeline_url
    >>> print(timeline_url('H1_DATA', 'Jan 1 2010', 'Jan 2 2010'))
    https://losc.ligo.org/timeline/segments/S6/H1_DATA/946339215/86400/
    """
    start = to_gps(start).gpsSeconds
    end = to_gps(end).gpsSeconds
    ifo = flag.split('_', 1)[0]
    try:
        dataset = find_datasets(start, end, detector=ifo)[0]
    except IndexError as exc:
        exc.args = ('No datasets matching [{} ... {}) for '
                    'detector={}'.format(start, end, ifo),)
        raise

    return '{}/timeline/segments/{}/{}/{}/{}/'.format(
        host, dataset, flag, start, end-start)


def get_segments(flag, start, end, verbose=False, timeout=60, host=LOSC_URL):
    """Download segments for a given flag within the GPS interval.

    Parameters
    ----------
    flag : `str`
        the name of the flag to query

    start : `int`, `str`
        the GPS start time (or parseable date string) to query

    end : `int`, `str`
        the GPS end time (or parseable date string) to query

    verbose : `bool`, optional
        show verbose download progress, default: `False`

    timeout : `int`, optional
        timeout for download (seconds)

    host : `str`, optional
        URL of LOSC host, default: ``'losc.ligo.org'``

    Returns
    -------
    segments : `~gwpy.segments.SegmentList`
        the list of GPS ``[start, end)`` segments

    Examples
    --------
    To download the list of GPS segments during which LIGO-Hanford was
    operational on Jan 1 2010:

    >>> from gwpy.segments.io.losc import get_segments
    >>> print(get_segments('H1_DATA', 'Jan 1 2010', 'Jan 2 2010'))
    [[946340946 ... 946351800)
     [946356479 ... 946360620)
     [946362652 ... 946369150)
     [946372854 ... 946382630)
     [946395595 ... 946396751)
     [946400173 ... 946404977)
     [946412312 ... 946413577)
     [946415770 ... 946422986)]
    """
    span = Segment(to_gps(start), to_gps(end))
    url = timeline_url(flag, start, end, host=host)
    segments = SegmentList()
    with get_readable_fileobj(url, show_progress=verbose,
                              remote_timeout=timeout) as response:
        data = response.read()
        for line in data.splitlines():
            start, end, duration = map(int, line.split())
            if duration != end - start:
                raise RuntimeError("Corrupt segment received from LOSC: "
                                   "printed duration does not match segment "
                                   "boundaries: {}".format(line.rstrip()))
            segments.append(Segment(LIGOTimeGPS(start), LIGOTimeGPS(end)))
    return SegmentList([span]) & segments
