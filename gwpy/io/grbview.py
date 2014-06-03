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

"""Query the GRBView web database
"""

import copy
import itertools
import numpy
import HTMLParser
import urllib2
import re
import warnings
import string

from astropy import units as aunits, coordinates as acoords, time as atime

from .. import (version, time)
from ..sources import GammaRayBurst

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

GRBVIEW_URL = "http://grbweb.icecube.wisc.edu/GRBview.php?GRB=%s"

GRBVIEW_KEY_MAP = {'GBM Public Data':'fermigbm'}

class GRBViewParser(HTMLParser.HTMLParser):

    def __init__(self, *args, **kwargs):
        HTMLParser.HTMLParser.__init__(self, *args, **kwargs)
        self.name_open = False
        self._key = None
        self.records = {}
        self._table_pos = 0
        self._table_open = False

    def handle_starttag(self, tag, attrs):
        self._tag = tag

        # handle tables
        if tag == 'table':
            self._table_open = True
            self._headers = []
            self._data = []

    def handle_data(self, data):
        if self._tag == 'font' and data:
            if data == 'Notice:':
                self._table_open = False
                return
            self._key = data.rstrip(':')
            if self._key in GRBVIEW_KEY_MAP.keys():
               self._key = GRBVIEW_KEY_MAP[self._key]
            self.records[self._key] = {}
        if self._table_open:
            if self._tag == 'th':
                self._headers.append(data.lower())
            elif self._tag == 'td':
                self._data.append(data)

    def handle_endtag(self, tag):
        if tag == 'table' and self._table_open and self._key:
            self._table_open = False
            self.records[self._key] = dict(zip(self._headers, self._data))

def query(name, detector=None):
    grb = name.upper()
    if grb.startswith("GRB"):
        grb = grb[3:]
    records = _query(grb)
    if not re.search('[A-Z]\Z', grb):
        i = 0
        while True:
            grb2 = grb + string.uppercase[i]
            new = _query(grb2)
            if new:
                records.update(new)
            else:
                break
            i +=1
    if len(records) == 0:
        raise ValueError("No records were found matching GRB='%s'" % name)
    grbs = []
    for key, params in records.iteritems():
        grb = GammaRayBurst()
        grb.name = params.get("grbname", None)
        grb.detector = key[0]
        grb.url = params.get("ftext", None)
        grb.time = params.get("uttime", None)
        grb.trig_id = int(params.get('trig', params.get('trig1', None)))
        if grb.time and grb.time != '-':
            grb.time = time.Time(grb.time, scale="utc")
        ra = params.get("ra", None)
        if ra == '-':
            ra = None
        dec = params.get("decl", None)
        if dec == '-':
            dec = None
        if ra and dec:
            grb.coordinates = acoords.ICRSCoordinates(float(ra), float(dec),
                                                      obstime=grb.time,
                                                      unit=(aunits.degree,
                                                            aunits.degree))
        err = params.get("err", None)
        if err and err != '-':
            grb.error = aunits.Quantity(float(err), unit=aunits.degree)
        t90 = params.get("t90", None)
        if t90 and t90 != '-':
            grb.t90 = aunits.Quantity(float(t90), unit=aunits.second)
        t1 = params.get("t1", None)
        if t1 and t1 != '-':
            grb.t1 = float(t1)
        t2 = params.get("t2", None)
        if t2 and t2 != '-':
            grb.t2 = float(t2)
        fluence = params.get("fluence", None)
        if fluence and fluence != '-':
            grb.fluence = aunits.Quantity(float(fluence), "erg / cm**2")
        grbs.append(grb)
    grbs = parse_grbview(grbs)
    detlist = set([grb.detector for grb in grbs])
    if detector:
        grbs = filter(lambda grb: re.match(detector, grb.detector, re.I), grbs)
    if len(grbs) == 0:
        raise KeyError("Records were found matching detectors ('%s'), but "
                       "not '%s'" % ("', '".join(detlist), detector))
    return sorted(grbs, key=lambda b: b.name)


def parse_grbview(grbs):
    """Parse the returns from grbview to get unique triggers
    """
    if len(grbs) == 1:
        return grbs
    grbs.sort(key=lambda grb: grb.detector)
    pairs = itertools.combinations(grbs, 2)
    grb_triggers = []
    keep = numpy.zeros(len(grbs)).astype(bool)
    for i,pair in enumerate(pairs):
        a,b = pair
        if a.trig_id is not None and a.trig_id == b.trig_id:
            new = GammaRayBurst()
            for attr in a.__slots__:
                if hasattr(a, attr) and getattr(a, attr):
                    setattr(new, attr, getattr(a, attr))
            for attr in b.__slots__:
                if attr == 'time' and b.time.iso.endswith('00:00:00.000'):
                    continue
                elif hasattr(b, attr) and getattr(b, attr):
                    setattr(new, attr, getattr(b, attr))
            try:
                new.coordinates = acoords.ICRSCoordinates(float(new.ra),
                                                          float(new.dec),
                                                          obstime=new.time,
                                                          unit=(aunits.radian,
                                                                aunits.radian))
            except AttributeError:
                pass
            if a in grb_triggers:
                grb_triggers.pop(grb_triggers.index(a))
            if b in grb_triggers:
                grb_triggers.pop(grb_triggers.index(b))
            grb_triggers.append(new)
        elif a not in grb_triggers:
            grb_triggers.append(a)
        elif b not in grb_triggers:
            grb_triggers.append(b)

    return grb_triggers


def _query(name):
    url = GRBVIEW_URL % (name)
    parser = GRBViewParser()
    parser.feed(urllib2.urlopen(url).read())
    parser.close()
    records = {}
    for detector,record in parser.records.iteritems():
        if detector is None:
            continue
        records[(detector, record["grbname"])] = record
    return records
