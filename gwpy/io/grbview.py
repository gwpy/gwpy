# Licensed under a 3-clause BSD style license - see LICENSE.rst

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

class GRBViewParser(HTMLParser.HTMLParser):

    def __init__(self, *args, **kwargs):
        HTMLParser.HTMLParser.__init__(self, *args, **kwargs)
        self.name_open = False
        self._key = None
        self.records = {}
        self._table_pos = 0

    def handle_starttag(self, tag, attrs):
        if tag == "input":
            attrs = dict(attrs)
            name = attrs['name']
            value = attrs['value']
            if value.lower().startswith('<a'):
                value = re.split('[=>]', value.lower())[1]
            if name == 'table':
                self.records[value] = {}
                self._key = value
            elif name.startswith("colname"):
                idx = int(name[7:])
                if self.records[self._key].has_key(idx):
                    self.records[self._key][idx][0] = value.lower()
                else:
                    self.records[self._key][idx] = [value.lower(), None]
            elif name.startswith("va"):
                idx = int(name[2:])
                if self.records[self._key].has_key(idx):
                    self.records[self._key][idx][1] = value
                else:
                    self.records[self._key][idx] = [None, value]


def query(name, detector=None):
    grb = name.upper()
    if grb.startswith("GRB"):
        grb = grb[3:]
    records = _query(name)
    if not re.match('[A-Z]', grb):
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
                                                      unit=(aunits.radian,
                                                            aunits.radian))
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
                                                          unit=(aunits.degree,
                                                                aunits.degree))
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
    for detector,rec in parser.records.iteritems():
        record = dict(rec.values())
        records[(detector, record["grbname"])] = record
    return records
