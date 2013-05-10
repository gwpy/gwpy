# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Query the GRBView web database
"""

import HTMLParser
import urllib2
import re
import warnings
import string

from astropy import units as aunits, coordinates as acoords, time as atime

from .. import version
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
        det = key[0]
        grb = GammaRayBurst.from_grbview(detector=det, **params)
        grbs.append(grb)
    detlist = set([grb.detector for grb in grbs])
    if detector:
        grbs = filter(lambda grb: re.match(detector, grb.detector, re.I), grbs)
    if len(grbs) == 0:
        raise KeyError("Records were found matching detectors ('%s'), but "
                       "not '%s'" % ("', '".join(detlist), detector))
    return sorted(grbs, key=lambda b: b.name)


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
