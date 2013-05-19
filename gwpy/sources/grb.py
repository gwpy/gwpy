# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Representation of the gamma-ray burst
"""

from math import log10
from scipy import stats

from astropy import units as aunits, coordinates as acoords

from .. import (time, version, detector)
from ..utils import lal

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

# long/short GRB distributions (log t90)  source: astro-ph/0205004
SHORT_GRB_DIST = stats.norm(scale=0.61, loc=-0.11)
LONG_GRB_DIST = stats.norm(scale=0.43, loc=1.54)


class GammaRayBurst(object):
    __slots__ = ["name", "detector", "time", "coordinates", "ra", "dec",
                 "error", "distance", "t90", "t1", "t2", "fluence", "url"]

    def __init__(self, **args):
        for key,val in args:
            setattr(key, val)

    @property
    def ra(self):
        return self.coordinates.ra.radians

    @property
    def dec(self):
        return self.coordinates.dec.radians

    @property
    def gps(self):
        if lal.SWIG_LAL:
            return lal.swiglal.LIGOTimeGPS(self.time.gps)
        else:
            return self.time.gps

    @classmethod
    def from_grbview(cls, **params):
        out = cls()
        out.name = params.get("grbname", None)
        out.detector = params.get("detector", None)
        out.url = params.get("ftext", None)
        out.time = params.get("uttime", None)
        if out.time and out.time != '-':
            out.time = time.Time(out.time, scale="utc")
        ra = params.get("ra", None)
        dec = params.get("decl", None)
        if ra and dec:
            out.coordinates = acoords.ICRSCoordinates(float(ra), float(dec),
                                                      obstime=out.time,
                                                      unit=(aunits.degree,
                                                            aunits.degree))
        err = params.get("err", None)
        if err and err != '-':
            out.error = aunits.Quantity(float(err), unit=aunits.degree)
        t90 = params.get("t90", None)
        if t90 and t90 != '-':
            out.t90 = aunits.Quantity(float(t90)*1e-3, unit=aunits.second)
        t1 = params.get("t1", None)
        if t1 and t1 != '-':
            out.t1 = float(t1)
        t2 = params.get("t2", None)
        if t2 and t2 != '-':
            out.t2 = float(t2)
        fluence = params.get("fluence", None)
        if fluence and fluence != '-':
            out.fluence = aunits.Quantity(float(fluence), "erg / cm**2")
        return out

    def is_short(self):
        sp = SHORT_GRB_DIST.pdf(log10(self.t90))
        lp = LONG_GRB_DIST.pdf(log10(self.t90))
        return sp / (sp + lp)

    def is_long(self):
        sp = SHORT_GRB_DIST.pdf(log10(self.t90))
        lp = LONG_GRB_DIST.pdf(log10(self.t90))
        return lp / (sp + lp)

    def antenna_reponse(self, ifo=None):
        if ifo:
            ifos = [ifo]
        else:
            ifos = detector.DETECTOR_BY_PREFIX.keys()
        quadsum = lambda r: (r[0]**2 + r[1]**2)**(1/2.)
        response = {}
        for det in ifos:
            response[det] = quadsum(detector.DETECTOR_BY_PREFIX[det].response(
                                        self.coordinates))
        if isinstance(ifo, basestring):
            return response[ifo]
        else:
            return response

    def __str__(self):
        return "GRB%s" % self.name

    def __repr__(self):
        return "GammaRayBurst(%s, detector='%s')" % (str(self), self.detector)
