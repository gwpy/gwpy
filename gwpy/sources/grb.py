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
    __slots__ = ['name', 'detector', 'time', 'coordinates',
                 'error', 'distance', 't90', 't1', 't2', 'fluence', 'url',
                 'trig_id']

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
    def query(cls, name, detector=None, source='grbview'):
        if source.lower() == 'grbview':
            from ..io import grbview
            return grbview.query(name, detector=detector)
        else:
            raise NotImplementedError("Querying from '%s' has not been "
                                      "implemented." % source)

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
