# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Representation of the gamma-ray burst
"""

from math import log10
from scipy import stats

import lal

from astropy import units as aunits, coordinates as acoords

from .. import (time, version, detector)

from .transient import (TransientSource, TransientSourceList)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

# long/short GRB distributions (log t90)  source: astro-ph/0205004
SHORT_GRB_DIST = stats.norm(scale=0.61, loc=-0.11)
LONG_GRB_DIST = stats.norm(scale=0.43, loc=1.54)


class GammaRayBurst(TransientSource):
    __slots__ = ['name', 'detector', 'time', 'coordinates',
                 'error', 'distance', 't90', 't1', 't2', 'fluence', 'url',
                 'trig_id']

    @classmethod
    def query(cls, name, detector=None, source='grbview'):
        grbs = GammaRayBurstList(name, detector=detector, source=source)
        if len(grbs) > 1:
            raise ValueError("Multiple records found for this GRB name."
                             "Please refine your search, or use "
                             "GammaRayBurstList.query to return all records.")
        return grbs[0]

    def is_short(self):
        sp = SHORT_GRB_DIST.pdf(log10(self.t90))
        lp = LONG_GRB_DIST.pdf(log10(self.t90))
        return sp / (sp + lp)

    def is_long(self):
        sp = SHORT_GRB_DIST.pdf(log10(self.t90))
        lp = LONG_GRB_DIST.pdf(log10(self.t90))
        return lp / (sp + lp)

    def __str__(self):
        return "GRB%s" % self.name

    def __repr__(self):
        return "GammaRayBurst(%s, detector='%s')" % (str(self), self.detector)


class GammaRayBurstList(TransientSourceList):
    """Representation of a list of GammaRayBursts.

    This list can represent multiple individual bursts, or individual
    detections of the same burst by different sattelites
    """
    @classmethod
    def query(cls, name, detector=None, source='grbview'):
        """Query the given source to find information on the given
        GammaRayBurst name
        """
        if source.lower() == 'grbview':
            from ..io import grbview
            return grbview.query(name, detector=detector)
        else:
            raise NotImplementedError("Querying from '%s' has not been "
                                      "implemented." % source)
