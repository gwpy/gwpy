# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Representation of a transient GW source
"""

from astropy import units as aunits

from .. import (version, detector)
from ..time import Time
from ..data import NDData

from .core import (Source, SourceList)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

class TransientSource(Source):
    """Generic short-duration (transient) source object.
        
    This class is designed to be sub-classed for specific sources
    """
    def __init__(self, time=None, coordinates=None, ra=None, dec=None):
        if time is not None:
            if not isinstance(time, Time):
                time = Time(time, format='gps')
            self.time = time
        if coordinates:
            if ra is not None or dec is not None:
                 raise ValueError("'ra' and/or 'dec' should not be given if "
                                  "'coordinates' is given, and vice-versa.")
            self.coordinates = coordinates
        elif ra is not None and dec is not None:
            self.coordinates = acoords.ICRSCoordinates(float(ra), float(dec),
                                                       obstime=self.time,
                                                       unit=(aunits.radian,
                                                             aunits.radian))

    @property
    def ra(self):
        return self.coordinates.ra.radians

    @property
    def dec(self):
        return self.coordinates.dec.radians

    @property
    def gps(self):
        return self.time.gps

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


class TransientSourceList(SourceList):
    """Generic list of short-duration sources.
    """
    @property
    def ra(self):
        return NDData([s.ra for s in self], name='right ascension',
                      unit=aunits.radian)

    @property
    def dec(self):
        return NDData([s.dec for s in self], name='declination',
                      unit=aunits.radian)

    @property
    def time(self):
        return NDData([s.time for s in self], name='time')

    @property
    def gps(self):
       return NDData([s.gps for s in self], name='gps', unit=aunits.second)
