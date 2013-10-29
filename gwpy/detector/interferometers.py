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

"""Defines objects representing the laser interferometer GW detector
"""

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ['LaserInterferometer']


class LaserInterferometer(object):
    """A model of a ground-based, laser-interferometer gravitational-wave
    detector.

    Attributes
    ----------
    name
    prefix
    vertex
    xend
    yend
    response_matrix

    Methods
    -------
    light_travel_time
    response
    time_delay
    time_delay_from_earth_center
    """
    def __init__(self):
        self.name = None
        self.prefix = None
        self.vertex = None
        self.response_matrix = None

    @property
    def _lal(self):
        """The LAL representation of this detector

        :type: :lalsuite:`LALDetector`
        """
        from lal import lalCachedDetectors
        _lal_ifos = [ifo for ifo in lalCachedDetectors if
                     ifo.frDetector.prefix == self.prefix]
        if len(_lal_ifos) == 0:
            raise ValueError("No LAL representation for detector '%s'"
                             % self.prefix)
        elif len(_lal_ifos) > 1:
            raise ValueError("Multiple LALDetectors with prefix '%s'"
                             % self.prefix)
        else:
            return _lal_ifos[0]

    def response(self, source, polarization=0.0):
        """Determine the F+, Fx antenna responses to a signal
        originating at the given source coordinates.

        Parameters
        ----------
        source : \
        :class:`~astropy.coordinates.coordsystems.SphericalCoordinatesBase`
            signal source position coordinates
        polarization : `float`, optional, default ``0.0``
            signal source polarization

        Returns
        -------
        response : `tuple`
            (F+, Fx) detector response values

        See Also
        --------
        :lalsuite:`XLALComputeDetAMResponse`
            for details on the underlying calculation
        """
        from lal import ComputeDetAMResponse
        return ComputeDetAMResponse(
                   self.response_matrix, source.ra.radians, source.dec.radians,
                   polarization,
                   lal.GreenwichMeanSiderealTime(source.obstime.gps))

    def time_delay(self, other, source):
        """Calculate the difference in signal arrival times between
        this `LaserInterferometer` and the other, based on the source
        position.

        Parameters
        ----------
        other : `LaserInterferometer`
            the other instrument against which to compare signal arrival
        source : \
        :class:`~astropy.coordinates.coordsystems.SphericalCoordinatesBase`
            signal source position coordinates

        Returns
        -------
        dt : `float`
            the difference in arrival times at each of the two
            `LaserInterferometer` detectors for the given signal

        See Also
        --------
        :lalsuite:`XLALArrivalTimeDiff`
            for details on the underlying calculation
        """
        from lal import ArrivalTimeDiff
        return ArrivalTimeDiff(self.response_matrix, other.response,
                               source.ra.radians, source.dec.radians,
                               source.obstime.gps)

    def time_delay_from_earth_center(self, source):
        """Calculate the difference in signal arrival times between
        this `LaserInterferometer` and the Earth centre, based on the source
        position.

        Parameters
        ----------
        source : \
        :class:`~astropy.coordinates.coordsystems.SphericalCoordinatesBase`
            signal source position coordinates

        Returns
        -------
        dt : `float`
            the delay in arrival of a signal from the given source at
            this `LaserInterferometer` compared to the Earth centre

        See Also
        --------
        :lalsuite:`XLALTimeDelayFromEarthCenter`
            for details on the underlying calculation
        """
        from lal import ArrivalTimeDiff
        return ArrivalTimeDiff(self.response_matrix, other.response,
                               source.ra.radians, source.dec.radians,
                               source.obstime.gps)

    def light_travel_time(self, other):
        """Calculate the line-of-sight light travel time between this
        `LaserInterferometer` and the other.

        Parameters
        ----------
        other : `LaserInterferometer`
            the other instrument against which to compare signal arrival

        Returns
        -------
        dt : `float`
            the light travel time (in seconds) between the two intstruments

        See Also
        --------
        :lalsuite:`XLALLightTravelTime`
            for details on the underlying calculation
        """
        from lal import LightTravelTime
        try:
            return LightTravelTime(self._lal, other._lal) * 1e-9
        except ValueError:
            raise ValueError("Cannot calculate light travel time without a "
                             "LAL representation of this detector")
