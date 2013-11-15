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

import warnings
import re

from astropy.coordinates import (CartesianPoints, SphericalCoordinatesBase)
from astropy.units import Unit

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ['LaserInterferometer']


class ConventionWarning(UserWarning):
    pass


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
        self._name = None
        self._prefix = None
        self._vertex = None
        self._xend = None
        self._yend = None
        self._response_matrix = None

    # ------------------------------------------------------------------------
    # LaserInterferometer attributes

    @property
    def name(self):
        """Name for this `LaserInterferometer`

        :type: `str`
        """
        return self._name

    @name.setter
    def name(self, n):
        self._name = n is not None and str(n) or n

    @property
    def prefix(self):
        """Identification prefix for this `LaserInterferometer`.

        The `prefix` should follow the convention of single upper-case
        character followed by a single number, e.g. 'X1'.

        :type: `str`
        """
        return self._prefix

    @prefix.setter
    def prefix(self, ifo):
        if not re.match('[A-Z][12]', ifo):
            warnings.warn("Prefix '%s' does not match the 'X1' convention",
                          ConventionWarning)
        self._prefix = ifo

    @property
    def vertex(self):
        """Position of the vertex for this `LaserInterferometer`.

        This should be the position of the beam-splitter, relative to
        the Earth centre, in earth-fixed coordinates.

        :type: :class:`~astropy.coordinates.distances.CartesianPoints`
        """
        return self._vertex

    @vertex.setter
    def vertex(self, v):
        if isinstance(v, CartesianPoints):
            self._vertex = v
        elif isinstance(v, SphericalCoordinatesBase):
            self._vertex = v.cartesian
        elif hasattr(v, 'unit'):
            self._vertex = CartesianPoints(*v, unit=v.unit)
        else:
            self._vertex = CartesianPoints(*v, unit=Unit('m'))

    @property
    def xend(self):
        """Position of the x-arm end mirror for this `LaserInterferometer`.

        This should be the position of the x-end optic, relative to
        the Earth centre, in earth-fixed coordinates.

        :type: :class:`~astropy.coordinates.distances.CartesianPoints`
        """
        return self._xend

    @xend.setter
    def xend(self, x):
        if isinstance(x, CartesianPoints):
            self._xend = x
        elif isinstance(x, SphericalCoordinatesBase):
            self._xend = x.cartesian
        elif hasattr(x, 'unit'):
            self._xend = CartesianPoints(*x, unit=x.unit)
        else:
            self._xend = CartesianPoints(*x, unit=Unit('m'))

    @property
    def yend(self):
        """Position of the y-end mirror for this `LaserInterferometer`.

        This should be the position of the x-end optic, relative to
        the Earth centre, in earth-fixed coordinates.

        :type: :class:`~astropy.coordinates.distances.CartesianPoints`
        """
        return self._yend

    @yend.setter
    def yend(self, y):
        if isinstance(x, CartesianPoints):
            self._xend = x
        elif isinstance(x, SphericalCoordinatesBase):
            self._xend = x.cartesian
        elif hasattr(x, 'unit'):
            self._xend = CartesianPoints(*x, unit=x.unit)
        else:
            self._xend = CartesianPoints(*x, unit=Unit('m'))

    @property
    def response_matrix(self):
        """The 3x3 Cartesian detector response tensor.

        :type: :class:`numpy.ndarray`
        """
        return self._response_matrix

    @response_matrix.setter
    def response_matrix(self, r):
        self._response_matrix = r

    @property
    def _lal(self):
        """The LAL representation of this detector.

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

    # ------------------------------------------------------------------------
    # LaserInterferometer methods

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
        from lal import (ComputeDetAMResponse, GreenwichMeanSiderealTime)
        return ComputeDetAMResponse(
                   self.response_matrix, source.ra.radians, source.dec.radians,
                   polarization, GreenwichMeanSiderealTime(source.obstime.gps))

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
