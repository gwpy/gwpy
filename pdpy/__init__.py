# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
#
# This file is part of pyDischarge.
#
# pyDischarge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyDischarge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyDischarge.  If not, see <http://www.gnu.org/licenses/>.

"""A python package for gravitational-wave astrophysics

pyDischarge is a collaboration-driven `Python <http://www.python.org>`_ package
providing tools for studying data from ground-based gravitational-wave
detectors.

pyDischarge provides a user-friendly, intuitive interface to the common time-domain
and frequency-domain data produced by the `LIGO <http://www.ligo.org>`_ and
`Virgo <http://www.ego-gw.it>`_ instruments and their analysis,
with easy-to-follow tutorials at each step.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "The LIGO Scientific Collaboration and the Virgo Collaboration"

from . import (
    plot,  # registers pydischarge.plot.Axes as default rectilinear axes
)

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = ''
