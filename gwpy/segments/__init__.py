# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module provides classes for generating and manipulating
data segments of the form [gps_start, gps_end).

The core of this module is adapted from the `Grid LIGO User Environment
(GLUE) package <http://glue.org>`.
"""

from .. import version

__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__version__ = version.version

from .core import (Segment, SegmentList, SegmentListDict)
from .flags import DataQualityFlag

from ..io import segwizard
from ..io.ligolw import segments as ligolw_segments
