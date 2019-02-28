# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2019)
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

"""Test setup for gwpy
"""

import warnings

import numpy

from matplotlib import (use, rcParams)

# register custom fixtures for all test modules
from .testing.fixtures import *  # noqa: F401,F403

# set random seed to 1 for reproducability
numpy.random.seed(1)

# -- plotting options

# ignore errors due from pyplot.show() using Agg
warnings.filterwarnings('ignore', message=".*non-GUI backend.*")

# force Agg for all tests
use('agg', warn=False, force=True)

# force simpler rcParams for all tests
# (fixtures or tests may update these individually)
# NOTE: this most-likely happens _after_ gwpy.plot has
#       updated the rcParams once, so these settings should persist
rcParams.update({
    'text.usetex': False,  # TeX is slow most of the time
})
