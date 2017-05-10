# -*- coding: utf-8 -*-
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

"""Compatibility module to import unittest
"""

import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock

try:
    import h5py
except ImportError:
    HAS_H5PY = False
else:
    HAS_H5PY = True

try:
    import lal
except ImportError:
    HAS_LAL = False
else:
    HAS_LAL = True

try:
    import dqsegdb
except ImportError:
    HAS_DQSEGDB = False
else:
    HAS_DQSEGDB = True

try:
    import m2crypto
except ImportError:
    HAS_M2CRYPTO = False
else:
    HAS_M2CRYPTO = True

try:
    import nds2
except ImportError:
    HAS_NDS2 = False
else:
    HAS_NDS2 = True
