# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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

"""Utilties for the GWpy test suite
"""

from importlib import import_module

from numpy.testing import (assert_array_equal, assert_array_almost_equal)

import pytest


# -- dependencies -------------------------------------------------------------

def has(module):
    """Test whether a module is available

    Returns `True` if `import module` succeeded, otherwise `False`
    """
    try:
        import_module(module)
    except ImportError:
        return False
    else:
        return True


def skip_missing_dependency(module):
    """Returns a mark generator to skip a test if the dependency is missing
    """
    return pytest.mark.skipif(not has(module),
                              reason='No module named %s' % module)

# -- assertions ---------------------------------------------------------------

def assert_quantity_equal(q1, q2):
    """Assert that two `~astropy.units.Quantity` objects are the same
    """
    _assert_quantity(q1, q2, array_assertion=assert_array_equal)


def assert_quantity_almost_equal(q1, q2):
    """Assert that two `~astropy.units.Quantity` objects are almost the same

    This method asserts that the units are the same and that the values are
    equal within precision.
    """
    _assert_quantity(q1, q2, array_assertion=assert_array_almost_equal)


def _assert_quantity(q1, q2, array_assertion=assert_array_equal):
    assert q1.unit == q2.unit
    array_assertion(q1.value, q2.value)
