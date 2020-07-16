# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""Unit tests for :mod:`gwpy.detector.units`
"""

import pytest

from astropy import units

from ..units import parse_unit

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@pytest.mark.parametrize('arg, unit', [
    (None, None),
    (units.m, units.m),
    ('meter', units.m),
    ('Volts', units.V),
    ('Meters/Second', units.m / units.s),
    ('Amp', units.ampere),
    ('MPC', units.megaparsec),
    ('degrees_C', units.Unit('Celsius')),
    ('DegC', units.Unit('Celsius')),
    ('degrees_F', units.Unit('Fahrenheit')),
])
def test_parse_unit(arg, unit):
    assert parse_unit(arg, parse_strict='silent') == unit


def test_parse_unit_strict():
    # check that errors get raise appropriately
    with pytest.raises(ValueError) as exc:
        parse_unit('metre', parse_strict='raise')

    # check that warnings get posted, and a custom NamedUnit gets returned
    with pytest.warns(units.UnitsWarning) as exc:
        u = parse_unit('metre', parse_strict='warn')
    assert str(exc[0].message) == ('metre is not a valid unit. Did you mean '
                                   'meter? Mathematical operations using this '
                                   'unit should work, but conversions to '
                                   'other units will not.')
    assert isinstance(u, units.IrreducibleUnit)
    assert str(u) == 'metre'

    # assert that a newly-created unit only gets created once
    u2 = parse_unit('metre', parse_strict='silent')
    assert u2 is u  # same object
    assert u == u2  # compare as equal (just in case)


@pytest.mark.parametrize('name', [
    'NONE',
    'undef',
    'strain',
    'coherence',
    'sec',
    'torr',
    'cf',
    'cfm',
    'ptcls',
])
def test_detector_units(name):
    # just check that such a unit exists and doesn't evaluate to False
    assert units.Unit(name)
