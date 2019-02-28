# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

"""Unit test for utils module
"""

import pytest

import numpy

from astropy import units

# import necessary modules
lal = pytest.importorskip('lal')
utils_lal = pytest.importorskip('gwpy.utils.lal')

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def test_to_lal_type_str():
    assert utils_lal.to_lal_type_str(float) == 'REAL8'
    assert utils_lal.to_lal_type_str(
        numpy.dtype('float64')) == 'REAL8'
    assert utils_lal.to_lal_type_str(11) == 'REAL8'
    with pytest.raises(ValueError):
        utils_lal.to_lal_type_str('blah')
    with pytest.raises(ValueError):
        utils_lal.to_lal_type_str(numpy.int8)
    with pytest.raises(ValueError):
        utils_lal.to_lal_type_str(20)


def test_find_typed_function():
    assert utils_lal.find_typed_function(
        'REAL8', 'Create', 'Sequence') is lal.CreateREAL8Sequence

    try:
        import lalframe
    except ImportError:  # no lalframe
        pass
    else:
        utils_lal.find_typed_function(
            'REAL4', 'FrStreamRead', 'TimeSeries',
            module=lalframe) is lalframe.FrStreamReadREAL4TimeSeries


def test_to_lal_unit():
    assert utils_lal.to_lal_unit('m') == lal.MeterUnit
    assert utils_lal.to_lal_unit('Farad') == lal.Unit(
        'm^-2 kg^-1 s^4 A^2')
    with pytest.raises(ValueError):
        utils_lal.to_lal_unit('rad/s')


def test_from_lal_unit():
    try:
        lalms = lal.MeterUnit / lal.SecondUnit
    except TypeError as exc:
        # see https://git.ligo.org/lscsoft/lalsuite/issues/65
        pytest.skip(str(exc))
    assert utils_lal.from_lal_unit(lalms) == units.Unit('m/s')
    assert utils_lal.from_lal_unit(lal.StrainUnit) == (
        units.Unit('strain'))


def test_to_lal_ligotimegps():
    assert utils_lal.to_lal_ligotimegps(123.456) == (
        lal.LIGOTimeGPS(123, 456000000))
