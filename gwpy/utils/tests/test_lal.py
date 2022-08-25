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
        return
    utils_lal.find_typed_function(
        'REAL4', 'FrStreamRead', 'TimeSeries',
        module=lalframe) is lalframe.FrStreamReadREAL4TimeSeries


@pytest.mark.parametrize(
    'dtype',
    [
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint16,
        numpy.uint32,
        numpy.uint64,
        numpy.float32,
        numpy.float64,
        numpy.complex64,
        numpy.complex128,
    ]
)
def test_from_lal_type(dtype):
    func = utils_lal.find_typed_function(dtype, 'Create', 'TimeSeries')
    lalts = func(None, None, 0, 1., None, 1)
    assert utils_lal.from_lal_type(lalts) is dtype
    assert utils_lal.from_lal_type(type(lalts)) is dtype


@pytest.mark.parametrize('name', [
    'XXX',
    'XXXCOMPLEX64ZZZ',
    'INNT2',
    'INT5',
    'REAL42',
])
def test_from_lal_type_errors(name):
    lal_type = type(name, (), {})
    with pytest.raises(ValueError, match='no known numpy'):
        utils_lal.from_lal_type(lal_type)


@pytest.mark.parametrize(("in_", "out", "scale"), (
    ("m", "m", 1),
    ("Farad", "m^-2 kg^-1 s^4 A^2", 1),
    ("m**(1/2)", "m^1/2", 1),
    ("km", "10^3 m", 1),
    ("123 m", "m", 123),
))
def test_to_lal_unit(in_, out, scale):
    assert utils_lal.to_lal_unit(in_) == (lal.Unit(out), scale)


def test_to_lal_unit_error():
    with pytest.raises(ValueError) as exc:
        utils_lal.to_lal_unit('rad/s')
    assert str(exc.value) == "LAL has no unit corresponding to 'rad'"


@pytest.mark.parametrize(("in_", "out"), (
    ("m s^-1", "m/s"),
    ("strain", "strain"),
    ("m^1/2", "m**(1/2.)"),
    ("10^3 m", "km"),
))
def test_from_lal_unit(in_, out):
    assert utils_lal.from_lal_unit(lal.Unit(in_)) == units.Unit(out)


def test_to_lal_ligotimegps():
    assert utils_lal.to_lal_ligotimegps(123.456) == (
        lal.LIGOTimeGPS(123, 456000000))
