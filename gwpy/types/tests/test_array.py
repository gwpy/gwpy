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

"""Unit test for gwpy.types classes
"""

import pickle
import warnings
from unittest import mock

import pytest

import numpy

from astropy import units
from astropy.time import Time

from ...detector import Channel
from ...testing import utils
from ...time import LIGOTimeGPS
from .. import Array

warnings.filterwarnings('always', category=units.UnitsWarning)
warnings.filterwarnings('always', category=UserWarning)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

SEED = 1
GPS_EPOCH = 12345
TIME_EPOCH = Time(12345, format='gps', scale='utc')
CHANNEL_NAME = 'G1:TEST-CHANNEL'
CHANNEL = Channel(CHANNEL_NAME)


class TestArray(object):
    """Test `gwpy.types.Array`
    """
    TEST_CLASS = Array
    DTYPE = None

    # -- setup ----------------------------------

    @classmethod
    def setup_class(cls):
        numpy.random.seed(SEED)
        cls.data = (numpy.random.random(100) * 1e5).astype(dtype=cls.DTYPE)

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs.setdefault('copy', False)
        return cls.TEST_CLASS(cls.data, *args, **kwargs)

    @classmethod
    @pytest.fixture()
    def array(cls):
        return cls.create()

    @property
    def TEST_ARRAY(self):
        try:
            return self._TEST_ARRAY
        except AttributeError:
            # create array
            self._TEST_ARRAY = self.create(name=CHANNEL_NAME, unit='meter',
                                           channel=CHANNEL_NAME,
                                           epoch=GPS_EPOCH)
            # customise channel a wee bit
            #    used to test pickle/unpickle when storing channel as
            #    dataset attr in HDF5
            self._TEST_ARRAY.channel.sample_rate = 128
            self._TEST_ARRAY.channel.unit = 'm'
            return self.TEST_ARRAY

    # -- test basic construction ----------------

    def test_new(self):
        """Test Array creation
        """
        # test basic empty contructor
        with pytest.raises(TypeError):
            self.TEST_CLASS()

        # test with some data
        array = self.create()
        utils.assert_array_equal(array.value, self.data)

        # test that copy=True ensures owndata
        assert self.create(copy=False).flags.owndata is False
        assert self.create(copy=True).flags.owndata is True

        # return array for subclasses to use
        return array

    def test_unit(self, array):
        # test default unit is dimensionless
        assert array.unit is units.dimensionless_unscaled

        # test deleter and recovery
        del array.unit
        del array.unit  # test twice to make sure AttributeError isn't raised
        assert array.unit is None

        # test unit gets passed properly
        array = self.create(unit='m')
        assert array.unit is units.m

        # test unrecognised units
        with mock.patch.dict(
                'gwpy.detector.units.UNRECOGNIZED_UNITS', clear=True), \
                pytest.warns(units.UnitsWarning):
            array = self.create(unit='blah')
        assert isinstance(array.unit, units.IrreducibleUnit)
        assert str(array.unit) == 'blah'

        # test setting unit doesn't work
        with pytest.raises(AttributeError):
            array.unit = 'm'
        del array.unit
        array.unit = 'm'
        assert array.unit is units.m

    def test_name(self, array):
        # test default is no name
        assert array.name is None

        # test deleter and recovery
        del array.name
        del array.name
        assert array.name is None

        # test simple name
        array = self.create(name='TEST CASE')
        assert array.name == 'TEST CASE'

        # test None gets preserved
        array.name = None
        assert array.name is None

        # but everything else gets str()
        array.name = 4
        assert array.name == '4'

    def test_epoch(self, array):
        # test default is no epoch
        assert array.epoch is None

        # test deleter and recovery
        del array.epoch
        del array.epoch
        assert array.epoch is None

        # test epoch gets parsed properly
        array = self.create(epoch=GPS_EPOCH)
        assert isinstance(array.epoch, Time)
        assert array.epoch == TIME_EPOCH

        # test epoch in different formats
        array = self.create(epoch=LIGOTimeGPS(GPS_EPOCH))
        assert array.epoch == TIME_EPOCH

        # test precision at high GPS times (to millisecond)
        gps = LIGOTimeGPS(1234567890, 123456000)
        array = self.create(epoch=gps)
        assert array.epoch.gps == float(gps)

        # test None gets preserved
        array.epoch = None
        assert array.epoch is None

    def test_channel(self, array):
        # test default channl is None
        assert array.channel is None

        # test deleter and recovery
        del array.channel
        del array.channel
        assert array.channel is None

        # test simple channel
        array = self.create(channel=CHANNEL_NAME)
        assert isinstance(array.channel, Channel)
        assert array.channel == CHANNEL

        # test existing channel doesn't get modified
        array = self.create(channel=CHANNEL)
        assert array.channel is CHANNEL

        # test preserves None
        array.channel = None
        assert array.channel is None

    def test_math(self, array):
        array.override_unit('Hz')
        # test basic operations
        arraysq = array ** 2
        utils.assert_array_equal(arraysq.value, self.data ** 2)
        assert arraysq.unit == units.Hz ** 2
        assert arraysq.name == array.name
        assert arraysq.epoch == array.epoch
        assert arraysq.channel == array.channel

    def test_copy(self):
        array = self.create(channel='X1:TEST')
        copy = array.copy()
        utils.assert_quantity_sub_equal(array, copy)
        assert copy.channel is not array.channel

    def test_repr(self, array):
        # just test that it runs
        repr(array)

    def test_str(self, array):
        # just test that it runs
        str(array)

    def test_pickle(self, array):
        # check pickle-unpickle yields unchanged data
        pkl = array.dumps()
        a2 = pickle.loads(pkl)
        utils.assert_quantity_sub_equal(array, a2)

    # -- test methods ---------------------------

    def test_tostring(self, array):
        assert array.tostring() == array.value.tobytes()

    def test_abs(self, array):
        utils.assert_quantity_equal(array.abs(), numpy.abs(array))

    def test_median(self, array):
        utils.assert_quantity_equal(
            array.median(), numpy.median(array.value) * array.unit)

    def test_override_unit(self, array):
        assert array.unit is units.dimensionless_unscaled

        # check basic override works
        array.override_unit('m')
        assert array.unit is units.meter

        # check parse_strict works for each of 'raise' (default), 'warn',
        # and 'silent'
        with mock.patch.dict(
                'gwpy.detector.units.UNRECOGNIZED_UNITS', clear=True):
            with pytest.raises(ValueError):
                array.override_unit('blah', parse_strict='raise')
            with pytest.warns(units.UnitsWarning):
                array.override_unit('blah', parse_strict='warn')
            array.override_unit('blah', parse_strict='silent')
        assert isinstance(array.unit, units.IrreducibleUnit)
        assert str(array.unit) == 'blah'

    def test_flatten(self, array):
        flat = array.flatten()
        assert flat.ndim == 1
        assert type(flat) is units.Quantity  # pylint: disable=C0123
        assert flat.shape[0] == numpy.prod(array.shape)
        try:
            utils.assert_quantity_equal(
                array.flatten('C'),
                array.T.flatten('F'),
            )
        except NotImplementedError:
            pass
