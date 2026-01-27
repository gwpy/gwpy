# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Unit test for timeseries module."""

import os
from io import BytesIO

import numpy
import pytest
from astropy import units
from matplotlib import rc_context

from ...detector import Channel
from ...segments import Segment
from ...testing import mocks, utils
from ...testing.errors import pytest_skip_flaky_network
from ...time import LIGOTimeGPS, Time
from ...types import Array2D
from .. import (
    Bits,
    StateTimeSeries,
    StateTimeSeriesDict,
    StateVector,
    StateVectorDict,
    StateVectorList,
)
from .test_core import (
    TestTimeSeriesBase as _TestTimeSeriesBase,
    TestTimeSeriesBaseDict as _TestTimeSeriesBaseDict,
    TestTimeSeriesBaseList as _TestTimeSeriesBaseList,
)
from .test_timeseries import (
    GWOSC_GW150914_DQ_BITS,
    GWOSC_GW150914_DQ_NAME,
    # Description of GW150914
    GWOSC_GW150914_IFO,
    GWOSC_GW150914_INJ_BITS,
    GWOSC_GW150914_INJ_NAME,
    GWOSC_GW150914_SEGMENT,
    GWOSC_GW190814_DQ_BITS,
    GWOSC_GW190814_DQ_NAME,
    # Description of GW190814
    GWOSC_GW190814_IFO,
    GWOSC_GW190814_INJ_BITS,
    GWOSC_GW190814_INJ_NAME,
    GWOSC_GW190814_SEGMENT,
)

# This is used to test reading of injections
GWOSC_GW150914_INJ_KWARGS = {
    "hdf5": {
        "path": "quality/injections",
        "value_dataset": "Injmask",
        "bits_dataset": "InjDescriptions",
    },
    "gwf": {
        "channel": GWOSC_GW150914_INJ_NAME["gwf"],
    },
}

GWOSC_GW190814_INJ_KWARGS = {
    "hdf5": {
        "path": "quality/injections",
        "value_dataset": "Injmask",
        "bits_dataset": "InjDescriptions",
    },
    "gwf": {
        "channel": GWOSC_GW190814_INJ_NAME["gwf"],
    },
}

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- StateTimeSeries ----------------------------------------------------------

class TestStateTimeSeries(_TestTimeSeriesBase):
    TEST_CLASS = StateTimeSeries

    @classmethod
    def setup_class(cls):
        cls.data = numpy.asarray(
            [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
            dtype=bool,
        )

    def test_new(self):
        sts = self.create()
        assert isinstance(sts, self.TEST_CLASS)
        assert sts.dtype is numpy.dtype("bool")

    def test_getitem(self, array):
        assert isinstance(array[0], numpy.bool_)

    def test_unit(self, array):
        assert array.unit is units.dimensionless_unscaled

        # check that we can't delete the unit
        with pytest.raises(AttributeError):
            del array.unit

        # check that we can't set the unit
        with pytest.raises(TypeError):
            self.create(unit="test")

    def test_math(self, array):
        # test that math operations give back booleans
        a2 = array ** 2
        assert a2.dtype is numpy.dtype(bool)
        utils.assert_array_equal(array.value, a2.value)

    @pytest.mark.requires("arrakis")
    @pytest.mark.parametrize("copy", [False, True])
    @pytest.mark.parametrize("dtype", [
        pytest.param(bool, id="bool"),
        pytest.param(numpy.dtype('bool'), id="np_bool"),
    ])
    def test_from_arrakis(self, copy, dtype):
        """Test `StateTimeSeries.from_arrakis`."""
        super().test_from_arrakis(copy, dtype)

    @pytest.mark.requires("nds2")
    def test_from_nds2_buffer(self):
        # build fake buffer
        nds_buffer = mocks.nds2_buffer(
            "X1:TEST",
            self.data,
            1000000000,
            self.data.shape[0],
            "m",
            name="test",
        )
        array = self.TEST_CLASS.from_nds2_buffer(nds_buffer)
        assert array.unit is units.dimensionless_unscaled
        assert array.dtype is numpy.dtype(bool)

    def test_to_dqflag(self, array):
        flag = array.to_dqflag()
        utils.assert_segmentlist_equal(
            flag.active,
            map(Segment, ((1, 4), (7, 8), (10, 13), (14, 15), (16, 20))),
        )
        utils.assert_segmentlist_equal(flag.known, [array.span])
        assert flag.name == array.name
        assert flag.label == array.name
        assert flag.description is None

        flag = array.to_dqflag(
            minlen=2,
            dtype=LIGOTimeGPS,
            name="Test flag",
            round=True,
            label="Label",
            description="Description",
        )
        utils.assert_segmentlist_equal(
            flag.active,
            map(Segment, ((1, 4), (10, 13), (16, 20))),
        )
        assert isinstance(flag.active[0][0], LIGOTimeGPS)
        assert flag.name == "Test flag"
        assert flag.label == "Label"
        assert flag.description == "Description"

    def test_override_unit(self, array):
        with pytest.raises(NotImplementedError):
            super().test_override_unit(array)

    def test_is_compatible_error_unit(self, array):
        pytest.skip(f"not implemented for {self.TEST_CLASS.__name__}")

    def test_to_from_pycbc(self):
        pytest.skip(f"not implemented for {self.TEST_CLASS.__name__}")

    def test_to_from_lal(self, array):
        with pytest.raises(NotImplementedError):
            super().test_to_from_lal(array)

    @pytest.mark.parametrize("copy", (False, True))
    def test_to_from_lal_no_copy(self, array, copy):
        with pytest.raises(NotImplementedError):
            super().test_to_from_lal_no_copy(array, copy)

    @pytest.mark.requires("lal")
    def test_to_from_lal_pow10_units(self, array):
        with pytest.raises(NotImplementedError):
            super().test_to_from_lal_pow10_units(array)

    @pytest.mark.requires("lal")
    def test_to_from_lal_scaled_units(self, array):
        with pytest.raises(NotImplementedError):
            super().test_to_from_lal_scaled_units(array)

    @pytest.mark.requires("lal")
    def test_to_from_lal_unrecognised_units(self, array):
        with pytest.raises(NotImplementedError):
            super().test_to_from_lal_unrecognised_units(array)


# -- StateTimeSeriesDict ------------------------------------------------------

class TestStateTimeSeriesDict(_TestTimeSeriesBaseDict):
    TEST_CLASS = StateTimeSeriesDict
    ENTRY_CLASS = StateTimeSeries
    DTYPE = bool

    def test_resample(self, instance):
        pytest.skip(f"not implemented for {type(instance).__name__}")


# -- Bits ---------------------------------------------------------------------

class TestBits:
    TEST_CLASS = Bits

    @pytest.mark.parametrize(("in_", "out"), [
        # list
        pytest.param(
            ["bit 0", "bit 1", "bit 2", None, "bit 3", ""],
            ["bit 0", "bit 1", "bit 2", None, "bit 3", None],
            id="list",
        ),
        # dict
        pytest.param(
            {1: "bit 1", 4: "bit 4", "6": "bit 6"},
            [None, "bit 1", None, None, "bit 4", None, "bit 6"],
            id="dict",
        ),
    ])
    def test_init(self, in_, out):
        """Test initialising `Bits`."""
        bits = self.TEST_CLASS(in_)
        assert bits == out
        assert bits.channel is None
        assert bits.epoch is None
        assert bits.description == {bit: None for bit in bits if bit is not None}

        bits = self.TEST_CLASS(in_, channel="L1:Test", epoch=0)
        assert bits.epoch == Time(0, format="gps")
        assert bits.channel == Channel("L1:Test")

    def test_str(self):
        """Test ``str(Bits)``."""
        bits = self.TEST_CLASS(["a", "b", None, "c"])
        assert str(bits) == os.linesep.join([
            "Bits(0: a",
            "     1: b",
            "     3: c,",
            "     channel=None,",
            "     epoch=None)",
        ])

    def test_repr(self):
        """Test ``repr(Bits)``."""
        bits = self.TEST_CLASS(["a", "b", None, "c"])
        assert repr(bits) == os.linesep.join([
            "<Bits(0: 'a'",
            "      1: 'b'",
            "      3: 'c',",
            "      channel=None,",
            "      epoch=None)>",
        ])

    def test_array(self):
        """Test converting `Bits` to an array."""
        bits = self.TEST_CLASS(["a", "b", None, "c"])
        utils.assert_array_equal(
            numpy.asarray(bits),
            ["a", "b", "", "c"],
        )


# -- StateVector---------------------------------------------------------------

class TestStateVector(_TestTimeSeriesBase):
    TEST_CLASS = StateVector
    DTYPE = "uint32"

    @classmethod
    def setup_class(cls):
        numpy.random.seed(0)
        cls.data = numpy.random.randint(
            2**4+1,
            size=100,
        ).astype(cls.DTYPE, copy=False)

    def test_bits(self, array):
        assert isinstance(array.bits, Bits)
        assert array.bits == [f"Bit {i}" for i in range(32)]

        bits = [f"Bit {i}" for i in range(4)]

        sv = self.create(bits=bits)
        assert isinstance(sv.bits, Bits)
        assert sv.bits.channel is sv.channel
        assert sv.bits.epoch == sv.epoch
        assert sv.bits == bits

        del sv.bits
        del sv.bits
        assert isinstance(sv.bits, Bits)
        assert sv.bits == [f"Bit {i}" for i in range(32)]

        sv = self.create(dtype="uint16")
        assert sv.bits == [f"Bit {i}" for i in range(16)]

    def test_boolean(self, array):
        b = array.boolean
        assert isinstance(b, Array2D)
        assert b.shape == (array.size, len(array.bits))
        # array[0] == 12, check boolean equivalent
        utils.assert_array_equal(b[0], [12 >> j & 1 for j in range(32)])

    def test_get_bit_series(self, array):
        # test default
        bs = array.get_bit_series()
        assert isinstance(bs, StateTimeSeriesDict)
        assert list(bs.keys()) == array.bits
        # check that correct number of samples match (simple test)
        assert bs["Bit 2"].sum() == 43

        # check that bits in gives bits out
        bs = array.get_bit_series(["Bit 0", "Bit 3"])
        assert list(bs.keys()) == ["Bit 0", "Bit 3"]
        assert [v.sum() for v in bs.values()] == [50, 41]

        # check that invalid bits throws exception
        with pytest.raises(
            ValueError,
            match=r"^Bit 'blah' not found in StateVector$",
        ):
            array.get_bit_series(["blah"])

    def test_plot(self, array):
        with rc_context(rc={"text.usetex": False}):
            plot = array.plot()
            # make sure there were no lines drawn
            assert len(plot.gca().lines) == 0
            # assert one collection for each of known and active segmentlists
            assert len(plot.gca().collections) == len(array.bits) * 2
            plot.save(BytesIO(), format="png")
            plot.close()

            # test timeseries plotting as normal
            plot = array.plot(format="timeseries")
            line = plot.gca().lines[0]
            utils.assert_array_equal(line.get_xdata(), array.xindex.value)
            utils.assert_array_equal(line.get_ydata(), array.value)
            plot.close()

    def test_resample(self, array):
        # check downsampling by factor of 2
        a2 = array.resample(array.sample_rate / 2.)
        assert a2.sample_rate == array.sample_rate / 2.
        assert a2.bits is array.bits
        utils.assert_array_equal(
            a2.value[:10],
            [12, 0, 3, 0, 4, 0, 6, 5, 8, 0],
        )

        # check upsampling raises NotImplementedError
        with pytest.raises(NotImplementedError):
            array.resample(array.sample_rate * 2.)

        # check resampling by non-integer factor raises error
        with pytest.raises(ValueError):
            array.resample(array.sample_rate * .75)
        with pytest.raises(ValueError):
            array.resample(array.sample_rate * 1.5)

    def test_to_from_lal_scaled_units(self):
        pytest.skip(
            f"not implemented for {self.TEST_CLASS.__name__}",
        )

    # -- data access ----------------------------

    @pytest.mark.parametrize("format", [
        "hdf5",
        pytest.param(  # only frameCPP actually reads units properly
            "gwf",
            marks=pytest.mark.requires("LDAStools.frameCPP"),
        ),
    ])
    @pytest_skip_flaky_network
    def test_fetch_open_data(self, format):
        sv = self.TEST_CLASS.fetch_open_data(
            GWOSC_GW150914_IFO,
            GWOSC_GW150914_SEGMENT[0],
            GWOSC_GW150914_SEGMENT[1],
            format=format,
            version=3,
        )
        ref = StateVector(
            [127, 127, 127, 127],
            unit="",
            t0=GWOSC_GW150914_SEGMENT[0],
            dt=1,
            name=GWOSC_GW150914_DQ_NAME[format],
            bits=GWOSC_GW150914_DQ_BITS[format],
        )
        utils.assert_quantity_sub_equal(sv, ref, exclude=["channel", "bits"])
        assert sorted(map(str, sv.bits)) == sorted(map(str, ref.bits))

    @pytest.mark.parametrize("format", [
        "hdf5",
        pytest.param(  # only frameCPP actually reads units properly
            "gwf",
            marks=pytest.mark.requires("LDAStools.frameCPP"),
        ),
    ])
    @pytest_skip_flaky_network
    def test_fetch_open_data_new_format(self, format):
        sv = self.TEST_CLASS.fetch_open_data(
            GWOSC_GW190814_IFO,
            GWOSC_GW190814_SEGMENT[0],
            GWOSC_GW190814_SEGMENT[1],
            format=format,
            version=3,
        )
        ref = StateVector(
            [127, 127, 127, 127],
            unit="",
            t0=GWOSC_GW190814_SEGMENT[0],
            dt=1,
            name=GWOSC_GW190814_DQ_NAME[format],
            bits=GWOSC_GW190814_DQ_BITS[format],
        )
        utils.assert_quantity_sub_equal(sv, ref, exclude=["channel", "bits"])
        assert sorted(map(str, sv.bits)) == sorted(map(str, ref.bits))


    @pytest.mark.parametrize("format", [
        "hdf5",
        pytest.param(  # only frameCPP actually reads units properly
            "gwf",
            marks=pytest.mark.requires("LDAStools.frameCPP"),
        ),
    ])
    @pytest_skip_flaky_network
    def test_fetch_open_data_injection(self, format):
        """This test checks that we are able to read and fetch the injection bits
        from GWOSC files (see #1811).

        To do so, we download the injection data from GWOSC and compare it to a
        known StateVector.
        """
        inj = self.TEST_CLASS.fetch_open_data(
            GWOSC_GW150914_IFO,
            GWOSC_GW150914_SEGMENT[0],
            GWOSC_GW150914_SEGMENT[1],
            format=format,
            version=3,
            **GWOSC_GW150914_INJ_KWARGS[format],
        )
        # We know that the first bytes are equal to 23
        ref = StateVector(
            [23, 23, 23, 23],
            unit="",
            t0=GWOSC_GW150914_SEGMENT[0],
            dt=1,
            name=GWOSC_GW150914_INJ_NAME[format],
            bits=GWOSC_GW150914_INJ_BITS[format],
        )
        utils.assert_quantity_sub_equal(inj, ref, exclude=["channel", "bits"])
        assert sorted(map(str, inj.bits)) == sorted(map(str, ref.bits))

    @pytest.mark.parametrize("format", [
        "hdf5",
        pytest.param(  # only frameCPP actually reads units properly
            "gwf",
            marks=pytest.mark.requires("LDAStools.frameCPP"),
        ),
    ])
    @pytest_skip_flaky_network
    def test_fetch_open_data_injection_new_format(self, format):
        """Check that we can read and fetch GWOSC injection bits in the new format.

        To do so, we download the injection data from GWOSC and compare it to a
        known StateVector.

        See (https://gitlab.com/gwpy/gwpy/-/issues/1811).
        """
        inj = self.TEST_CLASS.fetch_open_data(
            GWOSC_GW190814_IFO,
            GWOSC_GW190814_SEGMENT[0],
            GWOSC_GW190814_SEGMENT[1],
            format=format,
            version=3,
            **GWOSC_GW190814_INJ_KWARGS[format],
        )
        # We know that the first bytes are equal to 23
        ref = StateVector(
            [23, 23, 23, 23],
            unit="",
            t0=GWOSC_GW190814_SEGMENT[0],
            dt=1,
            name=GWOSC_GW190814_INJ_NAME[format],
            bits=GWOSC_GW190814_INJ_BITS[format],
        )
        utils.assert_quantity_sub_equal(inj, ref, exclude=["channel", "bits"])
        assert sorted(map(str, inj.bits)) == sorted(map(str, ref.bits))


    @pytest.mark.requires("nds2")
    def test_from_nds2_buffer(self):
        # build fake buffer
        nds_buffer = mocks.nds2_buffer(
            "X1:TEST",
            self.data,
            1000000000,
            self.data.shape[0],
            "",
            name="test",
        )
        array = self.TEST_CLASS.from_nds2_buffer(nds_buffer)
        assert array.unit is units.dimensionless_unscaled

    @pytest.mark.parametrize("ext", ("hdf5", "h5"))
    def test_read_write_hdf5(self, tmp_path, array, ext):
        array.bits = ["a", "b", "c", "d"]
        array.name = "test"

        tmp = tmp_path / f"test.{ext}"

        # write array (with auto-identify)
        array.write(tmp, overwrite=True)

        # check reading gives the same data (with/without auto-identify)
        new = type(array).read(tmp, format="hdf5")
        utils.assert_quantity_sub_equal(array, new)
        new = type(array).read(tmp)
        utils.assert_quantity_sub_equal(array, new)


# -- StateVectorDict ----------------------------------------------------------

class TestStateVectorDict(_TestTimeSeriesBaseDict):
    TEST_CLASS = StateVectorDict
    ENTRY_CLASS = StateVector
    DTYPE = "uint32"


# -- StateVectorList ----------------------------------------------------------

class TestStateVectorList(_TestTimeSeriesBaseList):
    TEST_CLASS = StateVectorList
    ENTRY_CLASS = StateVector
    DTYPE = "uint32"
