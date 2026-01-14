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

"""Unit test for frequencyseries module."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy
import pytest
from astropy import units
from numpy import shares_memory
from numpy.testing import assert_allclose
from scipy import signal

from ...signal import filter_design
from ...testing import utils
from ...timeseries import TimeSeries
from ...types.tests.test_series import TestSeries as _TestSeries
from .. import FrequencySeries

if TYPE_CHECKING:
    from pathlib import Path

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

LIGO_LW_ARRAY = r"""<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt">
<LIGO_LW>
  <LIGO_LW Name="REAL8FrequencySeries">
    <Time Type="GPS" Name="epoch">1000000000</Time>
    <Param Type="lstring" Name="channel:param">X1:TEST-CHANNEL_1</Param>
    <Array Type="real_8" Name="PSD1:array" Unit="Hz^-1">
      <Dim Start="10" Scale="1" Name="Frequency" Unit="Hz">5</Dim>
      <Dim Name="Frequency,Real">2</Dim>
      <Stream Delimiter=" " Type="Local">
        0 1
        1 2
        2 3
        3 4
        4 5
      </Stream>
    </Array>
  </LIGO_LW>
  <LIGO_LW Name="REAL8FrequencySeries">
    <Param Type="lstring" Name="channel:param">X1:TEST-CHANNEL_2</Param>
    <Param Type="real_8" Name="f0:param" Unit="s^-1">0</Param>
    <Array Type="real_8" Name="PSD2:array" Unit="s m^2">
      <Dim Start="0" Scale="1" Name="Frequency" Unit="s^-1">5</Dim>
      <Dim Name="Real">1</Dim>
      <Stream Delimiter=" " Type="Local">
        10
        20
        30
        40
        50
      </Stream>
    </Array>
  </LIGO_LW>
  <LIGO_LW Name="REAL8FrequencySeries">
    <Time Type="GPS" Name="epoch">1000000001</Time>
    <Array Type="real_8" Name="PSD2:array" Unit="s m^2">
      <Dim Start="0" Scale="1" Name="Frequency" Unit="s^-1">5</Dim>
      <Dim Name="Frequency,Real">3</Dim>
      <Stream Delimiter=" " Type="Local">
        0 10 1
        1 20 2
        2 30 3
        3 40 4
        4 50 5
      </Stream>
    </Array>
  </LIGO_LW>
</LIGO_LW>
"""


class TestFrequencySeries(_TestSeries):
    """Tests for `FrequencySeries`."""

    TEST_CLASS = FrequencySeries

    # -- test properties -------------

    def test_f0(self, array):
        """Test `FrequencySeries.f0`."""
        assert array.f0 is array.x0
        array.f0 = 4
        assert array.f0 == 4 * units.Hz

    def test_df(self, array):
        """Test `FrequencySeries.df`."""
        assert array.df is array.dx
        array.df = 4
        assert array.df == 4 * units.Hz

    def test_frequencies(self, array):
        """Test `FrequencySeries.frequencies`."""
        assert array.frequencies is array.xindex
        utils.assert_quantity_equal(
            array.frequencies, numpy.arange(array.size) * array.df + array.f0)

    # -- test methods ----------------

    @pytest.mark.usefixtures("usetex")
    def test_plot(self, array):
        """Test `FrequencySeries.plot`."""
        plot = array.plot()
        line = plot.gca().lines[0]
        utils.assert_array_equal(line.get_xdata(), array.xindex.value)
        utils.assert_array_equal(line.get_ydata(), array.value)
        plot.save(BytesIO(), format="png")
        plot.close()

    def test_ifft(self):
        """Test `FrequencySeries.ifft`."""
        # construct a TimeSeries, then check that it is unchanged by
        # the operation TimeSeries.fft().ifft()
        ts = TimeSeries([1.0, 0.0, -1.0, 0.0], sample_rate=1.0)
        utils.assert_quantity_sub_equal(ts.fft().ifft(), ts)
        utils.assert_allclose(ts.fft().ifft().value, ts.value)

    @pytest.mark.parametrize(("filter_form", "analog"), [
        pytest.param("zpk", True, id="zpk-analog"),
        pytest.param("zpk", False, id="zpk-digital"),
        pytest.param("ba", True, id="ba-analog"),
        pytest.param("ba", False, id="ba-digital"),
        pytest.param("sos", False, id="sos-digital"),
    ])
    def test_filter(
        self,
        array: FrequencySeries,
        filter_form: str,
        analog: bool,
    ):
        """Test `FrequencySeries.filter()`."""
        # Design a lowpass filter at nyquist/2
        nyq = array.frequencies.value[-1]
        filt_kw = {
            "btype": "low",
            "analog": analog,
            "output": filter_form,
        }
        if not analog:
            filt_kw["fs"] = nyq * 2
        filt = signal.butter(3, nyq / 2, **filt_kw)

        # Apply the filter
        a2 = array.filter(filt, analog=analog)

        # Check that we get back a FrequencySeries with the correct frequencies
        assert isinstance(a2, type(array))
        utils.assert_quantity_equal(a2.frequencies, array.frequencies)

        # Manually compute the filter response
        _, fresp = numpy.abs(filter_design.frequency_response(
            filt,
            array.frequencies.to("Hz").value,
            analog=analog,
            sample_rate=nyq * 2,
            unit="rad/s",
        ))

        # Check that the filter was applied correctly
        assert_allclose(a2.value, fresp * array.value)

    def test_zpk_analog_hertz(self, array):
        """Test `FrequencySeries.zpk(..., analog=True)`."""
        from gwpy.signal.filter_design import _convert_zpk_units

        zpk = ([100], [1], 1e-2)
        a2 = array.zpk(*zpk, unit="Hertz", analog=True)

        # Rebuild frequency-response manually
        z_rad, p_rad, k_rad = _convert_zpk_units(zpk, "Hz")
        omega = array.frequencies.value * 2 * numpy.pi
        fresp = numpy.abs(signal.freqs_zpk(z_rad, p_rad, k_rad, omega)[1])

        utils.assert_array_equal(a2.value, fresp * array.value)

    def test_inject(self):
        """Test `FrequencySeries.inject`."""
        # create a timeseries out of an array of zeros
        df, nyquist = 1, 2048
        nsamp = int(nyquist/df) + 1
        data = FrequencySeries(numpy.zeros(nsamp), f0=0, df=df, unit="")

        # create a second timeseries to inject into the first
        w_nyquist = 1024
        w_nsamp = int(w_nyquist/df) + 1
        sig = FrequencySeries(numpy.ones(w_nsamp), f0=0, df=df, unit="")

        # test that we recover this waveform when we add it to data,
        # and that the operation does not change the original data
        new_data = data.inject(sig)
        assert new_data.unit == data.unit
        assert new_data.size == data.size
        ind, = new_data.value.nonzero()
        assert len(ind) == sig.size
        utils.assert_allclose(new_data.value[ind], sig.value)
        utils.assert_allclose(data.value, numpy.zeros(nsamp))

    def test_interpolate(self):
        """Test `FrequencySeries.interpolate`."""
        # create a simple FrequencySeries
        df, nyquist = 1, 256
        nsamp = int(nyquist/df) + 1
        fseries = FrequencySeries(numpy.ones(nsamp), f0=1, df=df, unit="")

        # create an interpolated FrequencySeries
        newf = fseries.interpolate(df/2.)

        # check that the interpolated series is what was expected
        assert newf.unit == fseries.unit
        assert newf.size == 2*(fseries.size - 1) + 1
        assert newf.df == fseries.df / 2.
        assert newf.f0 == fseries.f0
        utils.assert_allclose(newf.value, numpy.ones(2*int(nyquist/df) + 1))

    @pytest.mark.requires("lal")
    def test_to_from_lal(self, array):
        """Test `FrequencySeries.to_lal` and `FrequencySeries.from_lal`."""
        import lal

        array.epoch = 0

        # check that to + from returns the same array
        lalts = array.to_lal()
        a2 = type(array).from_lal(lalts)
        utils.assert_quantity_sub_equal(array, a2, exclude=["channel"])

        # test copy=False
        a2 = type(array).from_lal(lalts, copy=False)
        assert shares_memory(a2.value, lalts.data.data)

        # test units
        array.override_unit("undef")
        with pytest.warns(
            UserWarning,
            match="LAL has no unit corresponding to 'NONE'",
        ):
            lalts = array.to_lal()
        assert lalts.sampleUnits == lal.DimensionlessUnit
        a2 = self.TEST_CLASS.from_lal(lalts)
        assert a2.unit == units.dimensionless_unscaled

    def test_to_from_pycbc(self, array):
        """Test `FrequencySeries.to_pycbc` and `FrequencySeries.from_pycbc`."""
        pycbctypes = pytest.importorskip("pycbc.types")

        array.epoch = 0

        # test default conversion
        pycbcfs = array.to_pycbc()
        assert isinstance(pycbcfs, pycbctypes.FrequencySeries)
        utils.assert_array_equal(array.value, pycbcfs.data)
        assert array.f0.value == 0 * units.Hz
        assert array.df.value == pycbcfs.delta_f
        assert array.epoch.gps == pycbcfs.epoch

        # go back and check we get back what we put in in the first place
        a2 = type(array).from_pycbc(pycbcfs)
        utils.assert_quantity_sub_equal(
            array, a2, exclude=["name", "unit", "channel"])

        # test copy=False
        a2 = type(array).from_pycbc(array.to_pycbc(copy=False), copy=False)
        assert shares_memory(array.value, a2.value)

    @pytest.mark.requires("pycbc")
    def test_to_from_pycbc_nonzero_f0(self, array):
        """Test `FrequencySeries.to_pycbc` conversion when ``f0 != 0``."""
        array.f0 = 1.
        with pytest.raises(
            ValueError,
            match=r"^Cannot convert FrequencySeries",
        ):
            array.to_pycbc()

    @pytest.mark.parametrize("fmt", [
        "txt",
        "csv",
    ])
    def test_read_write(self, array, fmt):
        """Test reading and writing FrequencySeries in various formats."""
        utils.test_read_write(
            array,
            fmt,
            assert_equal=utils.assert_quantity_sub_equal,
            assert_kw={
                "exclude": ["name", "channel", "unit", "epoch"],
            },
        )

    @staticmethod
    @pytest.fixture
    def ligolw(tmp_path) -> Path:
        """Return a `Path` with a LIGO_LW FrequencySeries array."""
        tmp = tmp_path / "test.xml"
        tmp.write_text(LIGO_LW_ARRAY)
        return tmp

    @pytest.mark.requires("lal", "igwn_ligolw")
    def test_read_ligolw(self, ligolw):
        """Test reading a `FrequencySeries` from ``LIGO_LW`` array."""
        array = FrequencySeries.read(ligolw, "PSD1")
        utils.assert_quantity_equal(
            array,
            [1, 2, 3, 4, 5] / units.Hz,
        )
        utils.assert_quantity_equal(
            array.frequencies,
            [10, 11, 12, 13, 14] * units.Hz,
        )
        assert numpy.isclose(array.epoch.gps, 1000000000)  # precision gah!
        assert array.unit == units.Hz ** -1

    @pytest.mark.requires("lal", "igwn_ligolw")
    def test_read_ligolw_params(self, ligolw):
        """Test reading a `FrequencySeries` from ``LIGO_LW`` array with params."""
        array = FrequencySeries.read(
            ligolw,
            channel="X1:TEST-CHANNEL_2",
        )
        assert list(array.value) == [10, 20, 30, 40, 50]
        assert array.epoch is None

    @pytest.mark.requires("lal", "igwn_ligolw")
    @pytest.mark.parametrize(("args", "match"), [
        # no name given, 'name' in error message
        pytest.param(
            [],
            "read: 'channel', 'epoch', 'f0', 'name'$",
            id="no-name",
        ),
        # name given, 'name' not in error message
        pytest.param(
            ("PSD2",),
            "read: 'channel', 'epoch', 'f0'$",
            id="name-only",
        ),
    ])
    def test_read_ligolw_error_multiple_array(self, args, match, ligolw):
        """Test reading a `FrequencySeries` from ``LIGO_LW`` with multiple <Array>."""
        # assert errors
        with pytest.raises(
            ValueError,
            match=match,
        ):  # multiple <Array> hits
            FrequencySeries.read(ligolw, *args)

    @pytest.mark.requires("lal", "igwn_ligolw")
    def test_read_ligolw_error_no_array(self, ligolw):
        """Test reading a `FrequencySeries` from ``LIGO_LW`` with no <Array>."""
        with pytest.raises(ValueError, match=r"^no <Array> elements found"):
            FrequencySeries.read(ligolw, "blah")

    @pytest.mark.requires("lal", "igwn_ligolw")
    def test_read_ligolw_error_no_match(self, ligolw):
        """Test reading a `FrequencySeries` from ``LIGO_LW`` with no matching Array."""
        with pytest.raises(
            ValueError,
            match="no <Array> elements found matching request",
        ):
            FrequencySeries.read(ligolw, epoch=0)

        with pytest.raises(
            ValueError,
            match="no <Array> elements found matching request",
        ):
            FrequencySeries.read(
                ligolw,
                "PSD1",
                f0=0,
            )

    @pytest.mark.requires("lal", "igwn_ligolw")
    def test_read_ligolw_error_no_param(self, ligolw):
        """Test reading a `FrequencySeries` from ``LIGO_LW`` with invalid param."""
        with pytest.raises(
            ValueError,
            match="no <Array> elements found matching request",
        ):
            FrequencySeries.read(
                ligolw,
                "PSD2",
                blah="blah",
            )

    @pytest.mark.requires("lal", "igwn_ligolw")
    def test_read_ligolw_error_dim(self, ligolw):
        """Test reading a `FrequencySeries` from ``LIGO_LW`` with invalid dim."""
        with pytest.raises(
            ValueError,
            match="cannot parse LIGO_LW Array with 3 dimensions in a Series",
        ):
            FrequencySeries.read(ligolw, epoch=1000000001)
