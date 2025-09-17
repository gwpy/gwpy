# Copyright (c) 2014-2017 Louisiana State University
#               2018-2025 Cardiff University
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

"""Tests for `gwpy.plot.gps`."""

from decimal import Decimal

import numpy
import pytest
from astropy.units import Unit
from matplotlib import pyplot

from ...time import LIGOTimeGPS
from .. import gps as plot_gps


class TestGPSMixin:
    """Tests for `GPSMixin`."""

    TYPE = plot_gps.GPSMixin

    def test_init(self):
        """Test `GPSMixin()`."""
        m = self.TYPE()
        assert m.unit is None
        assert m.epoch is None

    def test_init_args(self):
        """Test `GPSMixin()` with args."""
        m = self.TYPE(unit="second", epoch=100)
        assert m.unit is Unit("second")
        assert m.epoch == 100.

    @pytest.mark.parametrize(("in_", "out"), [
        pytest.param(
            None,
            None,
            id="None",
        ),
        pytest.param(
            1,
            1.,
            id="int",
        ),
        pytest.param(
            "1",
            1.,
            id="str",
        ),
        pytest.param(
            Decimal(12345),
            12345.,
            id="Decimal",
        ),
        pytest.param(
            numpy.float32(56789),
            56789.,
            id="numpy.float",
        ),
        pytest.param(
            LIGOTimeGPS(1234567890, 123000000),
            1234567890.123,
            id="LIGOTimeGPS",
        ),
    ])
    def test_epoch(self, in_, out):
        """Test `GPSMixin.epoch`."""
        mix = self.TYPE(epoch=in_)
        assert mix.epoch == out

    @pytest.mark.parametrize(("in_", "out"), [
        (None, None),
        (Unit("second"), Unit("second")),
        (3600, Unit("hour")),
        ("week", Unit("week")),
        ("weeks", Unit("week")),
    ])
    def test_unit(self, in_, out):
        """Test `GPSMixin.unit`."""
        mix = self.TYPE(unit=in_)
        assert mix.unit == out

    @pytest.mark.parametrize(("badunit", "match"), [
        pytest.param(
            "blah",
            "'blah' did not parse as unit",
            id="bad unit",
        ),
        pytest.param(
            "meter",
            "cannot set GPS unit to 'm'",
            id="wrong type",
        ),
        pytest.param(
            "yoctoday",
            "unrecognised unit 'yoctoday'",
            id="unsupported",
        ),
    ])
    def test_unit_error(self, badunit, match):
        """Test `GPSMixin.unit` error handling."""
        with pytest.raises(
            ValueError,
            match=match,
        ):
            self.TYPE(unit=badunit)

    @pytest.mark.parametrize(("unit", "name"), [
        (None, None),
        ("second", "seconds"),
        (Unit("sday"), "sdays"),  # no long_name
    ])
    def test_get_unit_name(self, unit, name):
        """Test `GPSMixin.get_unit_name()`."""
        mix = self.TYPE(unit=unit)
        assert mix.get_unit_name() == name

    @pytest.mark.parametrize(("unit", "scale"), [
        (None, 1),
        ("second", 1),
        ("minute", 60),
        ("hour", 3600),
    ])
    def test_scale(self, unit, scale):
        """Test `GPSMixin.scale`."""
        mix = self.TYPE(unit=unit)
        assert mix.scale == scale


class TestGpsTransform(TestGPSMixin):
    """Tests for `GPSTransform`."""

    TRANSFORM = plot_gps.GPSTransform
    EPOCH = 100.0
    UNIT = "minutes"
    SCALE = 60.
    X = 190.0
    A = 90.0
    B = 19/6.
    C = 1.5

    def test_init(self):
        """Test `GPSTransform()`."""
        t = self.TRANSFORM()
        assert t.transform(1.0) == 1.0

    def test_epoch(self):
        """Test `GPSTransform.epoch`."""
        transform = self.TRANSFORM(epoch=self.EPOCH)
        assert transform.get_epoch() == self.EPOCH
        assert transform.transform(self.X) == self.A
        assert numpy.isclose(
            transform.inverted().transform(transform.transform(self.X)),
            self.X)

    def test_scale(self):
        """Test `GPSTransform.scale`."""
        transform = self.TRANSFORM(unit=self.UNIT)
        assert transform.get_scale() == self.SCALE
        assert transform.transform(self.X) == self.B
        assert numpy.isclose(
            transform.inverted().transform(transform.transform(self.X)),
            self.X)

    def test_transform(self):
        """Test `GPSTransform.transform`."""
        transform = self.TRANSFORM(epoch=self.EPOCH, unit=self.UNIT)
        assert transform.transform(self.X) == self.C
        assert numpy.isclose(
            transform.inverted().transform(transform.transform(self.X)),
            self.X,
        )


class TestInverseGpsTransform(TestGpsTransform):
    """Tests for `InverseGPSTransform`."""

    TRANSFORM = plot_gps.InvertedGPSTransform
    A = 290.0
    B = 11400.0
    C = 11500.0


@pytest.mark.parametrize(
    "scale",
    sorted(filter(lambda x: x != "auto-gps", plot_gps.GPS_SCALES)),
)
def test_gps_scale(scale):
    """Test `GPSScale` in action."""
    u = Unit(scale[:-1])

    fig = pyplot.figure()
    ax = fig.add_subplot(xscale=scale)
    if scale == "years":
        x = numpy.arange(50)
    else:
        x = numpy.arange(1e2)
    ax.plot(x * u.decompose().scale, x)
    fig.canvas.draw()
    xscale = ax.get_xaxis()._scale
    assert xscale.get_unit() == Unit(scale[:-1])
    pyplot.close(fig)


@pytest.mark.parametrize(("scale", "unit"), [
    (1e-5, "ms"),
    (1e-4, "ms"),
    (1e-3, "s"),
    (1e-2, "s"),
    (1e-1, "s"),
    (1e0, "s"),
    (1e1, "min"),
    (1e2, "min"),
    (1e3, "h"),
    (1e4, "d"),
    (1e5, "wk"),
    (1e6, "wk"),
    (1e7, "yr"),
])
def test_auto_gps_scale(scale, unit):
    """Test ``auto-gps`` scale in action."""
    fig = pyplot.figure()
    ax = fig.add_subplot(xscale="auto-gps")
    ax.plot(numpy.arange(1e2) * scale, numpy.arange(1e2))
    xscale = ax.get_xaxis()._scale
    transform = xscale.get_transform()
    assert transform.unit.name == unit
    pyplot.close(fig)


def test_gps_formatting():
    """Test GPS time formatting."""
    fig = pyplot.figure()
    try:
        ax = fig.gca()
        ax.set_xscale("seconds", epoch=1238040211.67)
        ax.set_xlim(1238040211.17, 1238040212.17)
        fig.canvas.draw()
        ticks = ["-0.5", "-0.4", "-0.3", "-0.2", "-0.1",
                 "0", "0.1", "0.2", "0.3", "0.4", "0.5"]
        assert [x.get_text() for x in ax.get_xticklabels()] == ticks
    finally:
        pyplot.close(fig)
