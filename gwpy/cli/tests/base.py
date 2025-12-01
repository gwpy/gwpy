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

"""Unit tests for :mod:`gwpy.cli`."""

from __future__ import annotations

import contextlib
import warnings
from argparse import ArgumentParser
from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
    cast,
)

import pytest
from matplotlib import pyplot
from numpy.random import default_rng

from ...plot import Plot
from ...testing import utils
from ...timeseries import TimeSeries
from .. import (
    SpectrumProduct,
    TransferFunctionProduct,
    cliproduct,
)

if TYPE_CHECKING:
    from argparse import Namespace
    from pathlib import Path
    from typing import ClassVar

    NamespaceType = TypeVar("NamespaceType", bound=Namespace)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

CliProductType = TypeVar("CliProductType", bound=cliproduct.CliProduct)
ImageProductType = TypeVar("ImageProductType", bound=cliproduct.ImageProduct)
TimeDomainProductType = TypeVar(
    "TimeDomainProductType",
    bound=cliproduct.TimeDomainProduct,
)
FrequencyDomainProductType = TypeVar(
    "FrequencyDomainProductType",
    bound=cliproduct.FrequencyDomainProduct,
)

RNG0 = default_rng(0)
RNG1 = default_rng(1)

# set data for nds2_connection test fixture
NDS2_CONNECTION_FIXTURE_DATA = [
    TimeSeries(
        RNG0.random(10240),
        t0=0,
        sample_rate=1024,
        name="X1:TEST-CHANNEL",
    ),
    TimeSeries(
        RNG1.random(10240),
        t0=0,
        sample_rate=1024,
        name="Y1:TEST-CHANNEL",
    ),
]


# -- utilities -----------------------

def update_namespace(args: NamespaceType, **params) -> NamespaceType:
    """Update the `argparse.Namespace` with new parameters."""
    for key, value in params.items():
        setattr(args, key, value)
    return args


# -- function tests ------------------

def test_to_float():
    """Test `to_float`."""
    to_s = cliproduct.to_float("s")
    s = to_s(3)
    assert isinstance(s, float)
    assert s == 3.
    assert to_s("3ms") == 0.004


# -- class tests ---------------------

class _TestCliProduct(Generic[CliProductType]):
    """Test the `CliProduct` class."""

    ACTION: ClassVar[str | None] = None
    TEST_CLASS: ClassVar[type[cliproduct.CliProduct]] = cliproduct.CliProduct  # type: ignore[type-abstract]
    TEST_ARGS: ClassVar[list[str]] = [
        "--chan", "X1:TEST-CHANNEL",
        "--start", "0",
        "--nds2-server", "nds.test.gwpy",
        "--dpi", "100",
        "--geometry", "640x480",
    ]
    NDS2_CONNECTION_FIXTURE_DATA = NDS2_CONNECTION_FIXTURE_DATA

    # -- fixtures --------------------

    @pytest.fixture
    @classmethod
    def args(cls) -> Namespace:
        """Create and parser arguments for a given `CliProduct`.

        Returns the `argparse.Namespace`
        """
        parser = ArgumentParser()
        parser.add_argument("--verbose", action="count", default=1)
        parser.add_argument("--silent", action="store_true")
        cls.TEST_CLASS.init_cli(parser)
        return parser.parse_args([str(x) for x in cls.TEST_ARGS])

    @pytest.fixture
    @classmethod
    def prod(cls, args):
        """Return a `CliProduct`."""
        prod = cls.TEST_CLASS(args)
        yield prod
        with contextlib.suppress(RuntimeError):
            prod.plot.close()

    @staticmethod
    def _prod_add_data(prod: CliProductType) -> CliProductType:
        # we need this method separate, rather than in dataprod, so that
        # we can have classes override the dataprod fixture with extra
        # stuff properly
        dur = prod.duration
        fs = 512

        i = 0
        for start in prod.start_list:
            for chan in prod.chan_list:
                rng = default_rng(i)
                ts = TimeSeries(
                    rng.random(int(fs*dur)),
                    t0=start,
                    sample_rate=512,
                    name=chan,
                )
                prod.timeseries.append(ts)
                i += 1
        return prod

    @pytest.fixture
    @classmethod
    def dataprod(cls, prod):
        """Return a `CliProduct` with data."""
        return cls._prod_add_data(prod)

    @staticmethod
    def _plotprod_init(prod: CliProductType) -> CliProductType:
        prod.plot = cast("Plot", pyplot.figure(FigureClass=Plot))
        prod.plot.gca()  # Initialise some axes
        return prod

    @pytest.fixture
    @classmethod
    def plotprod(cls, dataprod):
        """Return a `CliProduct` with data and a plot."""
        return cls._plotprod_init(dataprod)

    # -- tests -----------------------

    def test_init(self, args: Namespace):
        """Test `CliProduct` initialization."""
        channels = [
            self.TEST_ARGS[i+1]
            for i, arg in enumerate(self.TEST_ARGS)
            if arg == "--chan"
        ]
        prod = self.TEST_CLASS(args)
        with pytest.raises(RuntimeError):
            assert prod.plot is None
        assert prod.plot_num == 0
        assert prod.start_list == [0]
        assert prod.duration == 10.
        assert prod.chan_list == channels
        assert prod.n_datasets == len(prod.start_list) * len(prod.chan_list)
        assert prod.timeseries == []
        assert prod.dpi == 100.
        assert prod.width == 640
        assert prod.height == 480
        assert prod.figsize == (6.4, 4.8)

    def test_action(self, prod: CliProductType):
        """Test `CliProduct.action`."""
        if self.ACTION is None:
            raise NotImplementedError
        assert prod.action is self.ACTION

    @pytest.mark.parametrize("level", [
        pytest.param(1, id="info"),
        pytest.param(2, id="debug"),
    ])
    def test_log(
        self,
        prod: CliProductType,
        level: int,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test `CliProduct.log`."""
        caplog.set_level("INFO", logger=prod.logger.name)
        with pytest.warns(
            DeprecationWarning,
            match="`CliProduct.log` is deprecated",
        ):
            prod.log(level, "Test")
        assert ("Test" in caplog.text) is (level <= 1)

    @pytest.mark.requires("nds2")
    @pytest.mark.usefixtures("nds2_connection")
    def test_get_data(self, prod: CliProductType):
        """Test `CliProduct.get_data`."""
        prod.get_data()

        utils.assert_quantity_sub_equal(
            prod.timeseries[0],
            NDS2_CONNECTION_FIXTURE_DATA[0],
            exclude=("channel",),
        )

    @pytest.mark.parametrize(("ftype", "filt"), [
        pytest.param("highpass", {"highpass": 10}, id="highpass"),
        pytest.param("lowpass", {"lowpass": 100}, id="lowpass"),
        pytest.param("bandpass", {"highpass": 10, "lowpass": 100}, id="bandpass"),
    ])
    def test_filter_timeseries(
        self,
        dataprod: CliProductType,
        ftype: str,
        filt: dict[str, float],
    ):
        """Test `CliProduct._filter_timeseries`."""
        ts = dataprod.timeseries[0]
        if ftype == "bandpass":
            result = ts.bandpass(filt["highpass"], filt["lowpass"])
        else:
            result = cast("TimeSeries", getattr(ts, ftype)(filt[ftype]))

        fts = dataprod._filter_timeseries(ts, **filt)  # type: ignore[arg-type]
        utils.assert_quantity_sub_equal(fts, result)

    @pytest.mark.parametrize(("params", "title"), [
        pytest.param(
            {"highpass": 100},
            "high pass (100.0)",
            id="highpass",
        ),
        pytest.param(
            {"lowpass": 100},
            "low pass (100.0)",
            id="lowpass",
        ),
        pytest.param(
            {"highpass": 100, "lowpass": 200},
            "band pass (100.0-200.)",
            id="highlowband",
        ),
        pytest.param(
            {"highpass": 100, "notch": [60]},
            "high pass (100.0), notch(60.0)",
            id="highnotch",
        ),
    ])
    def test_get_title(
        self,
        prod: CliProductType,
        params: dict[str, float],
        title: str,
    ):
        """Test `CliProduct.get_title`."""
        update_namespace(prod.args, **params)  # update parameters
        assert prod.get_title() == title

    def test_get_suptitle(self, prod: CliProductType):
        """Test `CliProduct.get_suptitle`."""
        assert prod.get_suptitle() == prod.chan_list[0]

    @pytest.mark.parametrize("params", [
        pytest.param({}, id="default"),
        pytest.param(
            {
                "xscale": "linear",
                "xmin": 0,
                "xmax": 5,
                "xlabel": "X-label",
                "yscale": "log",
                "ymin": 0,
                "ymax": 50,
                "ylabel": "Y-label",
            },
            id="custom",
        ),
    ])
    def test_set_plot_properties(
        self,
        plotprod: CliProductType,
        params: dict[str, float],
    ):
        """Test `CliProduct.set_plot_properties`."""
        args = plotprod.args
        update_namespace(args, **params)  # update parameters

        if isinstance(plotprod, TransferFunctionProduct):
            data = plotprod.tfs
        elif isinstance(plotprod, SpectrumProduct):
            data = plotprod.spectra
        else:
            data = plotprod.timeseries
        xmin = min(series.xspan[0] for series in data)
        xmax = max(series.xspan[1] for series in data)

        # ignore warnings from matplotlib about having no labels
        # (because we have cut some corners in preparing this test)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="No artists with labels found to put in legend",
                category=UserWarning,
            )
            plotprod.set_plot_properties()

        ax = plotprod.ax

        ymin, ymax = ax.get_ylim()

        assert ax.get_xscale() == params.get("xscale", args.xscale)
        assert ax.get_xlim() == (params.get("xmin", xmin),
                                 params.get("xmax", xmax))
        assert ax.get_xlabel() == params.get("xlabel", plotprod.get_xlabel())

        assert ax.get_yscale() == params.get("yscale", args.yscale or "linear")
        assert ax.get_ylim() == (params.get("ymin", ymin),
                                 params.get("ymax", ymax))
        assert ax.get_ylabel() == params.get("ylabel", plotprod.get_ylabel())

    @pytest.mark.requires("nds2")
    @pytest.mark.usefixtures("nds2_connection")
    def test_run(
        self,
        tmp_path: Path,
        prod: CliProductType,
    ):
        """Test `CliProduct.run`."""
        tmp = tmp_path / "plot.png"
        prod.args.out = str(tmp)
        prod.run()
        assert tmp.is_file()
        assert prod.plot_num == 1
        assert not prod.has_more_plots()


class _TestImageProduct(_TestCliProduct[ImageProductType], Generic[ImageProductType]):
    """Test the `ImageProduct` class."""

    TEST_CLASS: ClassVar[type[cliproduct.ImageProduct]] = cliproduct.ImageProduct  # type: ignore[type-abstract]

    def test_extra_plot_options(self, args: Namespace):
        """Test `ImageProduct.extra_plot_options`."""
        for key in ("nocolorbar", "cmap", "imin", "imax"):
            assert hasattr(args, key)

    @pytest.mark.parametrize("visible", [False, True])
    def test_set_plot_properties(
        self,
        plotprod: ImageProductType,
        visible: bool,
    ):
        """Test `ImageProduct.set_plot_properties`."""
        update_namespace(plotprod.args, nocolorbar=not visible)
        plotprod.set_plot_properties()
        if visible:
            label = plotprod.get_color_label()
            coloraxes = plotprod.ax.child_axes[0]
            assert coloraxes.get_ylabel() == label
        else:
            # check our axes doesn't have any children
            assert not plotprod.ax.child_axes


class _TestFFTMixin:
    """Test the `FFTMixin` class."""


class _TestTimeDomainProduct(
    _TestCliProduct[TimeDomainProductType],
    Generic[TimeDomainProductType],
):
    """Test the `TimeDomainProduct` class."""

    TEST_CLASS: ClassVar[type[cliproduct.TimeDomainProduct]] = (
        cliproduct.TimeDomainProduct  # type: ignore[type-abstract]
    )


class _TestFrequencyDomainProduct(
    _TestCliProduct[FrequencyDomainProductType],
    Generic[FrequencyDomainProductType],
):
    """Test the `FrequencyDomainProduct` class."""

    TEST_CLASS: ClassVar[type[cliproduct.FrequencyDomainProduct]] = (
        cliproduct.FrequencyDomainProduct  # type: ignore[type-abstract]
    )
