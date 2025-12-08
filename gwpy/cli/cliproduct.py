# Copyright (c) 2015-2020 Joseph Areeda
#               2020-2025 Cardiff University
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

"""Base class for CLI (`gwpy-plot`) products."""

from __future__ import annotations

import abc
import logging
import os.path
import re
import time
import warnings
from functools import wraps
from typing import (
    TYPE_CHECKING,
    cast,
)

from astropy.time import Time
from astropy.units import Quantity
from matplotlib import rcParams, style

from ..plot.colors import GW_OBSERVATORY_COLORS
from ..plot.gps import GPS_SCALES, GPSTransform
from ..plot.tex import label_to_latex
from ..segments import DataQualityFlag
from ..signal import filter_design
from ..signal.window import recommended_overlap
from ..time import to_gps
from ..timeseries import (
    TimeSeries,
    TimeSeriesDict,
)
from ..timeseries.timeseries import DEFAULT_FFT_METHOD
from ..utils import unique

if TYPE_CHECKING:
    from argparse import (
        ArgumentParser,
        Namespace,
        _ArgumentGroup,
    )
    from collections.abc import Callable
    from logging import Logger
    from os import PathLike
    from typing import (
        BinaryIO,
        ClassVar,
        Literal,
        ParamSpec,
        TypeVar,
    )

    from astropy.units import UnitBase
    from astropy.units.typing import QuantityLike
    from matplotlib.legend import Legend

    from ..plot import (
        Axes,
        Plot,
    )
    from ..signal.filter_design import ZpkType

    P = ParamSpec("P")
    R = TypeVar("R")

    TimeSeriesType = TypeVar("TimeSeriesType", bound=TimeSeries)

__author__ = "Joseph Areeda <joseph.areeda@ligo.org>"

BAD_UNITS = {
    "*",
    "Counts.",
}

#: GPS value below which it's probably not a GPS
MIN_GPS = 1e8

logger = logging.getLogger(__name__)


# -- utilities -----------------------

def timer(func: Callable[P, R]) -> Callable[P, R]:
    """Decorate a function to time itself and log its duration after return."""
    name = func.__name__

    @wraps(func)
    def timed_func(*args: P.args, **kwargs: P.kwargs) -> R:
        inst = cast("CliProduct", args[0])
        _start = time.time()
        out = func(*args, **kwargs)
        inst.logger.debug("%s took %.1f sec", name, time.time() - _start)
        return out

    return timed_func


def to_float(unit: str | UnitBase) -> Callable:
    """Return a function to convert a qantity string to a float in the given unit.

    Examples
    --------
    >>> conv = to_float('Hz')
    >>> conv('4 mHz')
    >>> 0.004
    """

    def converter(x: QuantityLike) -> float:
        return Quantity(x, unit).value

    converter.__doc__ = f"Convert the input to a `float` in {unit}."
    return converter


to_hz = to_float("Hz")
to_s = to_float("s")


# -- base product class --------------

class CliProduct(metaclass=abc.ABCMeta):
    """Base class for all cli plot products.

    Parameters
    ----------
    args : `argparse.Namespace`
        Command-line arguments as parsed using
        :meth:`~argparse.ArgumentParser.parse_args`

    Notes
    -----
    This object has two main entry points,

    - `CliProduct.init_cli` - which adds arguments and argument groups to
       an `~argparse.ArgumentParser`
    - `CliProduct.run` - executes the arguments to product one or more plots

    So, the basic usage is follows::

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> from gwpy.cli import CliProduct
    >>> CliProduct.init_cli(parser)
    >>> product = CliProduct(parser.parse_args())
    >>> product.run()

    The key methods for subclassing are

    - `CliProduct.action` - property defines 'name' for command-line subcommand
    - `CliProduct._finalize_arguments` - this is called from `__init__`
      to set defaults arguments for products that weren't set from the command
      line
    - `CliProduct.make_plot` - post-processes timeseries data and generates
      one figure
    """

    MIN_CHANNELS: ClassVar[int] = 1
    MIN_DATASETS: ClassVar[int] = 1
    MAX_DATASETS: ClassVar[int] = int(1e100)

    action: ClassVar[str | None] = None

    def __init__(
        self,
        args: Namespace,
        logger: Logger = logger,
    ) -> None:
        """Initialise this `CliProduct`."""
        #: input argument Namespace
        self.args = args

        #: Logger
        self.logger = logger

        # NB: finalizing may want to log if we're being verbose
        self._finalize_arguments(args)  # post-process args

        if args.style:  # apply custom styling
            style.use(args.style)

        #: the current figure object
        self.plot: Plot | None = None

        #: figure number
        self.plot_num = 0

        #: start times for data sets
        self.start_list = unique(map(
            int,
            (gps for gpsl in args.start for gps in gpsl),
        ))

        #: duration of each time segment
        self.duration = args.duration

        #: channels to load
        self.chan_list: list[str] = unique(c for clist in args.chan for c in clist)

        # use reduced as an alias for rds in channel name
        for idx in range(len(self.chan_list)):
            self.chan_list[idx] = self.chan_list[idx].replace("reduced", "rds")

        # total number of datasets that _should_ be acquired
        self.n_datasets = len(self.chan_list) * len(self.start_list)

        #: list of input data series (populated by get_data())
        self.timeseries: list[TimeSeries] = []

        #: dots-per-inch for figure
        self.dpi = args.dpi
        #: width and height in pixels
        self.width, self.height = map(float, self.args.geometry.split("x", 1))
        #: figure size in inches
        self.figsize = (self.width / self.dpi, self.height / self.dpi)
        #: Flag for data validation (like all zeroes)
        self.got_error = False

        # please leave this last
        self._validate_arguments()

    # -- abstract methods ------------

    @abc.abstractmethod
    def make_plot(self) -> Plot | None:
        """Generate the plot from time series and arguments.

        This method must be overridden by subclasses.

        Returns
        -------
        plot : `~gwpy.plot.Plot` or `None`
            The generated plot, or `None` if no plot could be made.
        """

    # -- properties ------------------

    @property
    def plot(self) -> Plot:
        """The current `~gwpy.plot.Plot` of this product.

        Raises a `RuntimeError` if the plot has not yet been created.
        """
        msg = "plot has not been created yet"
        try:
            plot = self._plot
        except AttributeError as exc:
            raise RuntimeError(msg) from exc
        if plot is None:
            raise RuntimeError(msg)
        return plot

    @plot.setter
    def plot(self, plot: Plot | None) -> None:
        """Set the current `~gwpy.plot.Plot` of this product."""
        self._plot = plot

    @property
    def axes(self) -> list[Axes]:
        """The list of `~matplotlib.axes.Axes` of this product's plot."""
        return cast("list[Axes]", self.plot.axes)

    @property
    def ax(self) -> Axes:
        """The current `~matplotlib.axes.Axes` of this product's plot."""
        return cast("Axes", self.plot.gca())

    @property
    def units(self) -> list[UnitBase]:
        """The (unique) list of data units for this product."""
        return unique(ts.unit for ts in self.timeseries)

    @property
    def usetex(self) -> bool:
        """Switch denoting whether LaTeX will be used or not."""
        return rcParams["text.usetex"]

    # -- utilities -------------------

    def log(
        self,
        level: int,
        msg: str,
        *args: object,
        **kwargs,
    ) -> None:
        """Print log message if verbosity is set high enough."""
        warnings.warn(
            "`CliProduct.log` is deprecated, use `CliProduct.logger` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        # Convert verbosity level int to logging level
        lvl = max(3 - level, 0) * 10
        self.logger.log(lvl, msg, *args, **kwargs)

    # -- argument parsing ------------

    # each method below is a classmethod so that the command-line
    # for a product can be set up without having to create an instance
    # of the class

    @classmethod
    def init_cli(cls, parser: ArgumentParser) -> None:
        """Set up the argument list for this product."""
        cls.init_data_options(parser)
        cls.init_plot_options(parser)

    @classmethod
    def init_data_options(cls, parser: ArgumentParser) -> None:
        """Set up data input and signal processing options."""
        cls.arg_channels(parser)
        cls.arg_data(parser)
        cls.arg_signal(parser)

    @classmethod
    def init_plot_options(cls, parser: ArgumentParser) -> None:
        """Set up plotting options."""
        cls.arg_plot(parser)
        cls.arg_xaxis(parser)
        cls.arg_yaxis(parser)

    # -- data options

    @classmethod
    def arg_channels(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for channel options."""
        group = parser.add_argument_group(
            "Data options",
            "What data to load",
        )
        group.add_argument(
            "--chan",
            "--channel",
            "--ifo",
            type=str,
            nargs="+",
            action="append",
            required=True,
            help="channels to load (or IFO prefix for GWOSC data)",
        )
        group.add_argument(
            "--start",
            type=to_gps,
            nargs="+",
            action="append",
            required=True,
            help=("one or more starting times (integer GPS or date/time string)"),
        )
        group.add_argument(
            "--duration",
            type=to_s,
            default=10,
            help="Duration (seconds) [10]",
        )
        return group

    @classmethod
    def arg_data(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for data options."""
        group = parser.add_argument_group(
            "Data source options",
            "Where to get the data",
        )
        meg = group.add_mutually_exclusive_group()
        meg.add_argument(
            "-c",
            "--framecache",
            type=os.path.abspath,
            help=(
                "path to a file containing a list of paths from which to read data"
            ),
        )
        meg.add_argument(
            "-n",
            "--nds2-server",
            metavar="HOSTNAME",
            help="name of nds2 server to use, default is to try all of them",
        )
        meg.add_argument(
            "--frametype",
            help="GWF frametype to read from",
        )
        meg.add_argument(
            "--gwosc",
            action="store_true",
            default=False,
            help="get data from GWOSC",
        )
        return group

    @classmethod
    def arg_signal(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for signal-processing options."""
        group = parser.add_argument_group(
            "Signal processing options",
            "What to do with the data before plotting",
        )
        group.add_argument(
            "--highpass",
            type=to_hz,
            help="Frequency for highpass filter",
        )
        group.add_argument(
            "--lowpass",
            type=to_hz,
            help="Frequency for lowpass filter",
        )
        group.add_argument(
            "--notch",
            type=to_hz,
            nargs="*",
            help="Frequency for notch (can give multiple)",
        )
        return group

    # -- plot options

    @classmethod
    def arg_plot(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for basic plot options."""
        group = parser.add_argument_group("Plot options")
        group.add_argument(
            "-g",
            "--geometry",
            default="1200x600",
            metavar="WxH",
            help="size of resulting image",
        )
        group.add_argument(
            "--dpi",
            type=int,
            default=rcParams["figure.dpi"],
            help="dots-per-inch for figure",
        )
        group.add_argument(
            "--interactive",
            action="store_true",
            help=(
                "presents display in an interactive window, see "
                "https://matplotlib.org/stable/users/explain/"
                "interactive.html#default-ui for details"
            ),
        )
        group.add_argument(
            "--title",
            action="store",
            help="Set title (below suptitle, defaults to parameter summary",
        )
        group.add_argument(
            "--suptitle",
            help="topmost title line (larger than the others)",
        )
        group.add_argument(
            "--out",
            default="gwpy.png",
            help=(
                "output filename (extension determines format: "
                "png, pdf, svg are available)"
            ),
        )

        # legends match input files in position are displayed if specified.
        group.add_argument(
            "--legend",
            nargs="+",
            action="append",
            default=[],
            help="strings to match data files",
        )
        group.add_argument(
            "--nolegend",
            action="store_true",
            help="do not display legend",
        )
        group.add_argument(
            "--nogrid",
            action="store_true",
            help="do not display grid lines",
        )

        # allow custom styling with a style file
        group.add_argument(
            "--style",
            metavar="FILE",
            help=(
                "path to custom matplotlib style sheet, see "
                "http://matplotlib.org/users/style_sheets.html#style-sheets "
                "for details of how to write one"
            ),
        )
        return group

    @classmethod
    def arg_xaxis(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for X-axis options."""
        return cls._arg_axis("x", parser)

    @classmethod
    def arg_yaxis(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for Y-axis options."""
        return cls._arg_axis("y", parser)

    @classmethod
    def _arg_axis(
        cls,
        axis: str,
        parser: ArgumentParser,
        **defaults,
    ) -> _ArgumentGroup:
        name = f"{axis.title()} axis"
        group = parser.add_argument_group(f"{name} options")

        # label
        group.add_argument(
            f"--{axis}label",
            default=defaults.get("label"),
            dest=f"{axis}label",
            help=f"{name} label",
        )

        # min and max
        for extrema in ("min", "max"):
            opt = axis + extrema
            group.add_argument(
                f"--{opt}",
                type=float,
                default=defaults.get(extrema),
                dest=opt,
                help=f"{extrema} value for {name}",
            )

        # scale
        scaleg = group.add_mutually_exclusive_group()
        scaleg.add_argument(
            f"--{axis}scale",
            type=str,
            default=defaults.get("scale"),
            dest=f"{axis}scale",
            help=f"scale for {name}",
        )
        if defaults.get("scale") == "log":
            scaleg.add_argument(
                f"--nolog{axis}",
                action="store_const",
                dest=f"{axis}scale",
                const=None,
                default="log",
                help=f"use linear {name}",
            )
        else:
            scaleg.add_argument(
                f"--log{axis}",
                action="store_const",
                dest=f"{axis}scale",
                const="log",
                default=None,
                help=f"use logarithmic {name}",
            )
        return group

    def _finalize_arguments(self, args: Namespace) -> None:
        """Sanity-check and set defaults for arguments."""
        # this method is called by __init__ (after command-line arguments
        # have been parsed)

        if args.out is None:
            args.out = "gwpy.png"

    def _validate_arguments(self) -> None:
        """Sanity check arguments and raise errors if required."""
        # validate number of data sets requested
        if len(self.chan_list) < self.MIN_CHANNELS:
            msg = f"this product requires at least {self.MIN_CHANNELS} channels"
            raise ValueError(msg)
        if self.n_datasets < self.MIN_DATASETS:
            msg = (
                f"{self.MIN_DATASETS} are required for this plot but only "
                f"{self.n_datasets} are supplied"
            )
            raise ValueError(msg)
        if self.n_datasets > self.MAX_DATASETS:
            msg = (
                f"A maximum of {self.MAX_DATASETS} datasets allowed for this "
                f"plot but {self.n_datasets} specified"
            )
            raise ValueError(msg)

    # -- data transfer ---------------

    @timer
    def get_data(self) -> None:
        """Get all the data.

        This method populates the `timeseries` list attribute
        """
        args = self.args

        # determine how we're supposed get our data
        if args.gwosc:
            source = "gwosc"
        elif args.framecache is not None:
            source = "cache"
        else:
            source = "nds2"

        # Get the data from NDS or Frames
        for start in self.start_list:
            end = start + self.duration
            if source.lower() == "gwosc":
                tsd = TimeSeriesDict()
                for chan in self.chan_list:
                    ifo = chan.split(":", 1)[0]
                    data = TimeSeries.fetch_open_data(
                        ifo,
                        start,
                        end,
                    )
                    data.name = f"{ifo}:{data.name}"
                    tsd.append({data.name: data})
            elif source.lower() == "nds2":
                tsd = TimeSeriesDict.get(
                    self.chan_list,
                    start,
                    end,
                    host=args.nds2_server,
                    frametype=args.frametype,
                )
            else:
                tsd = TimeSeriesDict.read(
                    args.framecache,
                    self.chan_list,
                    start=start,
                    end=end,
                )

            for data in tsd.values():
                if str(data.unit) in BAD_UNITS:
                    data.override_unit("undef")

                filtered = self._filter_timeseries(
                    data,
                    highpass=args.highpass,
                    lowpass=args.lowpass,
                    notch=args.notch,
                )

                if filtered.dtype.kind == "f":  # cast single to double
                    filtered = filtered.astype("float64", order="A", copy=False)

                self.timeseries.append(filtered)

        # report what we have if they asked for it
        self.logger.debug("Channels: %s", self.chan_list)
        self.logger.debug(
            "Start times: %s, duration %s",
            self.start_list,
            self.duration,
        )
        self.logger.debug("Number of time series: %d", len(self.timeseries))

    @staticmethod
    def _filter_timeseries(
        data: TimeSeriesType,
        highpass: QuantityLike | None = None,
        lowpass: QuantityLike | None = None,
        notch: list[QuantityLike] | None = None,
    ) -> TimeSeriesType:
        """Apply highpass, lowpass, and notch filters to some data."""
        # catch nothing to do
        if all(x is None for x in (highpass, lowpass, notch)):
            return data

        # build ZPK
        zpks: list[ZpkType] = []
        if highpass is not None and lowpass is not None:
            zpks.append(filter_design.bandpass(highpass, lowpass, data.sample_rate))
        elif highpass is not None:
            zpks.append(filter_design.highpass(highpass, data.sample_rate))
        elif lowpass is not None:
            zpks.append(filter_design.lowpass(lowpass, data.sample_rate))
        zpks.extend(filter_design.notch(freq, data.sample_rate) for freq in notch or [])
        zpk = filter_design.concatenate_zpks(*zpks)

        # apply forward-backward (zero-phase) filter
        return data.filter(zpk, filtfilt=True)

    # -- plotting --------------------

    def get_xlabel(self) -> str | None:
        """Return default X-axis label for plot."""
        return None

    def get_ylabel(self) -> str | None:
        """Return default Y-axis label for plot."""
        return None

    def get_title(self) -> str:
        """Return default title for plot."""
        highpass = self.args.highpass
        lowpass = self.args.lowpass
        notch = self.args.notch
        filt = ""
        if highpass and lowpass:
            filt += f"band pass ({highpass:.1f}-{lowpass:.1f})"
        elif highpass:
            filt += f"high pass ({highpass:.1f}) "
        elif lowpass:
            filt += f"low pass ({lowpass:.1f}) "
        if notch:
            filt += f", notch ({', '.join(map(str, notch))})"
        return filt

    def get_suptitle(self) -> str:
        """Return default super-title for plot."""
        ts = self.timeseries[0]
        return str(
            ts.name
            or ts.channel
            or "",
        )

    def set_plot_properties(self) -> None:
        """Finalize figure object and show() or save()."""
        self.set_axes_properties()
        self.set_legend()
        self.set_title(self.args.title)
        self.set_suptitle(self.args.suptitle)
        self.set_grid(enable=self.args.nogrid)

    def set_axes_properties(self) -> None:
        """Set properties for each axis (scale, limits, label)."""
        for ax in self.axes:
            self._scale_axes_from_data(ax=ax)
            self._set_xaxis_properties(ax=ax)
            self._set_yaxis_properties(ax=ax)

    @abc.abstractmethod
    def _scale_axes_from_data(self, ax: Axes) -> None:
        """Auto-scale the view based on visible data.

        In this base class, this method does nothing.
        """

    def _set_axis_properties(
        self,
        ax: Axes,
        axis: str,
    ) -> None:
        """Set properties for X/Y axis."""
        axis_name = axis[0]

        def _get(param: str) -> object:
            return getattr(ax, f"get_{axis_name}{param}")()

        def _set(param: str, *args: object, **kwargs) -> None:
            getattr(ax, f"set_{axis_name}{param}")(*args, **kwargs)

        scale = getattr(self.args, f"{axis}scale")
        label = getattr(self.args, f"{axis}label")
        min_ = getattr(self.args, f"{axis}min")
        max_ = getattr(self.args, f"{axis}max")

        # parse limits
        if (
            scale == "auto-gps"
            and min_ is not None
            and max_ is not None
            and max_ < MIN_GPS
        ):
            limits = (min_, min_ + max_)
        else:
            limits = (min_, max_)

        # set limits
        if limits[0] is not None or limits[1] is not None:
            _set("lim", *limits)

        # set scale
        if scale:
            _set("scale", scale)

        # reset scale with epoch if using GPS scale
        if _get("scale") in GPS_SCALES:
            _set("scale", scale, epoch=self.args.epoch)

        # set label
        if label is None:
            label = getattr(self, f"get_{axis_name}label")()
        if label:
            if self.usetex:
                label = label_to_latex(label)
            _set("label", label)

        # log
        dname = axis_name.upper()
        self.logger.debug(
            "%s-axis parameters: scale: %s, limits: %f -> %f",
            dname,
            _get("scale"),
            *cast("tuple[float, float]", _get("lim")),
        )
        self.logger.debug(
            "%s-axis label: %r",
            dname,
            _get("label"),
        )

    def _set_xaxis_properties(self, ax: Axes) -> None:
        """Set properties for X-axis."""
        self._set_axis_properties(ax, "x")

    def _set_yaxis_properties(self, ax: Axes) -> None:
        """Set properties for Y-axis."""
        self._set_axis_properties(ax, "y")

    def set_legend(self, ax: Axes | None = None) -> Legend | None:
        """Create a legend for this product (if applicable)."""
        if ax is None:
            ax = self.ax
        leg = self.ax.legend(prop={"size": 10})
        if leg and self.n_datasets == 1:
            try:
                leg.remove()
            except NotImplementedError:
                leg.set_visible(False)
        return leg

    def set_title(self, title: str, ax: Axes | None = None) -> None:
        """Set the title(s) for these `~matplotlib.axes.Axes`.

        The `Axes.title` actually serves at the sub-title for the plot,
        typically giving processing parameters and information.
        """
        if title is None:
            title_line = self.get_title().rstrip(", ")
        else:
            title_line = title

        if self.usetex:
            title_line = label_to_latex(title_line)
        if title_line:
            (ax or self.ax).set_title(title_line, fontsize=12)
            self.logger.debug("Title is: %s", title_line)

    def set_suptitle(self, suptitle: str | None) -> None:
        """Set the super title for this plot."""
        if not suptitle:
            suptitle = self.get_suptitle()
        if self.usetex:
            suptitle = label_to_latex(suptitle)
        self.plot.suptitle(suptitle, fontsize=18)
        self.logger.debug("Super title is: %s", suptitle)

    def set_grid(
        self,
        ax: Axes | None = None,
        *,
        enable: bool,
    ) -> None:
        """Set the grid parameters for this plot."""
        ax = ax or self.ax

        # Disable
        if not enable:
            ax.grid(visible=False)
            return

        # Major
        ax.grid(
            visible=True,
            which="major",
            color="k",
            linestyle="solid",
        )

        # Minor
        ax.grid(
            visible=True,
            which="minor",
            color="0.06",
            linestyle="dotted",
        )

    def save(self, outfile: str | PathLike | BinaryIO) -> None:
        """Save this product to the target `outfile`."""
        self.plot.savefig(outfile, edgecolor="white", bbox_inches="tight")
        self.logger.info("Wrote %s", outfile)

    def has_more_plots(self) -> bool:
        """Determine whether this product has more plots to be created."""
        return self.plot_num == 0

    @timer
    def _make_plot(self) -> None:
        """Override when product needs multiple saves."""
        self.plot = self.make_plot()

    def _color_by_ifo(self) -> list[str | None]:
        """Return a list of colours to use for datasets.

        In the specific case where each dataset name has a recognised IFO
        prefix, and there is only one of each type, the list will contain
        the relevant colour for that dataset, according to the colour scheme.

        In all other cases a list of `None` is returned.
        """
        # get all name prefices
        names = [
            str(series.name or series.channel).split(":", 1)[0]
            for series in self.timeseries
        ]
        unique = set(names)

        # if not all prefixes are unique, don't do anything
        nnames = len(names)
        if nnames != len(unique):
            return [None] * nnames

        # if not all prefices are in the colour scheme, don't do anything
        if not all(n in GW_OBSERVATORY_COLORS for n in names):
            return [None] * nnames

        # each prefix is in the colour scheme, and is unique, so use the
        # assigned colour
        return [GW_OBSERVATORY_COLORS[n] for n in names]

    # -- the one that does all the work

    def run(self) -> None:
        """Make the plot."""
        self.logger.debug("Arguments:")
        argsd = vars(self.args)
        for key in sorted(argsd):
            self.logger.debug("%15s = %s", key, argsd[key])

        # grab the data
        self.logger.info("---- Loading data ----")
        self.get_data()

        # for each plot
        self.logger.info("---- Generating figures ----")
        show_error = True  # control ours separate from product's
        while self.has_more_plots():
            self._make_plot()
            if self.plot:
                self.set_plot_properties()
                self.add_segs(self.args)
                if self.args.interactive:
                    self.logger.debug(
                        "Interactive manipulation of image should be available.",
                    )
                    self.plot.show()
                else:
                    self.save(self.args.out)
            elif show_error:
                # Some plots reject input data for reasons like all zeroes
                self.logger.warning(
                    "No plot produced because of data validation error.",
                )
                self.got_error = True
                show_error = False
            self.plot_num += 1

        self.logger.info("---- Complete ----")

    def add_segs(self, args: Namespace) -> None:
        """If requested add DQ segments."""
        std_segments = [
            "{ifo}:DMT-GRD_ISC_LOCK_NOMINAL:1",
            "{ifo}:DMT-DC_READOUT_LOCKED:1",
            "{ifo}:DMT-CALIBRATED:1",
            "{ifo}:DMT-ANALYSIS_READY:1",
        ]
        segments = []
        if hasattr(args, "std_seg"):
            if args.std_seg:
                segments = std_segments
            if args.seg:
                for seg in args.seg:
                    # NB: args.seg may be list of lists
                    segments += seg

            chan = args.chan[0][0]
            m = re.match("^([A-Za-z][0-9]):", chan)
            ifo = m.group(1) if m else "?:"

            start: Quantity | None = None
            end = 0
            for ts in self.timeseries:
                if start is not None:
                    start = min(ts.t0, start)
                    end = max(ts.t0 + ts.duration, end)
                else:
                    start = ts.t0
                    end = start + ts.duration

            for segment in segments:
                seg_name = segment.replace("{ifo}", ifo)
                seg_data = DataQualityFlag.query_dqsegdb(
                    seg_name,
                    start,
                    end,
                )

                self.plot.add_segments_bar(seg_data, label=seg_name)


# -- extensions ----------------------

class ImageProduct(CliProduct, metaclass=abc.ABCMeta):
    """Base class for all x/y/color plots."""

    DEFAULT_CMAP: ClassVar[str] = "viridis"
    MAX_DATASETS: ClassVar[int] = 1

    @classmethod
    def init_plot_options(cls, parser: ArgumentParser) -> None:
        """Configure plotting options, including colorbar."""
        super().init_plot_options(parser)
        cls.arg_color_axis(parser)

    @classmethod
    def arg_color_axis(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for colour-axis options."""
        group = parser.add_argument_group("Colour axis options")
        group.add_argument(
            "--imin",
            type=float,
            help="minimum value for colorbar",
        )
        group.add_argument(
            "--imax",
            type=float,
            help="maximum value for colorbar",
        )
        group.add_argument(
            "--cmap",
            default=cls.DEFAULT_CMAP,
            help=(
                "Colormap. See https://matplotlib.org/examples/color/"
                "colormaps_reference.html for options"
            ),
        )
        group.add_argument(
            "--color-scale",
            choices=("log", "linear"),
            help="scale for colorbar",
        )
        group.add_argument(
            "--norm",
            nargs="?",
            const="median",
            choices=("median", "mean"),
            metavar="NORM",
            help="normalise each pixel against average in that frequency bin",
        )
        group.add_argument(
            "--nocolorbar",
            action="store_true",
            help="hide the colour bar",
        )
        return group

    def get_color_label(self) -> str | None:
        """Return default colorbar label."""
        return None

    def set_axes_properties(self) -> None:
        """Set properties for each axis (scale, limits, label) and create a colorbar."""
        super().set_axes_properties()
        if not self.args.nocolorbar:
            self.set_colorbar()

    def set_colorbar(self) -> None:
        """Create a colorbar for this product."""
        self.ax.colorbar(label=self.get_color_label())

    def set_legend(self, ax: Axes | None = None) -> None:  # noqa: ARG002
        """Create a legend for this plot.

        This method does nothing, since image plots don't have legends.
        """
        return


class FFTMixin:
    """Mixin for `CliProduct` class that will perform FFTs.

    This just adds FFT-based command line options
    """

    DEFAULT_FFTLENGTH: ClassVar[float] = 1.0

    @classmethod
    def init_data_options(cls, parser: ArgumentParser) -> None:
        """Set up data input and signal processing options including FFTs."""
        super().init_data_options(parser)  # type: ignore[misc]
        cls.arg_fft(parser)

    @classmethod
    def arg_fft(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for FFT options."""
        group = parser.add_argument_group("Fourier transform options")
        group.add_argument(
            "--secpfft",
            type=float,
            default=cls.DEFAULT_FFTLENGTH,
            help="length of FFT in seconds",
        )
        group.add_argument(
            "--overlap",
            type=float,
            help="overlap as fraction of FFT length [0-1)",
        )
        group.add_argument(
            "--window",
            type=str,
            default="hann",
            help="window function to use when overlapping FFTs",
        )
        group.add_argument(
            "--average-method",
            dest="method",
            default=DEFAULT_FFT_METHOD,
            choices=("median", "welch", "bartlett"),
            help="FFT averaging method",
        )
        return group

    @classmethod
    def _arg_faxis(
        cls,
        axis: Literal["x", "y"],
        parser: ArgumentParser,
        **defaults,
    ) -> _ArgumentGroup:
        defaults.setdefault("scale", "log")
        name = f"{axis.title()} axis"
        group = parser.add_argument_group(f"{name} options")

        # label
        group.add_argument(
            f"--{axis}label",
            default=defaults.get("label"),
            dest=f"{axis}label",
            help=f"{name} label",
        )

        # min and max
        for extrema in ("min", "max"):
            meg = group.add_mutually_exclusive_group()
            for ax_ in (axis, "f"):
                meg.add_argument(
                    f"--{ax_}{extrema}",
                    type=float,
                    default=defaults.get(extrema),
                    dest=f"{axis}{extrema}",
                    help=f"{extrema} value for {name}",
                )

        # scale
        scaleg = group.add_mutually_exclusive_group()
        scaleg.add_argument(
            f"--{axis}scale",
            type=str,
            default=defaults.get("scale"),
            dest=f"{axis}scale",
            help=f"scale for {name}",
        )
        for ax_ in (axis, "f"):
            if defaults.get("scale") == "log":
                scaleg.add_argument(
                    f"--nolog{ax_}",
                    action="store_const",
                    dest=f"{axis}scale",
                    const=None,
                    default="log",
                    help=f"use linear {name}",
                )
            else:
                scaleg.add_argument(
                    f"--log{ax_}",
                    action="store_const",
                    dest=f"{axis}scale",
                    const="log",
                    default=None,
                    help="use logarithmic {name}",
                )

        return group

    def _finalize_arguments(self, args: Namespace) -> None:
        if args.overlap is None:
            try:
                args.overlap = recommended_overlap(args.window)
            except ValueError:
                args.overlap = 0.5
        return super()._finalize_arguments(args)  # type: ignore[misc]


class TimeDomainProduct(CliProduct, metaclass=abc.ABCMeta):
    """`CliProduct` with time on the X-axis."""

    @classmethod
    def arg_xaxis(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for X-axis options.

        This method includes the standard X-axis options, as well as a new
        ``--epoch`` option for the time axis.
        """
        group = super().arg_xaxis(parser)
        group.add_argument(
            "--epoch",
            type=to_gps,
            help=(
                "center X axis on this GPS time, may be absolute date/time or delta"
            ),
        )

        group.add_argument(
            "--std-seg",
            action="store_true",
            help="add DQ segment describing IFO state",
        )
        group.add_argument(
            "--seg",
            type=str,
            nargs="+",
            action="append",
            help="specify one or more DQ segment names",
        )
        return group

    def _finalize_arguments(self, args: Namespace) -> None:
        starts = [float(gps) for gpsl in args.start for gps in gpsl]
        if args.xscale is None:  # set default x-axis scale
            args.xscale = "auto-gps"
        if args.xmin is None:
            args.xmin = min(starts)
        if args.epoch is None:
            args.epoch = args.xmin
        elif args.epoch < MIN_GPS:
            args.epoch += min(starts)

        if args.xmax is None:
            args.xmax = max(starts) + args.duration
        return super()._finalize_arguments(args)

    def get_xlabel(self) -> str:
        """Return default X-axis label for plot."""
        trans = self.ax.xaxis.get_transform()
        if isinstance(trans, GPSTransform):
            epoch = trans.get_epoch()
            unit = trans.get_unit_name()
            utc = re.sub(r"\.0+", "", Time(epoch, format="gps", scale="utc").iso)
            return f"Time ({unit}) from {utc} ({epoch})"
        return ""


class FrequencyDomainProduct(FFTMixin, CliProduct, metaclass=abc.ABCMeta):
    """`CliProduct` with frequency on the X-axis."""

    def get_xlabel(self) -> str:
        """Return default X-axis label for plot."""
        return "Frequency (Hz)"
