# Copyright (c) 2021-2024 Evan Goetz
#               2021-2025 Cardiff University
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

"""Transfer function plots."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time

from ..plot.bode import BodePlot
from ..plot.tex import label_to_latex
from .cliproduct import FrequencyDomainProduct

if TYPE_CHECKING:
    from argparse import (
        ArgumentParser,
        Namespace,
        _ArgumentGroup,
    )
    from logging import Logger
    from typing import ClassVar

    from ..frequencyseries import FrequencySeries
    from ..plot import Axes
    from ..segments import Segment
    from ..timeseries import TimeSeries

__author__ = "Evan Goetz <evan.goetz@ligo.org>"

logger = logging.getLogger(__name__)


class TransferFunctionProduct(FrequencyDomainProduct):
    """Plot transfer function between two series."""

    action: ClassVar[str] = "transferfunction"

    MIN_DATASETS = 2

    def __init__(
        self,
        args: Namespace,
        logger: Logger = logger,
    ) -> None:
        """Create a new `TransferFunctionProduct`."""
        super().__init__(args, logger=logger)
        self.ref_chan = self.args.ref or self.chan_list[0]
        # deal with channel type appendages
        self.ref_chan = self.ref_chan.split(",", maxsplit=1)[0]
        self.plot_dB = self.args.plot_dB
        self.test_chan = self.chan_list[1]
        self.tfs: list[FrequencySeries] = []

    @classmethod
    def arg_channels(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for channel options."""
        group = super().arg_channels(parser)
        group.add_argument(
            "--ref",
            help="Reference channel against which others will be compared",
        )
        return group

    @classmethod
    def arg_yaxis(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for Y-axis options."""
        group = cls._arg_axis("ymag", parser, scale="log")
        group.add_argument(
            "--plot-dB",
            action="store_true",
            help="Plot transfer function in dB",
        )
        return cls._arg_axis("yphase", parser, scale="linear")

    def _finalize_arguments(self, args: Namespace) -> None:
        """Finalise command-line arguments for this plot product."""
        if args.ymagscale is None:
            args.ymagscale = "log"
        if args.plot_dB:
            args.ymagscale = "linear"

        super()._finalize_arguments(args)

    def get_title(self) -> str:
        """Generate the title for this plot."""
        gps = self.start_list[0]
        utc = Time(gps, format="gps", scale="utc").iso
        tstr = f"{utc} | {gps} ({self.duration})"

        fftstr = f"fftlength={self.args.secpfft}, overlap={self.args.overlap}"

        return f"{tstr}, {fftstr}"

    def get_ylabel(self) -> str:
        """Text for y-axis label."""
        ylabelstr = ""
        if self.ax is self.axes[0]:
            ylabelstr = "Magnitude"
            if self.plot_dB:
                ylabelstr += " [dB]"
        else:
            ylabelstr = "Phase [deg.]"

        return ylabelstr

    def get_suptitle(self) -> str:
        """Start of default super title, first channel is appended to it."""
        return f"Transfer function: {self.test_chan}/{self.ref_chan}"

    def _set_axis_properties(
        self,
        ax: Axes,
        axis: str,
    ) -> None:
        """Set properties for X/Y axis on a specific subplot."""
        # Remove xlabel from top subplot
        if ax is self.axes[0] and axis == "x":
            ax.set_xlabel("")
        super()._set_axis_properties(ax, axis)

    def _set_yaxis_properties(self, ax: Axes) -> None:
        """Set properties for Y-axis."""
        if ax is self.axes[0]:
            axis_name = "ymag"
        else:
            axis_name = "yphase"
        self.plot.sca(ax)
        self._set_axis_properties(ax, axis_name)

    def _scale_axes_from_data(self, ax: Axes) -> None:
        """Restrict data limits for Y-axis based on what you can see."""
        axes_type = "mag" if ax is self.axes[0] else "phase"

        # get tight limits for X-axis
        xmin = self.args.xmin
        xmax = self.args.xmax
        if xmin is None:
            xmin = min(tf.xspan[0] for tf in self.tfs)
            # this is then typically zero, so if the xscale is log or None
            # we'll need to set to be the next bin higher (one step of df)
            if (
                xmin == 0
                and (self.args.xscale == "log" or self.args.xscale is None)
            ):
                xmin = min(tf.df.value for tf in self.tfs)
        if xmax is None:
            xmax = max(tf.xspan[1] for tf in self.tfs)

        # autoscale view for Y-axis
        cropped = [tf.crop(xmin, xmax) for tf in self.tfs]
        ymin = None
        ymax = None
        for tf in cropped:
            if axes_type == "mag" and self.plot_dB:
                vals = 20 * np.log10(abs(tf.value))
            elif axes_type == "mag":
                vals = abs(tf.value)
            else:
                vals = np.angle(tf.value, deg=True)
            minval = float(min(vals))
            maxval = float(max(vals))
            if ymin is None or minval < ymin:
                ymin = minval
            if ymax is None or maxval > ymax:
                ymax = maxval

        yint = ax.yaxis.get_data_interval()
        if ymin is None:
            ymin = yint[0]
        if ymax is None:
            ymax = yint[1]
        ax.yaxis.set_data_interval(ymin, ymax, ignore=True)
        ax.autoscale_view(scalex=False)

    def set_plot_properties(self) -> None:
        """Finalize figure object and show() or save()."""
        self.set_axes_properties()
        self.set_title(self.args.title, self.axes[0])
        self.set_suptitle(self.args.suptitle)
        enable = not self.args.nogrid
        self.set_grid(ax=self.axes[0], enable=enable)
        self.set_grid(ax=self.axes[1], enable=enable)

    def make_plot(self) -> BodePlot:
        """Generate the transfer function plot from the time series."""
        args = self.args

        fftlength = float(args.secpfft)
        overlap = args.overlap
        self.logger.debug(
            "Calculating transfer function secpfft: %f, overlap: %f",
            fftlength,
            overlap,
        )
        if overlap is not None:
            overlap *= fftlength

        self.logger.debug("Reference channel: %s", self.ref_chan)

        # group data by segment
        groups: dict[Segment, dict[str, TimeSeries]] = {}
        for series in self.timeseries:
            seg = series.span
            name = str(series.name or series.channel or "")
            try:
                groups[seg][name] = series
            except KeyError:
                groups[seg] = OrderedDict()
                groups[seg][name] = series

        # -- plot

        plot = BodePlot(
            figsize=self.figsize,
            dpi=self.dpi,
            dB=self.plot_dB,
        )
        self.tfs = []

        # calculate transfer function
        for datadict in groups.values():
            refts = datadict.pop(self.ref_chan)
            for name, series in datadict.items():
                self.test_chan = name
                tf = series.transfer_function(
                    refts,
                    fftlength=fftlength,
                    overlap=overlap,
                    window=args.window,
                )

                label = name
                if len(self.start_list) > 1:
                    label += f", {series.t0.value}"
                if self.usetex:
                    label = label_to_latex(label)

                plot.add_frequencyseries(tf, dB=self.plot_dB, label=label)
                self.tfs.append(tf)

        if args.xscale == "log" and not args.xmin:
            args.xmin = 1 / fftlength

        return plot
