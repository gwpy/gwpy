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

"""The timeseries CLI product."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from ..plot import Plot
from ..plot.tex import label_to_latex
from .cliproduct import TimeDomainProduct

if TYPE_CHECKING:
    from argparse import Namespace
    from logging import Logger
    from typing import ClassVar

    from ..plot import Axes

__author__ = "Joseph Areeda <joseph.areeda@ligo.org>"

logger = logging.getLogger(__name__)


class TimeSeriesProduct(TimeDomainProduct):
    """Plot one or more time series."""

    action: ClassVar[str] = "timeseries"

    def __init__(
        self,
        args: Namespace,
        logger: Logger = logger,
    ) -> None:
        """Create a new `TimeSeriesProduct`."""
        super().__init__(args, logger=logger)

    def get_ylabel(self) -> str | None:
        """Text for y-axis label,  check if channel defines it."""
        units = self.units
        if len(units) == 1 and str(units[0]) == "":  # dimensionless
            return ""
        if len(units) == 1 and self.usetex:
            return units[0].to_string("latex")
        if len(units) == 1:
            return units[0].to_string()
        if len(units) > 1:
            return "Multiple units"
        return super().get_ylabel()

    def get_suptitle(self) -> str:
        """Start of default super title, first channel is appended to it."""
        return f"Time series: {self.chan_list[0]}"

    def get_title(self) -> str:
        """Return the title of this `TimeSeries` product."""
        suffix = super().get_title()
        # limit significant digits for minute trends
        rates = {ts.sample_rate.round(3) for ts in self.timeseries}
        fss = f"({'), ('.join(map(str, rates))})"
        return ", ".join([
            f"Fs: {fss}",
            f"duration: {self.duration}",
            suffix,
        ])

    def make_plot(self) -> Plot:
        """Generate the plot from time series and arguments."""
        plot = Plot(figsize=self.figsize, dpi=self.dpi)
        ax = plot.add_subplot(xscale="auto-gps")

        # handle user specified plot labels
        if self.args.legend:
            nlegargs = len(self.args.legend[0])
        else:
            nlegargs = 0
        if nlegargs > 0 and nlegargs != self.n_datasets:
            warnings.warn(
                "The number of legends specified must match the number of "
                "time series (channels * start times). "
                f"There are {len(self.timeseries)} series "
                f"and {len(self.args.legend)} legends",
                stacklevel=2,
            )
            nlegargs = 0    # don't use them

        # get colours
        colors = self._color_by_ifo()

        for i in range(self.n_datasets):
            series = self.timeseries[i]
            if nlegargs:
                label = self.args.legend[0][i]
            elif series.channel:  # GWOSC data doesn't have a 'channel'
                label = series.channel.name
            else:
                label = series.name
            if self.usetex:
                label = label_to_latex(label)
            ax.plot(series, label=label, color=colors[i])

        return plot

    def _scale_axes_from_data(self, ax: Axes) -> None:
        """Restrict data limits for Y-axis based on what you can see."""
        # get tight limits for X-axis
        if self.args.xmin is None:
            self.args.xmin = min(ts.xspan[0] for ts in self.timeseries)
        if self.args.xmax is None:
            self.args.xmax = max(ts.xspan[1] for ts in self.timeseries)

        # autoscale view for Y-axis
        cropped = [ts.crop(self.args.xmin, self.args.xmax) for
                   ts in self.timeseries]
        ymin = min(ts.value.min() for ts in cropped)
        ymax = max(ts.value.max() for ts in cropped)
        ax.yaxis.set_data_interval(ymin, ymax, ignore=True)
        ax.autoscale_view(scalex=False)
