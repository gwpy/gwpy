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

"""Coherence plots."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..plot import Plot
from ..plot.tex import label_to_latex
from .spectrum import SpectrumProduct

if TYPE_CHECKING:
    from argparse import (
        ArgumentParser,
        Namespace,
        _ArgumentGroup,
    )
    from logging import Logger
    from typing import ClassVar

    from matplotlib.legend import Legend

    from ..frequencyseries import FrequencySeries
    from ..plot import Axes
    from ..segments import Segment

__author__ = "Joseph Areeda <joseph.areeda@ligo.org>"

logger = logging.getLogger(__name__)


class CoherenceProduct(SpectrumProduct):
    """Plot coherence between two timeseries."""

    action: ClassVar[str] = "coherence"

    MIN_DATASETS: ClassVar[int] = 2

    def __init__(
        self,
        args: Namespace,
        logger: Logger = logger,
    ) -> None:
        """Create a new `Coherence` product."""
        super().__init__(args, logger=logger)
        self.ref_chan = self.args.ref or self.chan_list[0]
        # deal with channel type appendages
        self.ref_chan = self.ref_chan.split(",", 1)[0]

    @classmethod
    def arg_channels(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Set up channel arguments for this product."""
        group = super().arg_channels(parser)
        group.add_argument(
            "--ref",
            help="Reference channel against which others will be compared",
        )
        return group

    def _finalize_arguments(self, args: Namespace) -> None:
        if args.yscale is None:
            args.yscale = "linear"
        if args.yscale == "linear":
            if not args.ymin:
                args.ymin = 0
            if not args.ymax:
                args.ymax = 1.05
        return super()._finalize_arguments(args)

    def get_ylabel(self) -> str:
        """Text for y-axis label."""
        return "Coherence"

    def get_suptitle(self) -> str:
        """Start of default super title, first channel is appended to it."""
        return f"Coherence: {self.ref_chan}"

    def make_plot(self) -> Plot:
        """Generate the coherence plot from all time series."""
        args = self.args

        fftlength = float(args.secpfft)
        overlap = args.overlap
        self.logger.debug(
            "Calculating spectrum secpfft: %s, overlap: %s",
            fftlength,
            overlap,
        )
        if overlap is not None:
            overlap *= fftlength

        self.logger.debug("Reference channel: %s", self.ref_chan)

        # group data by segment
        groups: dict[Segment, dict[str, FrequencySeries]] = {}
        for series in self.timeseries:
            seg = series.span
            name = str(series.name or series.channel or "")
            try:
                groups[seg][name] = series
            except KeyError:
                groups[seg] = {}
                groups[seg][name] = series

        # -- plot

        plot = Plot(figsize=self.figsize, dpi=self.dpi)
        ax = plot.gca()
        self.spectra: list[FrequencySeries] = []

        # calculate coherence
        for group in groups.values():
            refts = group.pop(self.ref_chan)
            for name, series in group.items():
                coh = series.coherence(
                    refts,
                    fftlength=fftlength,
                    overlap=overlap,
                    window=args.window,
                )

                label = name
                if len(self.start_list) > 1:
                    label += f", {series.x0.value}"
                if self.usetex:
                    label = label_to_latex(label)

                ax.plot(coh, label=label)
                self.spectra.append(coh)

        if args.xscale == "log" and not args.xmin:
            args.xmin = 1/fftlength

        return plot

    def set_legend(self, ax: Axes | None = None) -> Legend | None:
        """Create a legend for this product."""
        leg = super().set_legend(ax=ax)
        if leg is not None:
            leg.set_title("Coherence with:")
        return leg
