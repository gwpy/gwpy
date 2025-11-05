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

"""Spectrum plots."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from astropy.time import Time

from ..plot import Plot
from ..plot.tex import label_to_latex
from ..utils import unique
from .cliproduct import FrequencyDomainProduct

if TYPE_CHECKING:
    from argparse import (
        ArgumentParser,
        Namespace,
        _ArgumentGroup,
    )
    from logging import Logger
    from typing import ClassVar

    from astropy.units import UnitBase

    from ..frequencyseries import FrequencySeries
    from ..plot import Axes

__author__ = "Joseph Areeda <joseph.areeda@ligo.org>"

logger = logging.getLogger(__name__)


class SpectrumProduct(FrequencyDomainProduct):
    """Plot the ASD spectrum of one or more time series."""

    action: ClassVar[str] = "spectrum"

    def __init__(
        self,
        args: Namespace,
        logger: Logger = logger,
    ) -> None:
        """Create a new `Spectrum`."""
        super().__init__(args, logger=logger)
        self.spectra: list[FrequencySeries] = []

    @property
    def units(self) -> list[UnitBase]:
        """The (unique) list of data units for this product."""
        return unique(fs.unit for fs in self.spectra)

    @classmethod
    def arg_xaxis(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for X-axis options."""
        # use frequency axis on X
        return cls._arg_faxis("x", parser)

    @classmethod
    def arg_yaxis(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for Y-axis options."""
        # default log Y-axis
        return cls._arg_axis("y", parser, scale="log")

    def _finalize_arguments(self, args: Namespace) -> None:
        """Sanity-check and set defaults for arguments."""
        if args.yscale is None:
            args.yscale = "log"
        super()._finalize_arguments(args)

    def get_ylabel(self) -> str | None:
        """Text for y-axis label."""
        if len(self.units) == 1:
            u = self.units[0].to_string("latex").strip("$")
            return fr"ASD $\left({u}\right)$"
        return "ASD"

    def get_suptitle(self) -> str:
        """Start of default super title, first channel is appended to it."""
        return f"Spectrum: {self.chan_list[0]}"

    def get_title(self) -> str:
        """Return default title for plot."""
        gps = self.start_list[0]
        utc = Time(gps, format="gps", scale="utc").iso
        tstr = f"{utc} | {gps} ({self.duration})"

        fftstr = f"fftlength={self.args.secpfft}, overlap={self.args.overlap}"

        return f"{tstr}, {fftstr}"

    def make_plot(self) -> Plot:
        """Generate the plot from time series and arguments."""
        args = self.args

        fftlength = float(args.secpfft)
        overlap = args.overlap
        method = args.method
        self.logger.debug(
            "Calculating spectrum secpfft: %f, overlap: %f",
            fftlength,
            overlap,
        )
        overlap *= fftlength

        # create plot
        plot = Plot(figsize=self.figsize, dpi=self.dpi)
        ax = plot.gca()

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
            nlegargs = 0  # don't use  themm

        # determine colour
        colors = self._color_by_ifo()

        for i in range(self.n_datasets):
            series = self.timeseries[i]
            if nlegargs:
                label = self.args.legend[0][i]
            else:
                label = series.name or str(series.channel or "")
                if len(self.start_list) > 1:
                    label += f", {series.t0.value}"

            asd = series.asd(
                fftlength=fftlength,
                overlap=overlap,
                method=method,
            )
            self.spectra.append(asd)

            if self.usetex:
                label = label_to_latex(label)

            # plot
            ax.plot(asd, label=label, color=colors[i])

        if args.xscale == "log" and not args.xmin:
            args.xmin = 1/fftlength

        return plot

    def _scale_axes_from_data(self, ax: Axes) -> None:
        """Restrict data limits for Y-axis based on what you can see."""
        # get tight limits for X-axis
        if self.args.xmin is None:
            self.args.xmin = min(fs.xspan[0] for fs in self.spectra)
        if self.args.xmax is None:
            self.args.xmax = max(fs.xspan[1] for fs in self.spectra)

        # autoscale view for Y-axis
        cropped = [fs.crop(self.args.xmin, self.args.xmax) for fs in self.spectra]
        ymin = min(fs.value.min() for fs in cropped)
        ymax = max(fs.value.max() for fs in cropped)
        ax.yaxis.set_data_interval(ymin, ymax, ignore=True)
        ax.autoscale_view(scalex=False)
