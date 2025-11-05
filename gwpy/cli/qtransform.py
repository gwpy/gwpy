# Copyright (c) 2017-2020 Joseph Areeda
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

"""Q-transform plots."""

from __future__ import annotations

import logging
import os.path
import re
from collections.abc import Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    cast,
)

from astropy.units import Quantity

from ..segments import Segment
from ..time import to_gps
from .spectrogram import SpectrogramProduct

if TYPE_CHECKING:
    from argparse import (
        ArgumentParser,
        Namespace,
        _ArgumentGroup,
    )
    from logging import Logger
    from os import PathLike
    from typing import (
        ClassVar,
        SupportsFloat,
    )

    from ..plot import Axes
    from ..spectrogram import Spectrogram

logger = logging.getLogger(__name__)


class QtransformProduct(SpectrogramProduct):
    """Plot the Q-transform (Omega)."""

    DEFAULT_FFTLENGTH = None  # type: ignore[assignment]
    MAX_DATASETS = 1
    action: ClassVar[str] = "qtransform"

    def __init__(
        self,
        args: Namespace,
        logger: Logger = logger,
    ) -> None:
        """Create a new `QtransformProduct`."""
        super().__init__(args, logger=logger)

        args = self.args
        self.qxfrm_args = {
            "gps": float(args.gps),
            "search": args.search / 2.,
            "fres": 0.5,
            "tres": args.tres,
            "whiten": not args.nowhiten,
        }
        if args.qrange is not None:
            self.qxfrm_args["qrange"] = args.qrange
        if args.frange is not None:
            self.qxfrm_args["frange"] = args.frange

    @classmethod
    def init_data_options(cls, parser: ArgumentParser) -> None:
        """Set up data input and signal processing options."""
        super().init_data_options(parser)
        cls.arg_qxform(parser)

    @classmethod
    def arg_channels(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for channel options."""
        group = parser.add_argument_group("Data options", "What data to load")
        group.add_argument(
            "--chan",
            required=True,
            help="Channel name.",
        )
        group.add_argument(
            "--gps",
            type=to_gps,
            required=True,
            help="Central time of transform",
        )
        group.add_argument(
            "--search",
            type=float,
            default=64,
            help="Time window around GPS to search",
        )
        return group

    @classmethod
    def arg_signal(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for signal options."""
        group = super().arg_signal(parser)
        group.add_argument(
            "--sample-freq",
            type=float,
            default=2048,
            help="Downsample freq",
        )
        return group

    @classmethod
    def arg_plot(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse._ArgumentGroup` for plot options."""
        group = super().arg_plot(parser)

        # remove --out option
        outopt = next(act for act in group._actions if act.dest == "out")
        group._remove_action(outopt)

        # and replace with --outdir
        group.add_argument(
            "--outdir",
            default=os.path.curdir,
            dest="out",
            type=os.path.abspath,
            help="Directory for output images",
        )

        return group

    @classmethod
    def arg_qxform(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Add an `~argparse.ArgumentGroup` for Q-transform options."""
        group = parser.add_argument_group("Q-transform options")
        group.add_argument(
            "--plot",
            nargs="+",
            type=float,
            default=[.5],
            help="One or more times to plot",
        )
        group.add_argument(
            "--frange",
            nargs=2,
            type=float,
            help="Frequency range to plot",
        )
        group.add_argument(
            "--qrange",
            nargs=2,
            type=float,
            help="Search Q range",
        )

        group.add_argument(
            "--nowhiten",
            action="store_true",
            help="do not whiten input before transform",
        )
        return group

    def _finalize_arguments(self, args: Namespace) -> None:
        """Derive standard args from our weird ones."""
        gps = args.gps
        search = args.search
        # ensure we have enough data for filter settling
        max_plot = max(args.plot)
        search = max(search, max_plot * 2 + 8)
        args.search = search
        self.logger.debug(
            "Search window: %s sec, max plot window %s",
            search,
            max_plot,
        )

        # make sure we don't create too big interpolations

        xpix = 1200.
        if args.geometry:
            m = re.match("(\\d+)x(\\d+)", args.geometry)
            if m:
                xpix = float(m.group(1))
        # save output x for individual tres calulation
        args.nx = xpix
        self.args.tres = search / xpix / 2
        self.logger.debug("Max time resolution (tres) set to %s", self.args.tres)

        args.start = [[int(gps - search/2)]]
        if args.epoch is None:
            args.epoch = args.gps
        args.duration = search

        args.chan = [[args.chan]]

        if args.color_scale is None:
            args.color_scale = "linear"

        args.overlap = 0  # so that FFTMixin._finalize_arguments doesn't fail

        xmin = args.xmin
        xmax = args.xmax

        super()._finalize_arguments(args)

        # unset defaults from `TimeDomainProduct`
        args.xmin = xmin
        args.xmax = xmax

    def get_ylabel(self) -> str:
        """Return the default y-axis label."""
        return "Frequency (Hz)"

    def get_color_label(self) -> str:
        """Return the default colorbar label."""
        return "Normalized energy"

    def get_suptitle(self) -> str:
        """Return the default super-title for plot."""
        return f"Q-transform: {self.chan_list[0]}"

    def get_title(self) -> str:
        """Return the default title for plot."""
        result = cast("Spectrogram", self.result)

        def fformat(x: SupportsFloat | Sequence[SupportsFloat]) -> str:
            """Format a float for display."""
            if isinstance(x, Sequence):
                return f"[{', '.join(map(fformat, x))}]"
            if isinstance(x, Quantity):
                x = x.value
            return f"{x:.2f}"

        bits: list[tuple[str, ...]] = [("Q", fformat(result.q))]
        bits.append(("tres", f"{self.qxfrm_args['tres']:.3g}"))
        if self.qxfrm_args.get("qrange"):
            bits.append(("q-range", fformat(self.qxfrm_args["qrange"])))
        if self.qxfrm_args["whiten"] is not None:
            bits.append(("whitened",))
        bits.extend([
            ("f-range", fformat(result.yspan)),
            ("e-range", f"[{result.min():.3g}, {result.max():.3g}]"),
        ])
        return ", ".join([": ".join(bit) for bit in bits])

    def get_spectrogram(self) -> Spectrogram | None:
        """Generate a Q-transform `Spectrogram`."""
        args = self.args

        fftlength = args.secpfft
        overlap = args.overlap * fftlength if fftlength else None
        method = args.method
        asd = self.timeseries[0].asd(
            fftlength=fftlength,
            overlap=overlap,
            method=method,
        )

        if (asd.value.min() == 0):
            self.logger.error(
                "Input data has a zero in ASD. Q-transform not possible.",
            )
            self.got_error = True
            return None

        gps = self.qxfrm_args["gps"]
        outseg = Segment(gps, gps).protract(args.plot[self.plot_num])

        # use the precomputed ASD as the whitener if needed
        if self.qxfrm_args.get("whiten") is True:
            self.qxfrm_args["whiten"] = asd

        # This section tries to optimize the amount of data that is
        # processed and the time resolution needed to create a good
        # image. NB:For each time span specified
        # NB: the timeseries h enough data for the longest plot
        inseg = outseg.protract(4) & self.timeseries[0].span
        proc_ts = self.timeseries[0].crop(*inseg)

        #  time resolution is calculated to provide about 4 times
        # the number of output pixels for interpolation
        tres = float(outseg.end - outseg.start) / 4 / self.args.nx
        self.qxfrm_args["tres"] = tres
        self.qxfrm_args["search"] = int(len(proc_ts) * proc_ts.dt.value)

        self.logger.debug("Q-transform arguments:")
        self.logger.debug("         outseg = %s", outseg)
        for key, val in sorted(self.qxfrm_args.items()):
            self.logger.debug("%15s = %s", key, val)

        qtrans = proc_ts.q_transform(outseg=outseg, **self.qxfrm_args)

        if args.ymin is None:  # set before Spectrogram.make_plot
            args.ymin = qtrans.yspan[0]

        return qtrans

    def _scale_axes_from_data(self, ax: Axes) -> None:
        """Scale axes based on data."""
        return super()._scale_axes_from_data(ax)

    def has_more_plots(self) -> bool:
        """Return `True` if there are more plots to make."""
        return self.plot_num < len(self.args.plot)

    def save(self, outdir: str | PathLike) -> None:  # type: ignore[override]
        """Save the plot to the specified output directory."""
        ts = self.timeseries[0]
        cname = re.sub(
            r"[-_:]", "_",
            str(ts.name or ts.channel or "DATA"),
        ).replace("_", "-", 1)
        png = f"{cname}-{float(self.args.gps)}-{self.args.plot[self.plot_num]}.png"
        outfile = Path(outdir) / png
        return super().save(outfile)
