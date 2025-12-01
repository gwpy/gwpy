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

"""Coherence spectrogram."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .spectrogram import SpectrogramProduct

if TYPE_CHECKING:
    from argparse import (
        ArgumentParser,
        Namespace,
        _ArgumentGroup,
    )
    from logging import Logger
    from typing import ClassVar

    from ..spectrogram import Spectrogram

__author__ = "Joseph Areeda <joseph.areeda@ligo.org>"

logger = logging.getLogger(__name__)


class CoherencegramProduct(SpectrogramProduct):
    """Plot the coherence-spectrogram comparing two time series."""

    DEFAULT_CMAP: ClassVar[str] = "plasma"
    MIN_DATASETS: ClassVar[int] = 2
    MAX_DATASETS: ClassVar[int] = 2
    action: ClassVar[str] = "coherencegram"

    def __init__(
        self,
        args: Namespace,
        logger: Logger = logger,
    ) -> None:
        """Create a new `CoherencegramProduct`."""
        super().__init__(args, logger=logger)
        self.ref_chan = self.args.ref or self.chan_list[0]
        # deal with channel type (e.g. m-trend)
        self.ref_chan = self.ref_chan.split(",", maxsplit=1)[0]

    @classmethod
    def arg_channels(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Configure `~argparse.ArgumentParser` arguments for channels."""
        group = super().arg_channels(parser)
        group.add_argument(
            "--ref",
            help="Reference channel against which others will be compared",
        )
        return group

    def _finalize_arguments(self, args: Namespace) -> None:
        """Finalise arguments with defaults."""
        if args.color_scale is None:
            args.color_scale = "linear"
        if args.color_scale == "linear":
            if args.imin is None:
                args.imin = 0.0
            if args.imax is None:
                args.imax = 1.0
        return super()._finalize_arguments(args)

    def get_ylabel(self) -> str:
        """Text for y-axis label."""
        return "Frequency (Hz)"

    def get_suptitle(self) -> str:
        """Start of default super title, first channel is appended to it."""
        a, b = self.chan_list
        return f"Coherence spectrogram: {a} vs {b}"

    def get_color_label(self) -> str:
        """Return the default colorbar label."""
        if self.args.norm:
            return f"Normalized to {self.args.norm}"
        return "Coherence"

    def get_stride(self) -> float:
        """Calculate the stride for the coherencegram."""
        fftlength = float(self.args.secpfft)
        overlap = self.args.overlap  # fractional overlap
        return max(
            self.duration / (self.width * 0.8),
            fftlength * (1 + (1 - overlap) * 32),
            fftlength * 2,
        )

    def get_spectrogram(self) -> Spectrogram:
        """Calculate the `Spectrogram` to be plotted."""
        args = self.args
        fftlength = float(args.secpfft)
        overlap = args.overlap  # fractional overlap
        stride = self.get_stride()
        self.logger.debug(
            "Calculating coherence spectrogram, secpfft: %s, overlap: %s",
            fftlength,
            overlap,
        )

        if overlap is not None:  # overlap in seconds
            overlap *= fftlength

        if self.timeseries[0].name == self.ref_chan:
            ref = 0
            other = 1
        else:
            ref = 1
            other = 0
        return self.timeseries[ref].coherence_spectrogram(
            self.timeseries[other],
            stride,
            fftlength=fftlength,
            overlap=overlap,
            window=args.window,
        )
