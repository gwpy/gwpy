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

"""Spectrogram plots."""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    cast,
)

from numpy import percentile

from .cliproduct import (
    FFTMixin,
    ImageProduct,
    TimeDomainProduct,
)

if TYPE_CHECKING:
    from argparse import (
        ArgumentParser,
        Namespace,
        _ArgumentGroup,
    )
    from logging import Logger
    from typing import ClassVar

    from astropy.units import UnitBase
    from matplotlib.collections import Collection
    from matplotlib.image import AxesImage

    from ..plot import (
        Axes,
        Plot,
    )
    from ..spectrogram import Spectrogram

__author__ = "Joseph Areeda <joseph.areeda@ligo.org>"

logger = logging.getLogger(__name__)

#: Maximum number of FFTs to calculate for high-resolution spectrograms
MAX_FFT_PER_SECOND: int = 3


class SpectrogramProduct(FFTMixin, TimeDomainProduct, ImageProduct):
    """Plot the spectrogram of a time series."""

    action: ClassVar[str] = "spectrogram"

    def __init__(
        self,
        args: Namespace,
        logger: Logger = logger,
    ) -> None:
        """Initialise this `SpectrogramProduct`."""
        super().__init__(args, logger=logger)

        #: attribute to hold calculated Spectrogram data array
        self.result: Spectrogram | None = None

    @classmethod
    def arg_yaxis(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """Configure Y-axis arguments."""
        return cls._arg_faxis("y", parser)

    def _finalize_arguments(self, args: Namespace) -> None:
        """Sanity-check and set defaults for arguments."""
        if args.color_scale is None:
            args.color_scale = "log"
        super()._finalize_arguments(args)

    @property
    def units(self) -> list[UnitBase]:
        """The (unique) list of data units for this product."""
        if self.result is None:
            return [None]
        return [self.result.unit]

    def get_ylabel(self) -> str | None:
        """Return the label for the Y-axis."""
        return "Frequency (Hz)"

    def get_title(self) -> str:
        """Return default title for plot."""
        return f"fftlength={self.args.secpfft}, overlap={self.args.overlap}"

    def get_suptitle(self) -> str:
        """Return default super-title for plot."""
        return f"Spectrogram: {self.chan_list[0]}"

    def get_color_label(self) -> str | None:
        """Text for colorbar label."""
        if self.args.norm:
            return f"Normalized to {self.args.norm}"
        if len(self.units) == 1 and self.usetex:
            u = self.units[0].to_string("latex").strip("$")
            return fr"ASD $\left({u}\right)$"
        if len(self.units) == 1:
            u = self.units[0].to_string("generic")
            return f"ASD ({u})"
        return super().get_color_label()

    def get_stride(self) -> float | None:
        """Calculate the stride for the spectrogram.

        This method returns the stride as a `float`, or `None` to indicate
        selected usage of `TimeSeries.spectrogram2`.
        """
        fftlength = float(self.args.secpfft)
        overlap = fftlength * self.args.overlap
        stride = fftlength - overlap
        nfft = self.duration / stride  # number of FFTs
        ffps = int(nfft / (self.width * 0.8))  # FFTs per second
        if ffps > MAX_FFT_PER_SECOND:
            return max(2 * fftlength, ffps * stride + fftlength - 1)
        return None  # do not use strided spectrogram

    def get_spectrogram(self) -> Spectrogram | None:
        """Calculate the `Spectrogram` to be plotted.

        This exists as a separate method to allow subclasses to override
        this and not the entire `get_plot` method, e.g. `Coherencegram`.

        This method should not apply the normalisation from `args.norm`.
        """
        args = self.args

        fftlength = float(args.secpfft)
        overlap = fftlength * args.overlap
        self.logger.debug(
            "Calculating spectrogram secpfft: %f, overlap: %f",
            fftlength,
            overlap,
        )

        stride = self.get_stride()

        if stride:
            specgram = self.timeseries[0].spectrogram(
                stride,
                fftlength=fftlength,
                overlap=overlap,
                method=args.method,
                window=args.window,
            )
            nfft = stride * (stride // (fftlength - overlap))
            self.logger.debug(
                "Spectrogram calc, stride: %f, fftlength: %f, "
                "overlap: %f, #fft: %d",
                stride,
                fftlength,
                overlap,
                nfft,
            )
        else:
            specgram = self.timeseries[0].spectrogram2(
                fftlength=fftlength,
                overlap=overlap,
                window=args.window,
            )
            nfft = specgram.shape[0]
            self.logger.debug(
                "HR-Spectrogram calc, fftlength: %f, overlap: %f, #fft: %d",
                fftlength,
                overlap,
                nfft,
            )

        return specgram ** (1/2.)   # ASD

    def make_plot(self) -> Plot | None:
        """Generate the plot from time series and arguments."""
        args = self.args

        # constant input causes unhelpful (weird) error messages
        # translate them to English
        inmin = self.timeseries[0].min().value
        if inmin == self.timeseries[0].max().value:
            if not self.got_error:
                self.logger.error(
                    "Input has constant values [%g]. "
                    "Spectrogram-like products cannot process them.",
                    inmin,
                )
            self.got_error = True
            return None

        # create 'raw' spectrogram
        specgram = self.get_spectrogram()

        # there may be data that can't be processed
        if specgram is None:
            return None

        # apply normalisation
        if args.norm:
            specgram = specgram.ratio(args.norm)

        self.result = specgram

        # -- update plot defaults

        if not args.ymin:
            args.ymin = 1/args.secpfft if args.yscale == "log" else 0

        norm = "log" if args.color_scale == "log" else None
        # vmin/vmax set in scale_axes_from_data()
        return specgram.plot(
            figsize=self.figsize,
            dpi=self.dpi,
            norm=norm,
            cmap=args.cmap,
        )

    def _scale_axes_from_data(self, ax: Axes) -> None:
        """Auto-scale the view based on visible data."""
        args = self.args
        result = cast("Spectrogram", self.result)

        # get tight axes limits from time and frequency Axes
        if args.xmin is None:
            args.xmin = result.xspan[0]
        if args.xmax is None:
            args.xmax = result.xspan[1]
        if args.ymin is None:
            args.ymin = result.yspan[0]
        if args.ymax is None:
            args.ymax = result.yspan[1]

        specgram = result.crop(
            args.xmin,
            min(args.xmax, result.xspan[1]),
        )

        # y axis cannot be cropped if non-linear
        ax.set_ylim(args.ymin, args.ymax)

        # auto scale colours
        if args.norm:
            imin = specgram.value.min()
            imax = specgram.value.max()
        else:
            imin = percentile(specgram.value, 1)
            imax = percentile(specgram.value, 100)
        imin = args.imin if args.imin is not None else imin
        imax = args.imax if args.imax is not None else imax

        if imin == 0 and args.color_scale == "log":
            imin = specgram.value[specgram.value > 0].min()

        self.logger.debug("Colorbar limits set to %f - %f", imin, imax)

        image: AxesImage | Collection
        try:
            image = ax.images[0]
        except IndexError:
            image = ax.collections[0]
        image.set_clim(imin, imax)
