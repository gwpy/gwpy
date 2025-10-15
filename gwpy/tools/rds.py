# Copyright (c) 2024-2025 Cardiff University
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

"""Create a reduced data set (RDS) using GWpy."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .. import init_logging
from ..time import to_gps
from ..timeseries import TimeSeriesDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..typing import GpsLike

logger = logging.getLogger(__name__)


def create_rds(
    channels: Sequence[str],
    start: GpsLike,
    end: GpsLike,
    outfile: Path,
    source: Sequence[str] | str | None = None,
    format: str | None = None,  # noqa: A002
    *,
    verbose: bool = False,
    **kwargs,
) -> None:
    """Create a reduced data set by grabbing data and writing to a file.

    Parameters
    ----------
    channels : `list` of `str`
        The list of channel names to get.

    start : `LIGOTimeGPS`
        The GPS start time for the RDS.

    end : `LIGOTimeGPS`
        The GPS end time for the RDS.

    outfile : `str`, `pathlib.Path`
        The path of the output file to create.

    source : `str`, `list` of `str`, optional
        One or more supported data sources to use.

    format : `str`, optional
        The format option to use when writing data.
        Default is inferred from the output file path.
        See `TimeSeriesDict.write.help()` for details on supported formats.

    verbose : `bool`, optional
        Show debug-level logging output.

    kwargs
        Other keyword arguments are passed to `TimeSeriesDict.get`.

    See Also
    --------
    TimeSeriesDict.get
        For details of how data are accessed.

    TimeSeriesDict.write
        For details of how data are written.
    """
    logger.debug("Getting data")
    data = TimeSeriesDict.get(
        channels,
        start,
        end,
        source=source,
        verbose=verbose,
        **kwargs,
    )
    logger.debug("Received data for %d channels", len(data))
    data.write(
        outfile,
        format=format,
        overwrite=True,
    )
    logger.debug("Wrote %s", outfile)


def create_parser() -> argparse.ArgumentParser:
    """Create an `argparse.ArgumentParser` for this tool."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--start",
        required=True,
        type=to_gps,
        help="Start time of data request",
    )
    parser.add_argument(
        "-e",
        "--end",
        required=True,
        type=to_gps,
        help="End time of data request",
    )
    parser.add_argument(
        "-c",
        "--channel",
        action="append",
        dest="channels",
        help="Data channel to request",
    )
    parser.add_argument(
        "-g",
        "--source",
        action="append",
        help=(
            "Source from which to get data. "
            "See `help(TimeSeries.get)` for documentation on the "
            "supported sources"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        type=Path,
        help="Output file in which to write data",
    )
    parser.add_argument(
        "-f",
        "--format",
        help=(
            "Format in which to write output data. "
            "Default is inferred from -o/--output-file"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print verbose output",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    """Run the command line tool to create a reduced data set (RDS)."""
    # Parse the command line arguments
    parser = create_parser()
    opts = parser.parse_args(args=args)

    # Set the logging level
    logging.getLogger("gwpy").setLevel(logging.DEBUG if opts.verbose else logging.INFO)

    # Create the RDS
    create_rds(
        opts.channels,
        opts.start,
        opts.end,
        opts.output_file,
        opts.source,
        format=opts.format,
        verbose=opts.verbose,
    )


if __name__ == "__main__":
    init_logging()
    main()
