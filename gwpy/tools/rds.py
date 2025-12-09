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

r"""Create a reduced data set (RDS) using GWpy.

This tool allows you to create a reduced data set (RDS) by specifying
a list of channels, a start and end time, and an output file.
The tool will fetch the data for the specified channels and time range,
and write it to the output file in the specified format.

Example usage:

    gwpy-rds \
        -c H1:GWOSC-4KHZ_R1_STRAIN \
        -c L1:GWOSC-4KHZ_R1_STRAIN \
        -s 1126259462 \
        -e 1126259522 \
        -o gw150914.gwf
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .. import __version__
from ..time import to_gps
from ..timeseries import TimeSeriesDict
from . import _utils

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from collections.abc import Sequence

    from ..typing import GpsLike

logger = _utils.get_logger(__name__)

DOC_VERSION = "latest" if "dev" in __version__ else __version__
DOC_EPILOG = rf"""
For more information, see the online documentation at:

https://gwpy.readthedocs.io/en/{DOC_VERSION}/tools/rds/
"""


def create_rds(
    channels: Sequence[str],
    start: GpsLike,
    end: GpsLike,
    outfile: Path,
    source: Sequence[str] | str | None = None,
    format: str | None = None,  # noqa: A002
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

    kwargs
        Other keyword arguments are passed to `TimeSeriesDict.get`.

    See Also
    --------
    TimeSeriesDict.get
        For details of how data are accessed.

    TimeSeriesDict.write
        For details of how data are written.
    """
    logger.info("Getting data")
    data = TimeSeriesDict.get(
        channels,
        start,
        end,
        source=source,
        **kwargs,
    )
    logger.info("Received data for %d channels", len(data))
    data.write(
        outfile,
        format=format,
        overwrite=True,
    )
    logger.info("Wrote %s", outfile)


def create_parser() -> ArgumentParser:
    """Create an `argparse.ArgumentParser` for this tool."""
    parser = _utils.ArgumentParser(
        description=__doc__,
        epilog=DOC_EPILOG,
        prog="gwpy-rds",
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
        "--ifo",
        action="append",
        dest="channels",
        help=(
            "Data channel or IFO to request; can be specified multiple times; "
            "an IFO prefix (e.g., 'H1') can be passed to request the latest "
            "public strain data from GWOSC"
        ),
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
        action="count",
        default=0,
        help="Increase verbosity; pass once for INFO, twice for DEBUG",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=__version__,
        help="Show the version number and exit",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    """Run the command line tool to create a reduced data set (RDS)."""
    # Parse the command line arguments
    parser = create_parser()
    opts = parser.parse_args(args=args)

    # Init verbose logging
    _utils.init_verbose_logging("gwpy", opts.verbose)

    # Create the RDS
    create_rds(
        opts.channels,
        opts.start,
        opts.end,
        opts.output_file,
        opts.source,
        format=opts.format,
    )


if __name__ == "__main__":
    main()
