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
a start and end time, a list of channels, and an output file.
The tool will fetch the data for the specified channels and time range,
and write it to the output file in the specified format.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .. import __version__
from ..io.registry import default_registry
from ..time import to_gps
from ..timeseries import TimeSeriesDict
from . import _utils

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from collections.abc import Sequence

    from ..time import SupportsToGps

logger = _utils.get_logger(__name__)

EXAMPLES = {
    "Get data for GW150914": " ".join((  # noqa: FLY002
        "gwpy-rds",
        "1126259462",
        "1126259522",
        "H1:GWOSC-4KHZ_R1_STRAIN",
        "L1:GWOSC-4KHZ_R1_STRAIN",
        "-o gw150914.gwf",
        "-O version=4",
    )),
}

DOC_VERSION = "latest" if "dev" in __version__ else __version__
DOC_EPILOG = rf"""
For more information, see the online documentation at:

https://gwpy.readthedocs.io/en/{DOC_VERSION}/tools/rds/
"""


def create_rds(
    channels: Sequence[str],
    start: SupportsToGps,
    end: SupportsToGps,
    outfile: Path,
    source: Sequence[str] | str | None = None,
    write_format: str | None = None,
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

    write_format : `str`, optional
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
        format=write_format,
        overwrite=True,
    )
    logger.info("Wrote %s", outfile)


# -- Docstring helpers ---------------

def _format_write_formats(epilog: str) -> tuple[str, list[dict[str, str]]]:
    """Document the list available formats for writing `TimeSeriesDict`."""
    # Get the list of available formats
    formats = [
        fmt["Format"]
        for fmt in default_registry.get_formats(TimeSeriesDict, readwrite="write")
    ]

    # Format the documentation for manual or help output
    formats_header = "Supported -f/--format values:"
    formats_lines = [formats_header, ""]
    man = _utils.is_manpage()
    for fmt in formats:
        if man:
            formats_lines.extend((
                r".IP \[bu]",
                fr"\fB{fmt}\fR",
            ))
        else:
           formats_lines.append(f"  - {fmt}")
    formats_doc = os.linesep.join(formats_lines)

    # Attach to epilog or manpage sections
    manpage_sections = []
    if man:
        manpage_sections.append(
            {
                "heading": "data formats",
                "content": formats_doc,
            },
        )
        return epilog, manpage_sections
    return formats_doc + os.linesep * 2 + epilog, []


# -- Command line interface ----------

def create_parser() -> ArgumentParser:
    """Create an `argparse.ArgumentParser` for this tool."""
    # Format the help message, including data sources
    epilog, manpage_sections = _format_write_formats(DOC_EPILOG)

    # Create parser
    parser = _utils.ArgumentParser(
        description=__doc__,
        epilog=epilog,
        examples=EXAMPLES,
        prog="gwpy-rds",
        manpage=manpage_sections,
    )
    parser.add_argument(
        "start",
        type=to_gps,
        help=(
            "Start time of data request. "
            "Can be specified as a GPS time, date/time string, or relative time; "
            "please ensure that date strings containing spaces are quoted."
        ),
    )
    parser.add_argument(
        "end",
        type=to_gps,
        help=(
            "End time of data request. "
            "Can be specified as a GPS time, date/time string, or relative time; "
            "please ensure that date strings containing spaces are quoted."
        ),
    )
    parser.add_argument(
        "channels",
        nargs="+",
        metavar="channel|ifo",
        help=(
            "Data channel or IFO to request; can be specified multiple times. "
            "An IFO prefix (e.g., 'H1') can be passed to request public strain "
            "data from GWOSC."
        ),
    )
    parser.add_argument(
        "-g",
        "--source",
        action="append",
        help=(
            "Source from which to get data. "
            "See `help(TimeSeries.get)` for documentation on the "
            "supported sources."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="gwpy-rds.h5",
        type=Path,
        help="Output file in which to write data.",
    )
    parser.add_argument(
        "-f",
        "--format",
        help=(
            "Format in which to write output data. "
            "Default is inferred from -o/--output-file. "
            "See below or `help(TimeSeriesDict.write)` for supported formats."
        ),
    )
    parser.add_argument(
        "-O",
        "--option",
        action="append",
        default=None,
        metavar="key=value",
        help="Additional options to pass to the TimeSeriesDict.get.",
    )
    parser.add_argument(
        "-L",
        "--log-name",
        default="gwpy",
        help=(
            "Name of the logger to configure for verbose output; "
            "use 'root' to enable logging in all modules."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity; pass once for INFO, twice for DEBUG.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=__version__,
        help="Show the version number and exit.",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    """Run the command line tool to create a reduced data set (RDS)."""
    # Parse the command line arguments
    parser = create_parser()
    opts = parser.parse_args(args=args)

    # Init verbose logging
    _utils.init_verbose_logging(opts.log_name, opts.verbose)

    # Parse additional options
    kwargs = _utils.parse_options_dict(opts.option or [])

    # Create the RDS
    create_rds(
        opts.channels,
        opts.start,
        opts.end,
        opts.output_file,
        opts.source,
        write_format=opts.format,
        **kwargs,
    )


if __name__ == "__main__":
    main()
