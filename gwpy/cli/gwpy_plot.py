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

"""Generate plots of GW observatory data using GWpy."""

from __future__ import annotations

import logging
import os
import sys
import time
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawTextHelpFormatter,
)
from typing import TYPE_CHECKING

from matplotlib import use

from .. import __version__
from ..log import init_logger
from . import PRODUCTS

if TYPE_CHECKING:
    from argparse import (
        Action,
        Namespace,
        _MutuallyExclusiveGroup,
    )
    from collections.abc import Iterable

__author__ = "Joseph Areeda <joseph.areeda@ligo.org>"

if __name__ == "__main__":
    if __spec__ is None:
        _log_name = "gwpy.cli.gwpy_plot"
    else:
        _log_name = __spec__.name
else:
    _log_name = __name__
logger = logging.getLogger(_log_name)

# if launched from a terminal with no display
# Must be done before modules like pyplot are imported
if len(os.getenv("DISPLAY", "")) == 0:
    use("Agg")

INTERACTIVE = hasattr(sys, "ps1")

EPILOG = f"""
Examples:

    $ gwpy-plot timeseries --chan H1:GDS-CALIB_STRAIN --start 1126259457

    $ gwpy-plot spectrum --chan H1:GDS-CALIB_STRAIN L1:GDS-CALIB_STRAIN --chan V1:Hrec_hoft_16384Hz --start 1187008866 --duration 32 --xmin 10 --xmax 4000

    $ gwpy-plot coherencegram --chan H1:GDS-CALIB_STRAIN H1:PEM-CS_ACC_PSL_PERISCOPE_X_DQ --start 1126260017 --duration 600

Written by {__author__}.
Report bugs to https://gitlab.com/gwpy/gwpy/-/issues/.
"""  # noqa: E501


def _init_logging(verbosity: int) -> None:
    """Set up logging."""
    # If user did not specify verbosity, don't change anything;
    # this allows the logging level to be configured by other means,
    # e.g. a config file or environment variable.
    if not verbosity:
        return
    # Otherwise, set the level for the gwpy logger based on verbosity
    level = max(3 - verbosity, 0) * 10
    init_logger("gwpy", level=level)


# -- init command line ---------------

class HelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    """Custom help formatter for `gwpy-plot`."""

    def _format_usage(
        self,
        usage: str | None,
        actions: Iterable[Action],
        groups: Iterable[_MutuallyExclusiveGroup],
        prefix: str | None,
    ) -> str:
        """Format the usage string for the help message."""
        if prefix is None:
            prefix = "Usage: "
        return super()._format_usage(
            usage,
            actions,
            groups,
            prefix,
        )


class _ArgumentParser(ArgumentParser):
    """Custom argument parser for `gwpy-plot`."""

    def __init__(self, **kwargs) -> None:
        """Initialize the argument parser."""
        super().__init__(**kwargs)
        self._positionals.title = "Positional arguments"
        self._optionals.title = "Options"


def create_parser() -> _ArgumentParser:
    """Create the command line argument parser for `gwpy-plot`."""
    parser = _ArgumentParser(
        description=__doc__,
        formatter_class=HelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=__version__,
        help="show program's version number and exit",
    )

    # set the argument parser to act as the parent
    parentparser = _ArgumentParser(add_help=False)
    parentparser._optionals.title = "Verbosity options"
    parentparser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase verbose output",
    )
    parentparser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="show only fatal errors",
    )

    # subparsers are dependent on which action is chosen
    subparsers = parser.add_subparsers(
        dest="mode", title="Actions",
        description="Select one of the following actions:")

    # Add the subparsers for each plot product
    for product, product_class in PRODUCTS.items():
        doc = product_class.__doc__ or ""
        subparser = subparsers.add_parser(
            product,
            help=doc.strip().split("\n", maxsplit=1)[0],
            parents=[parentparser],
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        product_class.init_cli(subparser)

    return parser


def parse_command_line(args: list[str] | None = None) -> Namespace:
    """Parse the command line arguments and return the parsed arguments."""
    parser = create_parser()
    return parser.parse_args(args=args)


# -- run -----------------------------

def main(args: list[str] | None = None) -> int:
    """Run gwpy-plot.

    Returns the relevant exit code, that can be passed to :func:`sys.exit`.
    """
    start = time.time()

    # parse the command line
    opts = parse_command_line(args=args)
    _init_logging(opts.verbose)
    logger.info("-- Welcome to gwpy-plot v%s --", __version__)

    # create a product object
    prod = PRODUCTS[opts.mode](opts)
    logger.debug("%s created", prod.action)

    # log how long it took us to get here
    setup_time = time.time() - start
    logger.debug("Setup time %.1fs", setup_time)

    # -- generate the plot
    prod.run()

    # overload the current namespace for interactive (i)python users
    if INTERACTIVE:
        # import pyplot so that user has access to it
        from matplotlib import pyplot as plt  # noqa: F401
        plot = prod.plot
        # pull raw data and plotted results from product for their use
        timeseries = prod.timeseries  # noqa: F841
        ax = plot.gca()  # noqa: F841
        print("Raw data is in 'timeseries', plot product is in 'prod'")  # noqa: T201

    run_time = time.time() - start
    logger.debug("Program run time: %.1fs", run_time)
    if prod.got_error:
        # Make sure when running batch they can test for error
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
