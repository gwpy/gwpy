# Copyright (c) 2025 Cardiff University
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

"""Utilities for command-line tools for GWpy."""

from __future__ import annotations

import argparse
import inspect
import logging
from argparse import (
    ArgumentDefaultsHelpFormatter,
    RawDescriptionHelpFormatter,
)
from typing import TYPE_CHECKING

from ..log import init_logger

if TYPE_CHECKING:
    from argparse import (
        Action,
        _MutuallyExclusiveGroup,
    )
    from collections.abc import Iterable
    from logging import Logger

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def get_logger(name: str, fallback: str = "__main__") -> logging.Logger:
    """Get a logger for the given name."""
    if name == "__main__":
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        spec = getattr(mod, "__spec__", None)
        if spec is not None:
            name = spec.name
        else:
            name = fallback
    return logging.getLogger(name)


def init_verbose_logging(
    name: str = "gwpy",
    verbosity: int = 0,
) -> Logger:
    """Configure logging based on a verbosity count."""
    # If user did not specify verbosity, don't change anything;
    # this allows the logging level to be configured by other means,
    # e.g. a config file or environment variable.
    if not verbosity:
        return init_logger(name)

    # Otherwise, set the level for the gwpy logger based on verbosity
    level = max(3 - verbosity, 0) * 10
    return init_logger(name, level=level)


class HelpFormatter(
    ArgumentDefaultsHelpFormatter,
    RawDescriptionHelpFormatter,
):
    """Custom help formatter for GWpy tools."""

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


class ArgumentParser(argparse.ArgumentParser):
    """Custom argument parser for GWpy tools."""

    def __init__(self, **kwargs) -> None:
        """Initialize the argument parser."""
        # Options for argparse-manpage
        manpage: list[dict[str, str]] = kwargs.pop("manpage", [])

        # Initialize the base class
        kwargs.setdefault("formatter_class", HelpFormatter)
        super().__init__(**kwargs)

        # Rename the argument groups to use title case
        self._positionals.title = "Positional arguments"
        self._optionals.title = "Options"

        # Add manpage generation if requested
        self._manpage = manpage
