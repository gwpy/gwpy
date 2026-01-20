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

"""Logging utilities for GWpy."""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

try:
    from coloredlogs import (
        ColoredFormatter,
        terminal_supports_colors as _terminal_supports_colors,
    )
except ImportError:
    def _terminal_supports_colors(stream: IO) -> bool:   # noqa: ARG001
        """Return `False` to indicate that colours are unsupported."""
        return False

if TYPE_CHECKING:
    from typing import IO

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = [
    "DEFAULT_LOG_DATEFMT",
    "DEFAULT_LOG_FORMAT",
    "get_default_level",
    "init_logger",
]

_LOG_LEVELS = logging.getLevelNamesMapping()

#: The default log message format.
#:
#: The format can be overridden by setting the ``GWPY_LOG_FORMAT``
#: environment variable.
DEFAULT_LOG_FORMAT = os.getenv(
    "GWPY_LOG_FORMAT",
    "%(asctime)s:%(name)s:%(levelname)s:%(message)s",
)

#: The default log date format.
#:
#: The format can be overridden by setting the ``GWPY_LOG_DATEFMT``
#: environment variable.
DEFAULT_LOG_DATEFMT = os.getenv(
    "GWPY_LOG_DATEFMT",
    "%Y-%m-%dT%H:%M:%S%z",
)


def get_default_level() -> int:
    """Return the default log level by inspecting the ``GWPY_LOG_LEVEL`` variable.

    If ``GWPY_LOG_LEVEL`` is not set, `logging.NOTSET` is returned.

    Returns
    -------
    level : `int`
        The log level to use by default for loggers.

    Examples
    --------
    >>> os.environ["GWPY_LOG_LEVEL"] = "debug"
    >>> get_default_level()
    10
    >>> os.environ["GWPY_LOG_LEVEL"] = logging.INFO
    >>> get_default_level()
    20
    """
    try:
        level = os.environ["GWPY_LOG_LEVEL"].upper()
    except KeyError:
        return logging.NOTSET
    if level.isdigit():
        return int(level)
    return _LOG_LEVELS[level]


def init_logger(
    name: str,
    level: int | str | None = None,
    *,
    stream: IO = sys.stderr,
    fmt: str = DEFAULT_LOG_FORMAT,
    datefmt: str | None = DEFAULT_LOG_DATEFMT,
    style: str = "%",
    color: bool = True,
) -> logging.Logger:
    """Return the logger with the given name, or create one as needed.

    This function calls `logging.getLogger` to find a logger with the
    specified ``name``, or to create one.

    If the returned logger has no handlers attached already, a new
    `logging.StreamHandler` will be created and attached based
    on ``stream``.

    Parameters
    ----------
    name : `str`
        The name of the logger.

    level : `int`, `str`, optional
        The level to set on the logger.
        If ``level=None`` (default) is given, the logging level will be
        determined from the ``GWPY_LOG_LEVEL`` environment variable, or
        set to `logging.NOTSET`.

    stream : `io.IOBase`, optional
        The stream to write log messages to.

    fmt : `str`, optional
        The message format to use for a new handler.

    datefmt : `str`, optional
        The date format to use for a new handler.

    style : `str`, optional
        The type of ``format`` string.

    color : `bool`, optional
        If `True` (default) try to use `coloredlogs.ColoredFormatter`
        as the default formatter.
        If `False` the default formatter is `logging.Formatter`.

    Returns
    -------
    logger : `logging.Logger`
        A new, or existing, `Logger` instance, with at least one
        `~logging.Handler` attached.

    Notes
    -----
    If ``color=True`` is given (default) the
    :func:`~humanfriendly.terminal.terminal_supports_colors` function is
    called to determine whether the output terminal can support colours.
    If that returns `False`, coloured formatting is not used.

    See Also
    --------
    logging.basicConfig
        For more complete descriptions of some of the keyword arguments.
    """
    # get the default level
    if level is None:
        level = get_default_level()

    # get the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # attach a new handler as needed
    if not logger.hasHandlers():
        handler = logging.StreamHandler(stream)
        if color and _terminal_supports_colors(stream):
            formatter_class = ColoredFormatter
        else:
            formatter_class = logging.Formatter
        handler.setFormatter(formatter_class(
            fmt=fmt,
            datefmt=datefmt,
            style=style,
        ))
        logger.addHandler(handler)
    return logger
