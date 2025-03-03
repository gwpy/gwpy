# Copyright (C) Cardiff University (2025)
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
import typing
from contextlib import contextmanager

try:
    from coloredlogs import (
        ColoredFormatter,
        terminal_supports_colors,
    )
except ImportError:
    def terminal_supports_colors(stream: IO) -> bool:   # noqa: ARG001
        """Return `False` to indicate that colours are unsupported."""
        return False

if typing.TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import IO

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

DEFAULT_FORMAT = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S.%f%z"


def get_default_loglevel() -> int:
    """Return the default log level by inspecting the ``GWPY_LOGLEVEL`` variable.

    If ``GWPY_LOGLEVEL`` is not set, `logging.NOTSET` is returned.

    Returns
    -------
    level : `int`
        The log level to use by default for loggers.

    Examples
    --------
    >>> os.environ["GWPY_LOGLEVEL"] = "DEBUG"
    >>> get_default_loglevel()
    10
    >>> os.environ["GWPY_LOGLEVEL"] = logging.INFO
    20
    """
    try:
        loglevel = os.environ["GWPY_LOGLEVEL"]
    except KeyError:
        return logging.NOTSET
    if loglevel.isdigit():
        return int(loglevel)
    levels = logging.getLevelNamesMapping()
    return levels[loglevel]


def get_logger(
    name: str,
    level: int | str | None = None,
    *,
    stream: IO = sys.stderr,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str | None = DEFAULT_DATEFMT,
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
        determined from the ``GWPY_LOGLEVEL`` environment variable, or
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
        level = get_default_loglevel()

    # get the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # attach a new handler as needed
    if not logger.hasHandlers():
        handler = logging.StreamHandler(stream)
        if color and terminal_supports_colors(stream):
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


@contextmanager
def logger(
    name: str,
    level: int | str | None,
) -> Iterator[logging.Logger]:
    """Return a `logging.Logger` with the given ``name`` and ``level``.

    The logging level is reset to its previous value after the context manager
    exits.

    Parameters
    ----------
    name : `str`
        The name of the logger.

    level : `int`, `str`, `None`
        The level to set on the logger.
        If `None` is given, the log level is not changed from its current value,
        but will still be reset when the context manager exits.

    Yields
    ------
    logger : `logging.Logger`
        The logger.

    Examples
    --------
    >>> with logger("gwpy.timeseries", "DEBUG") as log:
    ...     log.debug("something interesting")
    """
    logger = logging.getLogger(name)
    current = logger.getEffectiveLevel()
    if level is not None:
        logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(current)
