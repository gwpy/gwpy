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
import os
import textwrap
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
    from collections.abc import (
        Iterable,
        Mapping,
    )
    from logging import Logger
    from typing import TypeVar

    _ActionT = TypeVar("_ActionT", bound=Action)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def is_manpage() -> bool:
    """Return `True` if being called to generate a man page."""
    return any(
        "argparse_manpage" in frame.filename
        for frame in inspect.stack()
    )


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


def parse_option(value: str) -> tuple[str, bool | float | str | None]:
    """Parse a key=value option string into a key and value tuple.

    This function attempts to convert the value to a boolean or numeric
    type if possible; otherwise, it is returned as a string.

    Parameters
    ----------
    value : str
        The option string in the format "key=value".

    Returns
    -------
    key, value: tuple[str, bool | float | str | None]
        The parsed key and value.

    Examples
    --------
    >>> parse_option("threshold=0.5")
    ('threshold', 0.5)
    >>> parse_option("enable_feature=true")
    ('enable_feature', True)
    >>> parse_option("name=example")
    ('name', 'example')
    """
    key, val = value.split("=", 1)

    # Handle boolean and None
    if (vall := val.lower()) in ("true", "yes"):
        return key, True
    if vall in ("false", "no"):
        return key, False
    if vall == "none":
        return key, None

    # Handle numeric
    try:
        val = float(val)
    except ValueError:
        # String
        return key, val
    if val.is_integer():
        val = int(val)
    return key, val


def parse_options_dict(options: list[str]) -> dict[str, bool | float | str | None]:
    """Parse a list of key=value option strings into a dictionary.

    Parameters
    ----------
    options : list[str]
        A list of option strings in the format "key=value".

    Returns
    -------
    dict[str, str | bool | int | float]
        A dictionary of parsed key-value pairs.

    Examples
    --------
    >>> parse_options_dict(["threshold=0.5", "enable_feature=true", "name=example"])
    {'threshold': 0.5, 'enable_feature': True, 'name': 'example'}
    """
    out: dict[str, bool | float | str | None ] = {}
    for optstr in options:
        key, val = parse_option(optstr)
        out[key] = val
    return out


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
        # Extract docstring parameters
        description = kwargs.pop("description", None)
        epilog = kwargs.pop("epilog", None)
        examples: Mapping[str, str | list[str]] | None = kwargs.pop("examples", {})
        manpage: list[dict[str, str]] = kwargs.pop("manpage", [])

        # Format the docstring
        if description is not None:
            description, epilog, manpage_sections = (
                self._compile_docstring(
                    description,
                    epilog,
                    examples,
                )
            )
            manpage.extend(manpage_sections)

        # Initialize the base class
        kwargs.setdefault("formatter_class", HelpFormatter)
        super().__init__(
            description=description,
            epilog=epilog,
            **kwargs,
        )

        # Rename the argument groups to use title case
        self._positionals.title = "Positional arguments"
        self._optionals.title = "Options"

        # Add manpage generation if requested
        self._manpage = manpage

    def _format_examples(self, examples: Mapping[str, str | list[str]]) -> str:
        """Format examples for the docstring."""
        # Manpage format
        if is_manpage():
            lines: list[str] = []
            for desc, cmd in examples.items():
                if isinstance(cmd, str):
                    cmd = cmd.strip().splitlines()
                lines.extend((
                    r".IP \[bu]",
                    fr"\fB{desc}:\fR",
                    ".sp",
                    r".RS 4",
                    ".nf",
                    *(f"$ {line}" for line in cmd),
                    ".fi",
                    r".RE",
                ))
            return os.linesep.join(lines)
        # Argparse help format
        indnt = "  "
        lines = ["Examples:"]
        for desc, cmd in examples.items():
            lines.extend((
                "",
                textwrap.indent(f"{desc}:", indnt),
                "",
                textwrap.indent(f"$ {cmd}", indnt * 2),
            ))
        return os.linesep.join(lines)

    def _compile_docstring(
        self,
        description: str,
        epilog: str | None,
        examples: Mapping[str, str | list[str]] | None,
    ) -> tuple[str, str | None, list[dict[str, str]]]:
        """Compile the docstring for this tool, either for a manual page, or --help."""
        examples_doc = self._format_examples(examples or {})
        if examples and is_manpage():
            if epilog:
                description += os.linesep * 2 + epilog
            manpage_sections = [
                {
                    "heading": "examples",
                    "content": examples_doc,
                },
            ]
            return description, "", manpage_sections
        if examples and epilog:
            epilog = examples_doc + os.linesep * 2 + epilog
        elif examples:
            epilog = examples_doc
        return description, epilog, []

    def _add_action(self, action: _ActionT) -> _ActionT:
        """Add an action to the parser, combining groups for manpages."""
        if is_manpage():
            return self._optionals._add_action(action)
        return super()._add_action(action)
