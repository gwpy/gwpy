# Copyright (c) 2013-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Registry for FFT averaging methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...utils.decorators import deprecated_function

if TYPE_CHECKING:
    from collections.abc import Callable

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# registry dict for FFT averaging methods
METHODS: dict[str, Callable] = {}


def _format_name(name: str) -> str:
    return name.lower().replace("-", "_")


def register_method(
    func: Callable,
    name: str | None = None,
    deprecated: bool = False,
) -> str:
    """Register a method of calculating an average spectrogram.

    Parameters
    ----------
    func : `callable`
        function to execute

    name : `str`, optional
        name of the method, defaults to ``func.__name__``

    deprecated : `bool`, optional
        whether this method is deprecated (`True`) or not (`False`)

    Returns
    -------
    name : `str`
        the registered name of the function, which may differ
        pedantically from what was given by the user.
    """
    if name is None:
        name = func.__name__

    # warn about deprecated functions
    if deprecated:
        func = deprecated_function(
            func,
            (
                f"the '{name}' PSD method is deprecated, and will be removed "
                "in a future release, please consider using "
                f"'{name.split('-', 1)[1]}' instead"
            ),
        )

    name = _format_name(name)
    METHODS[name] = func
    return name


def get_method(name: str) -> Callable:
    """Return the PSD method registered with the given name."""
    # find method
    name = _format_name(name)
    try:
        return METHODS[name]
    except KeyError as exc:
        exc.args = f"no PSD method registered with name '{name}'",
        raise
