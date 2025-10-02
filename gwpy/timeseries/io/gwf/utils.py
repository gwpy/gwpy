# Copyright (c) 2014-2017 Louisiana State University
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

"""GWF I/O utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeVar

    _T = TypeVar("_T")


def _channel_dict_kwarg(
    value: dict[str, _T] | list[_T] | _T | None,
    channels: list[str],
    expected_type: type[_T] | None = None,
    varname: str = "value list",
) -> dict[str, _T | None]:
    """Format the given kwarg value in a dict with one value per channel.

    Parameters
    ----------
    value : any type
        Keyword argument value as given by user.

    channels : `list` of `str`
        List of channels being read.

    expected_type : `type`, `types.UnionType`, optional
        Valid type (or union of types) for value (or items in an iterable value).

    varname : `str`, optional
        The name of the keyword variable being parsed.
        This is only used to assist in error reporting.

    Returns
    -------
    dict : `dict`
        `dict` of values, one value per channel key
    """
    # if the value given is of the right type already use it for everything
    if (
        expected_type is not None
        and isinstance(value, expected_type)
        # check whether the expected type is a list or tuple, and whether
        # the input value is a list|tuple of lists|tuples
        and not (
            issubclass(expected_type, list | tuple)
            and len(value)
            and isinstance(value[0], expected_type)
        )
    ):
        # copy value for all channels
        return dict.fromkeys(channels, value)

    # zip list of values with list of channels
    if isinstance(value, list | tuple):
        try:
            return dict(zip(channels, value, strict=True))
        except ValueError as exc:
            exc.args = (
                str(exc).replace(
                    "argument 2", varname,
                ).replace("argument 1", "channels list"),
            )
            raise

    # return subset of value dict for channels
    if isinstance(value, dict):
        return {c: value[c] for c in set(channels).intersection(value)}

    # repeat value for all channels
    return dict.fromkeys(channels, value)
