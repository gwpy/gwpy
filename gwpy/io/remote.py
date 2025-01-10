# Copyright (C) Cardiff University (2024-)
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

"""Remote file I/O utilities for GWpy.

This module provides the `open_remote_file` function, which is applied
automatically when reading all files with the `gwpy.io.registry.UnifiedRead`
registry reader.
This means that operations like `TimeSeries.read("http://...")` can access
remote data.
"""

from __future__ import annotations

import typing

from astropy.utils.data import get_readable_fileobj

from ..utils.env import bool_env


def open_remote_file(
    url: str,
    *args,
    cache: bool | None = None,
    **kwargs,
) -> typing.BinaryIO:
    """Download a file and open it.

    This function is just a thin wrapper around
    `astropy.utils.data.get_readable_fileobj` with some modified defaults.

    In addition to enabling ``cache`` based on the environment, this function
    also copies any ``headers`` into the ``fsspec_kwargs`` by default to
    simplify passing headers to fsspec when downloading over HTTP.

    Parameters
    ----------
    url : `str`, file-like
        The name of the resource to access. Can be a local path, or a
        `file://` URL, or any remote URL supported by Astropy.

    cache : `bool`
        Whether to cache the contents of remote URLs.
        Default is `True` if the ``GWPY_CACHE`` environment variable
        is set to something 'truthy'.

    args, kwargs
        All other positional and keyword arguments are passed directly
        to `astropy.utils.data.get_readable_fileobj`.

    Returns
    -------
    file : file-like
        The file opened in binary format.

    See also
    --------
    astropy.utils.data.get_readable_fileobj
    """
    if cache is None:
        cache = bool_env("GWPY_CACHE", False)
    if "headers" in kwargs:
        kwargs.setdefault("fsspec_kwargs", {"headers": kwargs.pop("headers")})
    return get_readable_fileobj(
        url,
        cache=cache,
        encoding="binary",
        **kwargs,
    )
