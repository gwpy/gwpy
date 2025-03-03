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

The `open_remote_file` function also handles (or tries to handle) Pelican
URLs, via a hand-off to the function of the same name in `gwpy.io.pelican`.
"""

from __future__ import annotations

import typing

from astropy.utils.data import get_readable_fileobj

from ..utils.env import bool_env
from . import pelican as io_pelican

if typing.TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from typing import BinaryIO


def open_remote_file(
    url: str,
    *,
    cache: bool | None = None,
    encoding: str = "binary",
    **kwargs,
) -> AbstractContextManager[BinaryIO]:
    """Download a file and open it.

    This function is a wrapper around `astropy.utils.data.get_readable_fileobj`
    with the following customisations:

    - Default to ``cache=True`` if the ``GWPY_CACHE`` environment variable
      is set to something 'truthy'.

    - If the URL looks like a |Pelican|_ URL, hand off to a dedicated
      Pelican download wrapper to attempt the remote access.

    Parameters
    ----------
    url : `str`, file-like
        The name of the resource to access. Can be a local path, or a
        `file://` URL, or any remote URL supported by Astropy.

    cache : `bool`
        Whether to cache the contents of remote URLs.
        Default is `True` if the ``GWPY_CACHE`` environment variable
        is set to something 'truthy'.

    encoding : `str`, optional
        When ``'binary'`` (default), returns a file-like object where its
        ``read`` method returns `bytes` objects.

        When `None`, returns a file-like object with a
        ``read`` method that returns `str` (``unicode``) objects, using
        `locale.getpreferredencoding` as an encoding.  This matches
        the default behavior of the built-in `open` when no ``mode``
        argument is provided.

        When another string, it is the name of an encoding, and the
        file-like object's ``read`` method will return `str` (``unicode``)
        objects, decoded from binary using the given encoding.

    kwargs
        All other positional and keyword arguments are passed directly
        to `astropy.utils.data.get_readable_fileobj`.

    Returns
    -------
    file : file-like
        The file opened in binary format.

    See also
    --------
    astropy.utils.data.get_readable_fileobj
        For details of the underlying downloader.

    gwpy.io.pelican.open_remote_file
        For details of the Pelican-aware download wrapper.
    """
    # enable caching by default based on environment
    if cache is None:
        cache = bool_env("GWPY_CACHE", False)

    # if given a Pelican URL hand off to the Pelican-aware wrapper
    if io_pelican.is_pelican_url(url):
        return io_pelican.open_remote_file(
            url,
            cache=cache,
            encoding=encoding,
            **kwargs,
        )

    # download the file
    # note: get_readable_fileobj is a context manager, so doesn't execute here
    return get_readable_fileobj(
        url,
        cache=cache,
        encoding=encoding,
        **kwargs,
    )
