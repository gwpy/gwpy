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

import logging
from pathlib import PureWindowsPath
from typing import (
    TYPE_CHECKING,
    overload,
)
from unittest import mock

from astropy.utils import data as astropy_data
from urllib3.util import parse_url

from ..utils.env import bool_env
from . import pelican as io_pelican

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from typing import (
        IO,
        BinaryIO,
        Literal,
        TextIO,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = [
    "download_file",
    "is_remote",
    "open_remote_file",
]

log = logging.getLogger(__name__)


# -- Patch astropy to log downloads

_download_file_from_source = astropy_data._download_file_from_source

def _logging_download_file_from_source(url: str, *args, **kwargs) -> str:
    """Patch for astropy's _download_file_from_source to log the URL.

    This is helpful for debugging downloads with multiple sources or
    redirects.
    """
    log.debug("Downloading file from %s", url)
    out = _download_file_from_source(url, *args, **kwargs)
    log.debug("Downloaded file from %s to %s", url, out)
    return out


def _mock_download_file_from_source() -> AbstractContextManager:
    """Context manager to mock astropy's internal download function to log URLs."""
    return mock.patch(
        "astropy.utils.data._download_file_from_source",
        _logging_download_file_from_source,
    )


# -- Remote file handling

def is_remote(url: str) -> bool:
    r"""Return `True` if ``url`` points at a remote URL.

    This function just inspects the 'scheme' of the URL, if present,
    and returns `True` if the scheme *isn't* ``"file"``.

    On Windows, any ``url`` that includes a drive assignment, e.g:
    ``"C:\\Users\\me\\data.txt"`` will return `False`.

    Parameters
    ----------
    url : `str`, `pathlib.Path`
        The URL to inspect.

    Returns
    -------
    remote : `bool`
        `True` if the URL looks like it is remote (would need a
        network connection for access), otherwise `False`.
    """
    if PureWindowsPath(url).drive:  # windows drive path
        return False
    # check the URL scheme
    return parse_url(url).scheme not in (None, "file")


@overload
def open_remote_file(
    url: str,
    *,
    cache: bool | None = None,
    show_progress: bool = False,
    encoding: Literal["binary"] = "binary",
    **kwargs,
) -> AbstractContextManager[BinaryIO]: ...

@overload
def open_remote_file(
    url: str,
    *,
    cache: bool | None = None,
    show_progress: bool = False,
    encoding: str | None,
    **kwargs,
) -> AbstractContextManager[TextIO]: ...

def open_remote_file(
    url: str,
    *,
    cache: bool | None = None,
    show_progress: bool = False,
    encoding: str | None = "binary",
    **kwargs,
) -> AbstractContextManager[IO]:
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

    show_progress: `bool`, optional
        Print verbose progress information to the screen.

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

    See Also
    --------
    astropy.utils.data.get_readable_fileobj
        For details of the underlying downloader.

    gwpy.io.pelican.open_remote_file
        For details of the Pelican-aware download wrapper.
    """
    return _handle_remote_file(
        url,
        mode="open",
        cache=cache,
        show_progress=show_progress,
        encoding=encoding,
        **kwargs,
    )


def download_file(
    url: str,
    *,
    cache: bool | None = None,
    show_progress: bool = False,
    **kwargs,
) -> str:
    """Download a file from a URL and optionally cache the result.

    This function is a wrapper around `astropy.utils.data.download_file`
    with the following customisations:

    - File URLs (``file://``) and plain paths (without a URL scheme)
      are returned unmodified **without** being downloaded or cached.

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

    show_progress: `bool`, optional
        Print verbose progress information to the screen.

    kwargs
        All other positional and keyword arguments are passed directly
        to `astropy.utils.data.get_readable_fileobj`.

    Returns
    -------
    load_path : str
        The local path that the file was downloaded to.

    See Also
    --------
    astropy.utils.data.download_file
        For details of the underlying downloader.

    gwpy.io.pelican.open_remote_file
        For details of the Pelican-aware download wrapper.
    """
    return _handle_remote_file(
        url,
        mode="download",
        cache=cache,
        show_progress=show_progress,
        **kwargs,
    )


@overload
def _handle_remote_file(
    url: str,
    mode: Literal["open"],
    *,
    cache: bool | None = None,
    **kwargs,
) -> AbstractContextManager[BinaryIO]: ...

@overload
def _handle_remote_file(
    url: str,
    mode: Literal["download"],
    *,
    cache: bool | None = None,
    **kwargs,
) -> str: ...

def _handle_remote_file(
    url: str,
    mode: Literal["open", "download"] = "open",
    *,
    cache: bool | None = None,
    **kwargs,
) -> AbstractContextManager[IO] | str:
    """Handle getting a remote file, including Pelican URLs.

    This just abstracts out the common code used by both
    `open_remote_file` and `download_file`.
    """
    # enable caching by default based on environment
    if mode == "download" and not is_remote(url):
        return url

    if mode == "download":
        get = astropy_data.download_file
        pelican_get = io_pelican.download_file
    else:
        get = astropy_data.get_readable_fileobj
        pelican_get = io_pelican.open_remote_file  # type: ignore[assignment]

    if cache is None:
        cache = bool_env("GWPY_CACHE", default=False)

    with _mock_download_file_from_source():
        # if given a Pelican URL hand off to the Pelican-aware wrapper
        if io_pelican.is_pelican_url(url):
            return pelican_get(
                url,
                cache=cache,
                **kwargs,
            )

        # download the file
        # note: get_readable_fileobj is a context manager, so doesn't execute here
        return get(
            url,
            cache=cache,
            **kwargs,
        )
