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

"""I/O utilities for Pelican.

This module requires the :doc:`requests-pelican <requests-pelican:index>` package.

This module handles translation of ``pelican://`` URLs into lists of HTTP URLs
to pass back to `astropy.utils.data.download_file` and friends, and some
SciToken authorisation initialisation.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import (
    cache,
    wraps,
)
from typing import TYPE_CHECKING
from unittest import mock

from astropy.utils import data as astropy_data

from . import scitokens as io_scitokens

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager
    from typing import (
        Any,
        BinaryIO,
        Literal,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = [
    "download_file",
    "is_pelican_url",
    "open_remote_file",
    "query_director",
    "resolve_pelican_url",
]

log = logging.getLogger(__name__)


def _pelicanfs_protocols() -> list[str]:
    """Get list of known Pelican protocols from `pelicanfs`."""
    from pelicanfs.core import PelicanFileSystem
    return [klass.protocol for klass in [
        PelicanFileSystem,
        *PelicanFileSystem.__subclasses__(),
    ]]


def _requests_pelican_protocols() -> list[str]:
    """Get list of known Pelican protocols from `requests-pelican`."""
    from requests_pelican.federation import KNOWN_FEDERATIONS
    return ["pelican", *KNOWN_FEDERATIONS]


@cache
def _pelican_protocols() -> list[str]:
    """Return all known Pelican protocols.

    Needs `pelicanfs` or `requests-pelican`, otherwise returns an empty list.
    """
    for protocol_finder in (
        _pelicanfs_protocols,
        _requests_pelican_protocols,
    ):
        try:
            return protocol_finder()
        except ImportError:
            pass
    return []


@cache
def is_pelican_url(url: str) -> bool:
    """Return `True` if ``url`` looks like a Pelican URL.

    Needs the `pelicanfs` package, otherwise returns `False`.
    """
    try:
        protocols = list(_pelican_protocols())
    except ImportError:
        return False
    return any(url.startswith(f"{proto}://") for proto in protocols)


def query_director(
    url: str,
    **kwargs,
) -> tuple[list[str], bool, dict[str, Any]]:
    """Query a Pelican director for information.

    Parameters
    ----------
    url : `str`
        The Pelican URL of interest.

    kwargs
        All keyword arguments are passed to
        `requests_pelican.director.get_director_response`.

    Returns
    -------
    urls : `list` of `str`
        The list of HTTP URLs that serve the Pelican URL.

    need_auth : `bool`
        `True` if the HTTP URLs require token authorisation.

    auth_kw : `dict`
        Keyword arguments to pass to the
        `add_http_authorization_header` function to find/generate
        the appropriate token.

    See Also
    --------
    requests_pelican.director.get_director_response
    """
    from requests_pelican.director import get_director_response
    from urllib3.util import parse_url

    purl = parse_url(url)
    director = get_director_response(
        str(purl),
        **kwargs,
    )
    auth_kw: dict[str, Any] = {}
    if (
        # directory says we need auth
        (need_auth := director.namespace["require-token"])
        # but this is not an IGWN namespace (tokens are separately issued)
        and not director.namespace["namespace"].startswith("/igwn")
    ):
        for auth in director.auth:
            for key, val in auth.items():
                auth_kw.setdefault(key, []).append(val)
    return (
        director.urls(purl.path),
        need_auth,
        auth_kw,
    )


def resolve_pelican_url(
    url: str,
    **kwargs,
) -> list[str]:
    """Resolve a Pelican URL into one or more HTTP URLs."""
    return query_director(url, **kwargs)[0]


# -- astropy download hackery --------
#
# astropy.utils.data relies on an 'is_url' function to determine
# whether a target URL is an HTTP url. If that returns False,
# get_readable_filobj refuses to download a file, even if `sources`
# is given with a list of real HTTP URLs.
#
# So, we hack around it for Pelican.
#

@wraps(astropy_data.is_url)
def is_url(url: str) -> bool:
    """Return `True` for all Pelican URLs.

    If ``url`` doesn't look like a Pelican URL, hand off to
    `astropy.utils.data.is_url`.
    """
    if is_pelican_url(url):
        return True
    return astropy_data.is_url(url)


def _pelican_is_url_context() -> AbstractContextManager:
    return mock.patch("astropy.utils.data._is_url", is_url)


@contextmanager
def _pelican_download_context(
    reader: AbstractContextManager[BinaryIO],
) -> Iterator[BinaryIO]:
    """Wrap around ``reader`` to mock out `astropy.utils.data.is_url`."""
    with _pelican_is_url_context():  # noqa: SIM117
        with reader as result:
            yield result


def _prepare_download_kwargs(
    url: str,
    federation: str | None,
    cache: bool | Literal["update"],
    kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Prepare the keyword arguments for downloading a file from Pelican."""
    from requests_pelican.pelican import pelican_uri

    url = pelican_uri(url, federation=federation)

    # check if the file is already in the cache
    if cache is True:
        with _pelican_is_url_context():
            need_download = not astropy_data.is_url_in_cache(url)
    else:
        need_download = True

    # if we are going to download the file, query the Pelican director
    # for HTTP(S) sources, and required auth information
    if need_download:
        log.debug("Getting Pelican information from director")

        # query Pelican director for information
        sources, need_auth, auth_kw = query_director(url)

        # resolve URL to real HTTP URLs
        kwargs.setdefault("sources", sources)
        log.debug(
            "Resolved %d HTTP sources for %s",
            len(sources),
            url.rsplit("/", 1)[-1],
        )

        # add auth
        if need_auth:
            io_scitokens.add_http_authorization_header(
                kwargs.setdefault("http_headers", {}),
                url=url,
                error=False,
                create=False,
                **auth_kw,
            )

    return url, kwargs


def open_remote_file(
    url: str,
    *,
    federation: str | None = None,
    cache: bool | Literal["update"] = False,
    show_progress: bool = False,
    **kwargs,
) -> AbstractContextManager[BinaryIO]:
    """Download a remote file from a Pelican federation (and open it).

    This function is a wrapper around `astropy.utils.data.get_readable_fileobj`
    that uses :doc:`requests-pelican <requests-pelican:index>` to resolve the
    actual HTTP(S) URLs and pass those as the ``sources`` keyword to
    :func:`~astropy.utils.data.get_readable_fileobj`.

    Also, if the Pelican director informs that an authorisation token is
    required, this function attempts to an HTTP ``Authorization`` header
    using a locally-discovered `~scitokens.SciToken`
    (requires :doc:`igwn-auth-utils <igwn-auth-utils:index>`).

    Parameters
    ----------
    url : `str`
        The name of the resource to access. Must be a Pelican federation
        URL, either using the ``pelican://`` scheme or a federation
        scheme understood by `requests-pelican` (e.g. ``osdf://``), or
        the ``federation`` keyword must be provided.

    federation: `str`, optional
        The URL of the federation. This is required if ``url`` does not
        include federation information.

    cache : `bool`, ``"update"``, optional
        Whether to cache the contents of remote URLs.

    show_progress: `bool`, optional
        Print verbose progress information to the screen.

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
    """
    url, kwargs = _prepare_download_kwargs(
        url,
        federation,
        cache,
        kwargs,
    )

    # download the file using our 'is_url' hack
    # note: get_readable_fileobj is a context manager, so doesn't execute here
    return _pelican_download_context(astropy_data.get_readable_fileobj(
        url,
        cache=cache,
        show_progress=show_progress,
        **kwargs,
    ))


def download_file(
    url: str,
    *,
    federation: str | None = None,
    cache: bool | Literal["update"] = False,
    show_progress: bool = False,
    **kwargs,
) -> str:
    """Download a remote file from a Pelican federation.

    This function is a wrapper around `astropy.utils.data.download_file`
    that uses :doc:`requests-pelican <requests-pelican:index>` to resolve the
    actual HTTP(S) URLs and pass those as the ``sources`` keyword to
    :func:`~astropy.utils.data.download_file`.

    Also, if the Pelican director informs that an authorisation token is
    required, this function attempts to an HTTP ``Authorization`` header
    using a locally-discovered `~scitokens.SciToken`
    (requires :doc:`igwn-auth-utils <igwn-auth-utils:index>`).

    Parameters
    ----------
    url : `str`
        The name of the resource to access. Must be a Pelican federation
        URL, either using the ``pelican://`` scheme or a federation
        scheme understood by `requests-pelican` (e.g. ``osdf://``), or
        the ``federation`` keyword must be provided.

    federation: `str`, optional
        The URL of the federation. This is required if ``url`` does not
        include federation information.

    cache : `bool`, ``"update"``, optional
        Whether to cache the contents of remote URLs.

    show_progress: `bool`, optional
        Print verbose progress information to the screen.

    kwargs
        All other positional and keyword arguments are passed directly
        to `astropy.utils.data.download_file`.

    Returns
    -------
    file : file-like
        The file opened in binary format.

    See Also
    --------
    astropy.utils.data.download_file
    """
    url, kwargs = _prepare_download_kwargs(
        url,
        federation,
        cache,
        kwargs,
    )

    # download the file using our 'is_url' hack
    with _pelican_is_url_context():
        return astropy_data.download_file(
            url,
            cache=cache,
            show_progress=show_progress,
            **kwargs,
        )
