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

"""Utility module to manage SciTokens for authorised data access.

The functions in this module require the following optional extra
packages:

- |htgettoken|_
- :doc:`igwn-auth-utils <igwn-auth-utils:index>`
- :doc:`requests-scitokens <requests-scitokens:index>`
- :doc:`scitokens <scitokens:index>`
"""

from __future__ import annotations

import logging
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from contextlib import AbstractContextManager

    from scitokens import SciToken

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

log = logging.getLogger(__name__)

__all__ = [
    "add_http_authorization_header",
    "find_token",
    "get_scitoken",
]


# -- find an existing token

def find_token(
    audience: str,
    scope: str,
    **kwargs,
) -> SciToken:
    """Find and load a `Scitoken` for the given ``audience`` and ``scope``.

    Parameters
    ----------
    audience : `str`
        The required value for the ``aud`` claim.

    scope : `str`
        The required value for the ``scope`` claim. Can be a space-separated
        list of multiple scopes, _all_ of which must be matched by the token.

    kwargs
        All other keyword arguments are passed to
        `igwn_auth_utils.find_scitoken`.

    Returns
    -------
    token : `scitokens.SciToken`
        The token, if found, that matches the required claims.

    Raises
    ------
    igwn_auth_utils.IgwnAuthError
        If no valid token can be found.

    See Also
    --------
    igwn_auth_utils.find_scitoken
        For details on acceptable keyword arguments, and how tokens are
        discovered.

    gwpy.io.scitoken.get_scitoken
        To acquire a new token from the token issuer.
    """
    from igwn_auth_utils import find_scitoken
    return find_scitoken(
        audience,
        scope,
        **kwargs,
    )


# -- get a token

def _format_argv(**kwargs) -> list[str]:
    """Format arguments for ``htgettoken``."""
    args = []
    for key, value in kwargs.items():
        arg = f"--{key}"
        if value is False:  # disabled
            continue
        if value in (True, None):
            args.append(arg)
        else:
            args.extend((arg, str(value)))
    return args


def get_scitoken(
    *args: str,
    minsecs: float = 60,
    quiet: bool = True,
    **kwargs,
) -> SciToken:
    """Get a new `SciToken` using |htgettoken|_.

    Parameters
    ----------
    args
        All positional arguments are passed as arguments to
        `htgettoken.main`.

    minsecs : `float`, optional
        The minimum remaining lifetime to reuse an existing bearer token.

    quiet : `bool`, optional
        If `True`, supress output from `htgettoken`.

    kwargs
        All ``key: value`` keyword arguments (including ``minsecs`` and
        ``quiet``) are passed as ``--key=value`` options to
        `htgettoken.main`. Keywords with the value `True` are passed simply
        as ``--key``, while those with the value `False` are omitted.

    Returns
    -------
    token : `scitokens.SciToken`
        The new scitoken acquired by `htgettoken`.

    See Also
    --------
    htgettoken
    """
    import htgettoken
    from scitokens import SciToken

    if not sys.stdout.isatty():
        # don't prompt if we can't get a response
        kwargs.setdefault("nooidc", True)

    # parse output file if given
    outfile = kwargs.pop("outfile", None)

    # if not given, use a temporary directory
    ctx: AbstractContextManager
    if outfile is None:
        ctx = tempfile.TemporaryDirectory()
        filename = "test.token"
    else:
        outfile = Path(outfile)
        filename = outfile.name
        ctx = nullcontext(str(outfile.parent))

    # get token in a temporary directory
    with ctx as tmpdir:
        path = Path(tmpdir) / filename
        kwargs.setdefault("outfile", path)
        argv = list(args) + _format_argv(
            minsecs=minsecs,
            quiet=quiet,
            **kwargs,
        )
        try:
            htgettoken.main(argv)
        except SystemExit as exc:  # bad args
            msg = "htgettoken failed, see full traceback for details"
            raise RuntimeError(msg) from exc
        return SciToken.deserialize(path.read_text().strip())


# -- HTTP token handling -------------

def add_http_authorization_header(
    headers: MutableMapping,
    *,
    token: SciToken | None = None,
    error: bool = True,
    url: str | None = None,
    audience : str | None = None,
    create: bool = True,
    **claims,
) -> None:
    """Try and generate an HTTP ``Authorization`` header for a SciToken.

    This function tries to find an existing token that matches the required
    claims, otherwise will attempt to create (acquire) a new token.

    That token is then serialised to create the argument for an
    ``Authorization`` header using the ``Bearer`` scheme.

    Parameters
    ----------
    headers : `dict`
        The `dict` in which to store the ``Authorization`` header.

    url : `str`, optional
        The URL for which to attempt to find an existing token (to
        match the ``audience`` claim).

    error : `bool`, optional
        If `True` raise exceptions if tokens cannot be found or generated.

    token : `scitokens.SciToken`, optional
        A deserialised token to use for the header.

    audience : `str`, optional
        The required value for the ``aud`` claim.

    create : `bool`, optional
        If `True`, attempt to create (acquire) a new token if an existing
        token is not found.

    claims
        Other keyword arguments are passed as claims to
        `requests_scitokens.HTTPSciTokenAuth.find_token` or
        `gwpy.io.scitokens.get_scitoken`.

    See Also
    --------
    requests_scitokens.HTTPSciTokenAuth.find_token
        For details on how tokens are discovered.

    gwpy.io.scitokens.get_scitoken
        For details on how tokens are created (acquired).
    """
    error_obj: Exception | None = None
    error_types = (
        ImportError,
        RuntimeError,
        ValueError,
    )

    try:
        from requests_scitokens import HTTPSciTokenAuth
    except ImportError as exc:
        if error:
            raise
        log.debug("Cannot add SciToken Authorization header: %s", str(exc))
        return

    # construct the auth handler
    auth = HTTPSciTokenAuth(token=token, audience=audience)

    # try and find a token
    if auth.token is None:
        claims.setdefault("scope", None)
        # try and discover a token
        try:
            auth.token = auth.find_token(
                url=url,
                find_func=find_token,
                **claims,
            )
        except error_types as exc:
            error_obj = exc

    # try and create a token
    if auth.token is None and create:
        # try and generate a new token
        try:
            auth.token = get_scitoken(nooidc=True, **claims)
        except error_types as exc:
            error_obj = exc

    # if we got a token, serialize it and return
    if auth.token:
        headers["Authorization"] = auth.generate_bearer_header()
        log.debug("Added Authorization header with SciToken for %s", url)
        return

    # handle failure
    errmsg = "failed to identify a valid SciToken"
    if not error:
        log.debug("%s: %s", errmsg, str(error_obj))
        return
    raise RuntimeError(errmsg) from error_obj
