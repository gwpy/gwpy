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

"""Utility module to initialise Kerberos ticket-granting tickets.

This module provides a lazy-mans python version of the 'kinit'
command-line tool using the python-gssapi library.

See the documentation of the `kinit` function for example usage.
"""


from __future__ import annotations

import getpass
import logging
import os
import re
import subprocess
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING
from unittest import mock

if TYPE_CHECKING:
    from gssapi import (
        Credentials,
        Name,
    )

    from .utils import FileSystemPath

from ..utils.decorators import deprecated_function

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = [
    "kinit",
]

logger = logging.getLogger(__name__)

KLIST = which("klist") or "klist"

try:
    _IPYTHON = __IPYTHON__  # type: ignore[name-defined]
except NameError:
    _IPYTHON = False


# -- exceptions ---------------------------------

class KerberosError(RuntimeError):
    """Kerberos (krb5) operation failed."""



# -- utilities ----------------------------------

def _validate_keytab(
    username: str | None = None,
    realm: str | None = None,
    keytab: FileSystemPath | None = None,
) -> tuple[str | None, str | None]:
    """Get the default keytab and check that a file exists.

    This method reformats the ``username`` into a full principal name, as best
    it can, using the given ``realm`` or one pulled out of the ``keytab``.

    Parameters
    ----------
    username : `str`
        The username for the Kerberos principal (possibly including a realm,
        as given by the user to `kinit()`).

    realm : `str`
        The realm of the Kerberos principal
        (as given by the user to `kinit()`).

    keytab : `str`, `pathlib.Path`
        The path of the Kerberos keytab to use (as given by the user to `kinit()`).

    Returns
    -------
    principal : `str`
        The best-formatted full principal name.

    keytab : `str`
        The keytab path.

    Raises
    ------
    gwpy.io.kerberos.KerberosError
        If the ``username`` and/or ``realm`` conflict with that discovered
        in the ``keytab`` (if given).

    gssapi.exceptions.GSSError
        If ``keytab`` is given but a principal cannot be parsed from it.
    """
    import gssapi

    # if the user gave the keytab, use it
    if keytab:
        strict = True
    # otherwise try the environment, but be lenient
    else:
        keytab = os.environ.get("KRB5_KTNAME", None)
        strict = False

    # no keytab, use the username we were given
    if keytab is None:
        return username, None

    keytab = str(Path(keytab).expanduser())
    try:  # get principal from default keytab
        principal = _keytab_principal(keytab)
        puser, prealm = principal.split("@", 1)
    except (
        gssapi.exceptions.GSSError,
        ValueError,
    ):
        if strict:
            raise
        return username, None

    # user didn't specify a name or a realm, use what we got from the keytab
    if not username and not realm:
        return principal, keytab

    # user _did_ specify username or realm, so if they match the keytab,
    # then we can use it
    try:
        user, urealm = username.split("@", 1)  # type: ignore[union-attr]
    except (
        AttributeError,  # username is None
        ValueError,  # username doesn't contain an '@'
    ):
        user = username
        urealm = realm
    if (
        (user is None or user == puser)
        and (urealm is None or urealm == prealm)
    ):
        return principal, keytab

    if strict:
        message = (
            f"principal '{principal}' from keytab ('{keytab}') doesn't match "
            f"username '{username}'"
        )
        if realm:
            message += f", or realm '{realm}'"
        raise KerberosError(message)

    return username, None


def _check_interactive(
    username: str | None,
    password: str | None,
    keytab: FileSystemPath | None,
) -> None:
    """Check that we can prompt for necessary information.

    This function raises an exception if we need to prompt for
    information but we are not in an interactive session.
    """
    _prompt_username = username is None
    _prompt_password = not keytab and password is None
    if (
        not sys.stdout.isatty()
        and not _IPYTHON
        and (_prompt_username or _prompt_password)
    ):
        msg = (
            "cannot generate Kerberos ticket in a non-interactive session, "
            "please manually create a ticket, or consider using a keytab file"
        )
        raise KerberosError(msg)


def _get_principal(
    username: str | None,
    realm: str | None,
) -> Name:
    """Prompt for and canonicalise the principal name."""
    import gssapi

    if username is None:
        username = input(
            f"Kerberos principal (user@{realm or 'REALM'}): ",
        )
    if "@" not in username and realm:
        username = f"{username}@{realm}"
    principal = gssapi.Name(
        base=username,
        name_type=gssapi.NameType.kerberos_principal,
    )
    try:
        # applies default realm if not given
        return principal.canonicalize(
            gssapi.MechType.kerberos,
        )
    except gssapi.exceptions.GSSError as exc:
        msg = "failed to canonicalize Kerberos principal name, please specify `realm`"
        raise KerberosError(msg) from exc


def _acquire_keytab(
    principal: Name,
    keytab: FileSystemPath,
    ccache: FileSystemPath | None = None,
    lifetime: int | None = None,
) -> Credentials:
    """Acquire a Kerberos TGT using a keytab."""
    import gssapi
    keytab = os.fspath(keytab)
    store: dict[bytes | str, bytes | str] = {
        "client_keytab": keytab,
    }
    if ccache:
        store["ccache"] = os.fspath(ccache)
    with mock.patch.dict("os.environ", {"KRB5_KTNAME": keytab}):
        creds = gssapi.Credentials(
            name=principal,
            store=store,
            usage="initiate",
            lifetime=lifetime,
        )
    creds.inquire()
    return creds


def _acquire_password(
    principal: Name,
    password: str,
    ccache: FileSystemPath | None = None,
    lifetime: int | None = None,
) -> Credentials:
    """Acquire a Kerberos TGT using principal/password."""
    import gssapi
    raw_creds = gssapi.raw.acquire_cred_with_password(
        principal,
        password.encode("utf-8"),
        lifetime=lifetime,
        usage="initiate",
    )
    creds = gssapi.Credentials(base=raw_creds.creds)
    creds.inquire()
    creds.store(
        store={"ccache": os.fspath(ccache)} if ccache else None,
        usage="initiate",
        overwrite=True,
    )
    return creds


def _keytab_principal(
    keytab: FileSystemPath,
) -> str:
    """Return the principal assocated with a Kerberos keytab file."""
    import gssapi
    with mock.patch.dict("os.environ", {"KRB5_KTNAME": os.fspath(keytab)}):
        return str(gssapi.Credentials(usage="accept").name)


def kinit(
    username: str | None = None,
    password: str | None = None,
    realm: str | None = None,
    keytab: FileSystemPath | None = None,
    ccache: FileSystemPath | None = None,
    lifetime: int | None = None,
    krb5ccname: str | None = None,
    *,
    verbose: bool | None = None,
) -> Credentials:
    """Initialise a Kerberos ticket-granting ticket (TGT).

    Parameters
    ----------
    username : `str`, optional
        Name principal for Kerberos credential, will be prompted for
        if not given.

    password : `str`, optional
        Cleartext password of user for given realm, will be prompted for
        if not given.

        .. warning::

            Passing passwords in plain text presents a security risk, please
            consider using a Kerberos keytab file to store credentials.

    realm : `str`, optional
        Name of realm to authenticate against if not given as part of
        ``username``.
        Defaults to ``'default_realm'``; see ``man krb5.conf(5)``.

    keytab : `str`, `~pathlib.Path`, optional
        Path to keytab file. If not given this will be read from the
        ``KRB5_KTNAME`` environment variable. See notes for more details.

    ccache : `str`, `~pathlib.Path`, optional
        Path to Kerberos credentials cache.

    lifetime : `int`, optional
        Desired liftime of the Kerberos credential (may not be respected
        by the underlying GSSAPI implementation); pass `None` to use
        the maximum permitted liftime (default).

        This is currently not respected by MIT Kerberos (the most common
        GSSAPI implementation).

    verbose : `bool`, optional
        DEPRECATED. This argument does nothing, instead of using ``verbose=True``
        use logging to control output.

    krb5ccname : `str`, optional
        DEPRECATED. This argument has been renamed to ``ccache``.

    Returns
    -------
    creds : `gssapi.Credentials`
        The acquired Kerberos credentials.

    Notes
    -----
    If a keytab is given, or is read from the ``KRB5_KTNAME`` environment
    variable, this will be used to guess the principal, if it
    contains only a single credential.

    Examples
    --------
    Example 1: standard user input, with password prompt:

    >>> kinit('albert.einstein')
    Password for albert.einstein@LIGO.ORG:
    Kerberos ticket generated for albert.einstein@LIGO.ORG

    Example 2: extract username and realm from keytab, and use that
    in authentication:

    >>> kinit(keytab='~/.kerberos/ligo.org.keytab', verbose=True)
    Kerberos ticket generated for albert.einstein@LIGO.ORG
    """
    try:
        import gssapi
    except ImportError as exc:
        msg = (
            "cannot generate Kerberos credentials without python-gssapi, "
            "or run `kinit` from your terminal manually."
        )
        raise ImportError(msg) from exc

    # handle deprecated keywords
    if krb5ccname:
        warnings.warn(
            f"The `krb5ccname` keyword for {__name__}.kinit was renamed "
            "to `ccache`, and will stop working in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if ccache is None:
            ccache = krb5ccname
    if verbose is not None:
        warnings.warn(
            f"The `verbose` keyword for {__name__}.kinit is deprecated, "
            "and will stop working in a future release. "
            "Please use logging to control output.",
            DeprecationWarning,
            stacklevel=2,
        )

    # get keytab and check we can use it (username in keytab)
    try:
        username, keytab = _validate_keytab(username, realm, keytab)
    except gssapi.exceptions.GSSError as exc:
        msg = f"Kerberos keytab '{keytab}' is invalid, see traceback for full details"
        raise KerberosError(msg) from exc

    # refuse to prompt if we can't get an answer
    # note: jupyter streams are not recognised as interactive
    #       (isatty() returns False) so we have a special case here
    _check_interactive(
        username,
        password,
        keytab,
    )

    # get username
    principal = _get_principal(
        username,
        realm,
    )

    # get password
    if not keytab and password is None:
        password = getpass.getpass(prompt=f"Password for {principal}: ")

    # generate credential
    try:
        if keytab:
            creds = _acquire_keytab(
                principal,
                keytab,
                ccache=ccache,
                lifetime=lifetime,
            )
        else:
            creds = _acquire_password(
                principal,
                str(password),
                ccache=ccache,
                lifetime=lifetime,
            )
    except gssapi.exceptions.GSSError as exc:
        msg = (
            f"failed to generate Kerberos TGT for {principal}, "
            "see traceback for full details"
        )
        raise KerberosError(msg) from exc

    logger.info(
        "Kerberos ticket acquired for %s (%d seconds remaining)",
        creds.name,
        creds.lifetime,
    )

    return creds


# -- deprecated ---------------------------------

@deprecated_function
def parse_keytab(keytab: str) -> list[tuple[str, str, int]]:  # pragma: no cover
    """Read the contents of a KRB5 keytab file, returning a list of credentials.

    Parameters
    ----------
    keytab : `str`
        path to keytab file

    Returns
    -------
    creds : `list` of `tuple`
        the (unique) list of `(username, realm, kvno)` as read from the
        keytab file

    Examples
    --------
    >>> from gwpy.io.kerberos import parse_keytab
    >>> print(parse_keytab("creds.keytab"))
    [('albert.einstein', 'LIGO.ORG', 1)]
    """
    try:
        out = subprocess.check_output(
            [KLIST, "-k", keytab],
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        msg = "failed to locate klist, cannot read keytab"
        raise KerberosError(msg) from exc
    except subprocess.CalledProcessError as exc:
        msg = f"cannot read keytab '{keytab}'"
        raise KerberosError(msg) from exc
    principals: list[tuple[str, str, int]] = []
    line: str | bytes
    for line in out.splitlines():
        if isinstance(line, bytes):
            line = line.decode("utf-8")  # noqa: PLW2901
        try:
            kvno, principal, = re.split(r"\s+", line.strip(" "), maxsplit=1)
        except ValueError:
            continue
        else:
            if not kvno.isdigit():
                continue
            principals.append((*principal.split("@"), int(kvno)))
    # return unique, ordered list
    return list(OrderedDict.fromkeys(principals))
