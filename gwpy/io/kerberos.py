# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2023)
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

import getpass
import os
import re
import subprocess
import sys
import warnings
from collections import OrderedDict
from unittest import mock

from ..utils.decorators import deprecated_function

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = ['kinit']

try:
    _IPYTHON = __IPYTHON__
except NameError:
    _IPYTHON = False

DEFAULT_REALM = "LIGO.ORG"


class KerberosError(RuntimeError):
    """Kerberos (krb5) operation failed
    """
    pass


def kinit(
    username=None,
    password=None,
    realm=None,
    keytab=None,
    ccache=None,
    lifetime=None,
    krb5ccname=None,
    verbose=None,
):
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
        Name of realm to authenticate against, if not given as part of
        ``username``, defaults to ``'LIGO.ORG'``.

    keytab : `str`, optional
        Path to keytab file. If not given this will be read from the
        ``KRB5_KTNAME`` environment variable. See notes for more details.

    ccache : `str`, optional
        Path to Kerberos credentials cache.

    lifetime : `int`, optional
        Desired liftime of the Kerberos credential (may not be respected
        by the underlying GSSAPI implementation); pass `None` to use
        the maximum permitted liftime (default).

        This is currently not respected by MIT Kerberos (the most common
        GSSAPI implementation).

    verbose : `bool`, optional
        Print verbose output (if `True`), or not (`False)`; default is `True`
        if any user-prompting is needed, otherwise `False`.

    Notes
    -----
    If a keytab is given, or is read from the ``KRB5_KTNAME`` environment
    variable, this will be used to guess the principal, if it
    contains only a single credential.

    Examples
    --------
    Example 1: standard user input, with password prompt::

    >>> kinit('albert.einstein')
    Password for albert.einstein@LIGO.ORG:
    Kerberos ticket generated for albert.einstein@LIGO.ORG

    Example 2: extract username and realm from keytab, and use that
    in authentication::

    >>> kinit(keytab='~/.kerberos/ligo.org.keytab', verbose=True)
    Kerberos ticket generated for albert.einstein@LIGO.ORG
    """
    try:
        import gssapi
    except ImportError as exc:
        raise type(exc)(
            "cannot generate kerberos credentials without python-gssapi, ",
            "or run `kinit` from your terminal manually."
        )

    # handle deprecated keyword
    if krb5ccname:
        warnings.warn(
            "The `krb5ccname` keyword for gwpy.io.kerberos.kinit was renamed "
            "to `ccache`, and will stop working in a future release.",
            DeprecationWarning,
        )
        if ccache is None:
            ccache = krb5ccname

    # get keytab and check we can use it (username in keytab)
    if keytab is None:
        keytab = os.environ.get('KRB5_KTNAME', None)
        if keytab is None or not os.path.isfile(keytab):
            keytab = None
    if keytab:
        keyprincipal = _keytab_principal(keytab)
        if _use_keytab(username, keytab):
            username = str(keyprincipal)
        else:
            keytab = False

    # refuse to prompt if we can't get an answer
    # note: jupyter streams are not recognised as interactive
    #       (isatty() returns False) so we have a special case here
    if (
        not sys.stdout.isatty()
        and not _IPYTHON
        and (username is None or (not keytab and password is None))
    ):
        raise KerberosError("cannot generate kerberos ticket in a "
                            "non-interactive session, please manually create "
                            "a ticket, or consider using a keytab file")

    # get credentials
    if username is None:
        verbose = True
        username = input(
            "Kerberos principal (user@REALM): ",
        )
    if "@" not in username:
        username = f"{username}@{DEFAULT_REALM}"
    principal = gssapi.Name(
        base=username,
        name_type=gssapi.NameType.kerberos_principal,
    )
    if not keytab and password is None:
        verbose = True
        password = getpass.getpass(prompt=f"Password for {principal}: ")

    # generate credential
    acquire_kw = {  # common options for acquire methods
        "ccache": ccache,
        "lifetime": lifetime,
    }
    try:
        if keytab:
            creds = _acquire_keytab(principal, str(keytab), **acquire_kw)
        else:
            creds = _acquire_password(principal, password, **acquire_kw)
    except gssapi.exceptions.GSSError:
        raise KerberosError(
            f"failed to generate Kerberos TGT for {principal}",
        )
    if verbose:
        print(
            f"Kerberos ticket acquired for {creds.name} "
            f"({creds.lifetime} seconds remaining)",
        )


def _acquire_keytab(principal, keytab, ccache=None, lifetime=None):
    """Acquire a Kerberos TGT using a keytab.
    """
    import gssapi
    store = {
        "client_keytab": str(keytab),
    }
    if ccache:
        store["ccache"] = str(ccache)
    with mock.patch.dict("os.environ", {"KRB5_KTNAME": keytab}):
        creds = gssapi.Credentials(
            name=principal,
            store=store,
            usage="initiate",
            lifetime=lifetime,
        )
    creds.inquire()
    return creds


def _acquire_password(principal, password, ccache=None, lifetime=None):
    """Acquire a Kerberos TGT using principal/password.
    """
    import gssapi
    raw_creds = gssapi.raw.acquire_cred_with_password(
        name=principal,
        password=password.encode("utf-8"),
        usage="initiate",
    )
    creds = gssapi.Credentials(raw_creds.creds)
    creds.inquire()
    creds.store(
        store={"ccache": str(ccache)} if ccache else None,
        usage="initiate",
        overwrite=True,
    )
    return creds


def _keytab_principal(keytab):
    """Return the principal assocated with a Kerberos keytab file.
    """
    import gssapi
    with mock.patch.dict("os.environ", {"KRB5_KTNAME": str(keytab)}):
        try:
            creds = gssapi.Credentials(usage="accept")
        except gssapi.exceptions.MissingCredentialsError as exc:
            warnings.warn(str(exc))
            return None
    return creds.name


def _use_keytab(username, principal):
    """Return `True` if a keytab principal matches the requested username.
    """
    if not username:
        return True
    username = str(username)
    return (
        ("@" in username and username == str(principal))
        or username == str(principal).split("@", 1)[0]
    )


# -- deprecated

@deprecated_function
def parse_keytab(keytab):  # pragma: no cover
    """Read the contents of a KRB5 keytab file, returning a list of
    credentials listed within

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
        out = subprocess.check_output(['klist', '-k', keytab],
                                      stderr=subprocess.PIPE)
    except OSError:
        raise KerberosError("Failed to locate klist, cannot read keytab")
    except subprocess.CalledProcessError:
        raise KerberosError(f"Cannot read keytab '{keytab}'")
    principals = []
    for line in out.splitlines():
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        try:
            kvno, principal, = re.split(r'\s+', line.strip(' '), 1)
        except ValueError:
            continue
        else:
            if not kvno.isdigit():
                continue
            principals.append(tuple(principal.split('@')) + (int(kvno),))
    # return unique, ordered list
    return list(OrderedDict.fromkeys(principals).keys())
