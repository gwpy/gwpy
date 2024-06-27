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

"""Unit tests for :mod:`gwpy.io.kerberos`
"""

import os
import subprocess
from unittest import mock

import pytest

from .. import kerberos as io_kerberos

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@pytest.fixture(autouse=True)
def mock_krb5_env():
    with mock.patch.dict(os.environ):
        for key in (
            'KRB5_KTNAME',
            'KRB5CCNAME',
        ):
            os.environ.pop(key, None)
        yield


def kerberos_name(name):
    import gssapi
    return gssapi.Name(
        base=name,
        name_type=gssapi.NameType.kerberos_principal,
    )


@mock.patch.dict("sys.modules", {"gssapi": None})
def test_kinit_no_gssapi():
    """Test that we get an augmented error message if ``gssapi`` is missing.
    """
    with pytest.raises(
        ImportError,
        match="cannot generate kerberos credentials without python-gssapi",
    ):
        io_kerberos.kinit()


@pytest.mark.requires("gssapi")
@mock.patch("sys.stdout.isatty", return_value=True)
@mock.patch("gwpy.io.kerberos.input", return_value="rainer.weiss")
@mock.patch("getpass.getpass", return_value="test")
@mock.patch("gssapi.raw.acquire_cred_with_password")
@mock.patch("gssapi.Credentials")
def test_kinit_up(creds, acquire, getpass, input_, _, capsys):
    """Test `gwpy.io.kerberos.kinit` with username and password given.

    Note: without a real credential to use, we can't do much more than check
    that the function under test passes the right arguments to the GSSAPI
    library.
    """
    acquire.return_value = mock.MagicMock()

    # basic call should prompt for username and password
    io_kerberos.kinit()
    input_.assert_called_with(
        "Kerberos principal (user@REALM): ",
    )
    getpass.assert_called_with(
        prompt="Password for rainer.weiss@LIGO.ORG: ",
    )

    # and then use the results in gssapi calls
    acquire.assert_called_with(
        name=kerberos_name("rainer.weiss@LIGO.ORG"),
        password="test".encode("utf-8"),
        usage="initiate",
    )


@pytest.mark.requires("gssapi")
@mock.patch('gwpy.io.kerberos.input')
@mock.patch('getpass.getpass')
@mock.patch("gssapi.raw.acquire_cred_with_password")
@mock.patch("gssapi.Credentials")
def test_kinit_up_kwargs(creds, acquire, getpass, input_):
    """Test `gwpy.io.kerberos.kinit` with username and password given.

    Note: without a real credential to use, we can't do much more than check
    that the function under test passes the right arguments to the GSSAPI
    library.
    """
    io_kerberos.kinit(
        username="albert.einstein@EXAMPLE.COM",
        password="test",
    )
    input_.assert_not_called()
    getpass.assert_not_called()
    acquire.assert_called_with(
        name=kerberos_name("albert.einstein@EXAMPLE.COM"),
        password="test".encode("utf-8"),
        usage="initiate",
    )


@pytest.mark.requires("gssapi")
@mock.patch("gwpy.io.kerberos._acquire_password")
def test_kinit_keytab_dne(acquire_passwd, tmp_path):
    """Test `gwpy.io.kerberos.kinit` with a non-existent keytab.
    """
    keytab = tmp_path / "keytab"  # does not exist

    # check that missing keytab goes to password auth
    with pytest.warns(
        UserWarning,
        match=rf"{keytab.name} is nonexistent or empty\Z",
    ):
        io_kerberos.kinit(
            username='test',
            password='passwd',
            keytab=keytab,
        )
    acquire_passwd.assert_called_once_with(
        kerberos_name("test@LIGO.ORG"),
        "passwd",
        ccache=None,
        lifetime=None,
    )


@pytest.mark.requires("gssapi")
@mock.patch.dict("os.environ")
@mock.patch("gssapi.Credentials")
@mock.patch("gwpy.io.kerberos._keytab_principal")
def test_kinit_keytab(principal, creds, tmp_path):
    """Test `gwpy.io.kerberos.kinit` can handle keytabs properly.
    """
    principal.return_value = kerberos_name("rainer.weiss@LIGO.ORG")

    keytab = tmp_path / "keytab"
    keytab.touch()
    ccache = tmp_path / "ccache"

    # test keytab kwarg
    io_kerberos.kinit(
        keytab=keytab,
        ccache=ccache,
        lifetime=1000,
    )
    creds.assert_called_once_with(
        name=kerberos_name("rainer.weiss@LIGO.ORG"),
        store={
            "client_keytab": str(keytab),
            "ccache": str(ccache),
        },
        usage="initiate",
        lifetime=1000,
    )

    # pass keytab via environment
    creds.reset()
    os.environ["KRB5_KTNAME"] = str(keytab)
    io_kerberos.kinit()
    creds.assert_called_with(
        name=kerberos_name("rainer.weiss@LIGO.ORG"),
        store={
            "client_keytab": str(keytab),
        },
        usage="initiate",
        lifetime=None,
    )


@pytest.mark.requires("gssapi")
def test_kinit_notty():
    """Test `gwpy.io.kerberos.kinit` raises an error in a non-interactive
    session if it needs to prompt for information.

    By default all tests are executed by pytest in a non-interactive session
    so we don't have to mock anything!
    """
    with pytest.raises(io_kerberos.KerberosError):
        io_kerberos.kinit()


@pytest.mark.requires("gssapi")
@mock.patch("gwpy.io.kerberos._acquire_password")
def test_kinit_error(_acquire_password):
    """Test that `gwpy.io.kerberos.kinit` propagates `GSSError`s appropriately.
    """
    import gssapi
    _acquire_password.side_effect = gssapi.exceptions.GSSError(0, 0)

    with pytest.raises(
        io_kerberos.KerberosError,
        match=(
            r"\Afailed to generate Kerberos TGT for test@EXAMPLE.COM\Z"
        ),
    ):
        io_kerberos.kinit(
            username="test@EXAMPLE.COM",
            password="test",
        )


# -- deprecated

@pytest.mark.requires("gssapi")
@mock.patch("gwpy.io.kerberos._acquire_password")
def test_kinit_krb5ccname(_):
    """Test that the ``krb5ccname`` keyword emits a deprecation warning.
    """
    with pytest.warns(
        DeprecationWarning,
        match=(
            "The `krb5ccname` keyword for gwpy.io.kerberos.kinit was renamed"
        ),
    ):
        io_kerberos.kinit(
            username="test@EXAMPLE.COM",
            password="test",
            krb5ccname="cache",
        )


# mocked klist output
KLIST = b"""Keytab name: FILE:/test.keytab
KVNO Principal
---- -------------------------------
   1 albert.einstein@LIGO.ORG
   2 ronald.drever@LIGO.ORG
   2 ronald.drever@LIGO.ORG"""


@mock.patch('subprocess.check_output', return_value=KLIST)
def test_parse_keytab(check_output):
    """Test `gwpy.io.kerberos.parse_keytab.
    """
    # assert principals get extracted correctly
    with pytest.deprecated_call():
        principals = io_kerberos.parse_keytab('test.keytab')
    assert principals == [('albert.einstein', 'LIGO.ORG', 1),
                          ('ronald.drever', 'LIGO.ORG', 2)]


@mock.patch("subprocess.check_output")
@pytest.mark.parametrize(("se", "match"), [
    (OSError, "^Failed to locate klist, cannot read keytab$"),
    (
        subprocess.CalledProcessError(1, "error"),
        "Cannot read keytab 'test.keytab'",
    ),
])
def test_parse_keytab_oserror(mock_check_output, se, match):
    """Test `gwpy.io.kerberos.parse_keytab` fails appropriately.
    """
    mock_check_output.side_effect = se
    with pytest.deprecated_call(), pytest.raises(
        io_kerberos.KerberosError,
        match=match,
    ):
        io_kerberos.parse_keytab('test.keytab')
