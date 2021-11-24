# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

# mocked klist output
KLIST = b"""Keytab name: FILE:/test.keytab
KVNO Principal
---- -------------------------------
   1 albert.einstein@LIGO.ORG
   2 ronald.drever@LIGO.ORG
   2 ronald.drever@LIGO.ORG"""

# mock os.environ
MOCK_ENV = None


def setup_module():
    global MOCK_ENV
    MOCK_ENV = mock.patch.dict(os.environ, {})
    MOCK_ENV.start()
    for key in (
        'KRB5_KTNAME',
        'KRB5CCNAME',
    ):
        os.environ.pop(key, None)


def teardown_module():
    if MOCK_ENV is not None:
        MOCK_ENV.stop()


@mock.patch('subprocess.check_output', return_value=KLIST)
def test_parse_keytab(check_output):
    """Test `gwpy.io.kerberos.parse_keytab
    """
    # assert principals get extracted correctly
    principals = io_kerberos.parse_keytab('test.keytab')
    assert principals == [('albert.einstein', 'LIGO.ORG', 1),
                          ('ronald.drever', 'LIGO.ORG', 2)]

    # assert klist fail gets raise appropriately
    check_output.side_effect = [
        OSError('something'),
        subprocess.CalledProcessError(1, 'something else'),
    ]
    with pytest.raises(io_kerberos.KerberosError) as exc:
        io_kerberos.parse_keytab('test.keytab')
    assert str(exc.value) == "Failed to locate klist, cannot read keytab"
    with pytest.raises(io_kerberos.KerberosError) as exc:
        io_kerberos.parse_keytab('test.keytab')
    assert str(exc.value) == "Cannot read keytab 'test.keytab'"


@mock.patch('sys.stdout.isatty', return_value=True)
@mock.patch('gwpy.io.kerberos.input', return_value='rainer.weiss')
@mock.patch('getpass.getpass', return_value='test')
@mock.patch('subprocess.Popen')
def test_kinit_up(popen, getpass, input_, _, capsys):
    """Test `gwpy.io.kerberos.kinit` with username and password given
    """
    proc = popen.return_value
    proc.poll.return_value = 0

    # basic call should prompt for username and password
    io_kerberos.kinit()
    input_.assert_called_with(
        "Please provide username for the LIGO.ORG kerberos realm: ",
    )
    getpass.assert_called_with(
        prompt="Password for rainer.weiss@LIGO.ORG: ",
    )
    popen.assert_called_with(
        ['kinit', 'rainer.weiss@LIGO.ORG'],
        stdin=-1,
        stdout=-1,
        env=None,
    )
    proc.communicate.aossert_called_with(b'test')


@mock.patch('gwpy.io.kerberos.input')
@mock.patch('getpass.getpass')
@mock.patch('subprocess.Popen')
def test_kinit_up_kwargs(popen, getpass, input_):
    """Test `gwpy.io.kerberos.kinit` with username and password given
    """
    proc = popen.return_value
    proc.poll.return_value = 0

    io_kerberos.kinit(
        username='albert.einstein',
        password='test',
        exe='/usr/bin/kinit',
    )
    input_.assert_not_called()
    getpass.assert_not_called()
    popen.assert_called_with(
        ['/usr/bin/kinit', 'albert.einstein@LIGO.ORG'],
        stdin=-1,
        stdout=-1,
        env=None,
    )
    popen.return_value.communicate.assert_called_with(b'test')


@mock.patch('gwpy.io.kerberos.parse_keytab')
@mock.patch('subprocess.Popen')
def test_kinit_keytab_dne(popen, parse_keytab):
    """Test `gwpy.io.kerberos.kinit` with a non-existent keytab
    """
    proc = popen.return_value
    proc.poll.return_value = 0

    # test keytab from environment not found (default) prompts user
    io_kerberos.kinit(username='test', password='passwd',
                      exe='/bin/kinit')
    parse_keytab.assert_not_called()
    popen.assert_called_with(
        ['/bin/kinit', 'test@LIGO.ORG'],
        stdin=-1,
        stdout=-1,
        env=None,
    )


@mock.patch.dict(os.environ, {'KRB5_KTNAME': '/test.keytab'})
@mock.patch('os.path.isfile', return_value=True)
@mock.patch(
    'gwpy.io.kerberos.parse_keytab',
    return_value=[['rainer.weiss', 'LIGO.ORG']],
)
@mock.patch('subprocess.Popen')
def test_kinit_keytab(popen, *unused_mocks):
    """Test `gwpy.io.kerberos.kinit` can handle keytabs properly
    """
    proc = popen.return_value
    proc.poll.return_value = 0

    # test keytab kwarg
    io_kerberos.kinit(keytab='test.keytab', exe='/bin/kinit')
    popen.assert_called_with(
        ['/bin/kinit', '-k', '-t', 'test.keytab', 'rainer.weiss@LIGO.ORG'],
        stdin=-1,
        stdout=-1,
        env=None,
    )

    # pass keytab via environment
    io_kerberos.kinit(exe='/bin/kinit')
    popen.assert_called_with(
        ['/bin/kinit', '-k', '-t', '/test.keytab', 'rainer.weiss@LIGO.ORG'],
        stdin=-1,
        stdout=-1,
        env=None,
    )


@mock.patch('subprocess.Popen')
def test_kinit_krb5ccname(popen):
    """Test `gwpy.io.kerberos.kinit` passes `KRB5CCNAME` to /bin/kinit
    """
    # test using krb5ccname (credentials cache)
    # this will raise error because we haven't patched the poll() method
    # to return 0, but will test that we get the right error
    with pytest.raises(subprocess.CalledProcessError):
        io_kerberos.kinit(username='test', password='test',
                          krb5ccname='/test_cc.krb5', exe='/bin/kinit')
    popen.assert_called_with(
        ['/bin/kinit', 'test@LIGO.ORG'],
        stdin=-1,
        stdout=-1,
        env={'KRB5CCNAME': '/test_cc.krb5'},
    )


def test_kinit_notty():
    """Test `gwpy.io.kerberos.kinit` raises an error in a non-interactive
    session if it needs to prompt for information.

    By default all tests are executed by pytest in a non-interactive session
    so we don't have to mock anything!
    """
    with pytest.raises(io_kerberos.KerberosError):
        io_kerberos.kinit(exe='/bin/kinit')
