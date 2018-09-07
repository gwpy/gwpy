# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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
import sys

import pytest

from ...tests.mocks import mock
from ...tests.utils import TemporaryFilename
from .. import kerberos as io_kerberos

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# remove real user's keytab, if present
_KTNAME = os.environ.get('KRB5_KTNAME', '_NO KT_')

# mocked klist output
KLIST = b"""Keytab name: FILE:/test.keytab
KVNO Principal
---- -------------------------------
   1 albert.einstein@LIGO.ORG
   2 ronald.drever@LIGO.ORG"""


def setup_module():
    os.environ.pop('KRB5_KTNAME', None)


def teardown_module():
    if _KTNAME != '_NO KT_':
        os.environ['KRB5_KTNAME'] = _KTNAME


def mock_popen_return(popen, out='', err='', returncode=0):
    mocked_p = mock.Mock()
    mocked_p.__enter__ = mock.Mock(return_value=mocked_p)
    mocked_p.__exit__ = mock.Mock(return_value=None)
    mocked_p.configure_mock(**{
        'communicate.return_value': (out, err),
        'poll.return_value': returncode,
        'returncode': returncode,
    })
    popen.return_value = mocked_p


@mock.patch('subprocess.Popen')
def test_parse_keytab(mocked_popen):
    mock_popen_return(mocked_popen, out=KLIST)

    # assert principals get extracted correctly
    principals = io_kerberos.parse_keytab('test.keytab')
    assert principals == [['albert.einstein', 'LIGO.ORG'],
                          ['ronald.drever', 'LIGO.ORG']]

    # assert klist fail gets raise appropriately
    mock_popen_return(mocked_popen, returncode=1)
    with pytest.raises(io_kerberos.KerberosError):
        io_kerberos.parse_keytab('test.keytab')


@mock.patch('gwpy.io.kerberos.which', return_value='/bin/kinit')
@mock.patch('subprocess.Popen')
@mock.patch('getpass.getpass', return_value='test')
@mock.patch('gwpy.io.kerberos.input', return_value='rainer.weiss')
def test_kinit(raw_input_, getpass, mocked_popen, which, capsys):
    mocked_popen.return_value.poll.return_value = 0

    # default popen kwargs
    popen_kwargs = {'stdin': -1, 'stdout': -1, 'env': None}

    # pass username and password, and kinit exe path
    io_kerberos.kinit(username='albert.einstein', password='test',
                      exe='/usr/bin/kinit', verbose=True)
    mocked_popen.assert_called_with(
        ['/usr/bin/kinit', 'albert.einstein@LIGO.ORG'], **popen_kwargs)
    out, err = capsys.readouterr()
    assert out == (
        'Kerberos ticket generated for albert.einstein@LIGO.ORG\n')

    # configure klisting (remove Drever)
    mock_popen_return(mocked_popen, out=KLIST.rsplit(b'\n', 1)[0])
    os.environ['KRB5_KTNAME'] = '/test.keytab'

    # test keytab from environment not found (default) prompts user
    io_kerberos.kinit()
    mocked_popen.assert_called_with(
        ['/bin/kinit', 'rainer.weiss@LIGO.ORG'], **popen_kwargs)

    # test keytab from enviroment found
    with TemporaryFilename(suffix='.keytab') as tmp:
        io_kerberos.kinit(keytab=tmp)
        mocked_popen.assert_called_with(
            ['/bin/kinit', '-k', '-t', tmp, 'albert.einstein@LIGO.ORG'],
            **popen_kwargs)

    os.environ.pop('KRB5_KTNAME', None)

    # pass keytab
    io_kerberos.kinit(keytab='test.keytab')
    mocked_popen.assert_called_with(
        ['/bin/kinit', '-k', '-t', 'test.keytab',
         'albert.einstein@LIGO.ORG'], **popen_kwargs)

    # don't pass keytab (prompts for username and password)
    io_kerberos.kinit()
    getpass.assert_called_with(
        prompt='Password for rainer.weiss@LIGO.ORG: ', stream=sys.stdout)
    mocked_popen.assert_called_with(
        ['/bin/kinit', 'rainer.weiss@LIGO.ORG'], **popen_kwargs)

    # test using krb5ccname (credentials cache)
    io_kerberos.kinit(krb5ccname='/test_cc.krb5')
    popen_kwargs['env'] = {'KRB5CCNAME': '/test_cc.krb5'}
    mocked_popen.assert_called_with(
        ['/bin/kinit', 'rainer.weiss@LIGO.ORG'], **popen_kwargs)
