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

"""Unit test for utils module
"""

import subprocess

from six import PY2

import pytest

from gwpy.utils import shell
from gwpy.utils import deps  # deprecated

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def test_shell_call():
    out, err = shell.call(["echo", "This works"])
    assert out == 'This works\n'
    assert err == ''

    out2, err2 = shell.call("echo 'This works'")
    assert out == out2
    assert err == err2

    with pytest.raises(OSError):
        shell.call(['this-command-doesnt-exist'])
    with pytest.raises(subprocess.CalledProcessError):
        shell.call('this-command-doesnt-exist')
    with pytest.raises(subprocess.CalledProcessError):
        shell.call('false')
    with pytest.warns(UserWarning):
        shell.call('false', on_error='warn')


def test_which():
    try:
        result, _ = shell.call('which true')
    except Exception as e:
        pytest.skip(str(e))
    else:
        result = result.rstrip('\n')
    assert shell.which('true') == result


def test_import_method_dependency():
    import subprocess
    mod = deps.import_method_dependency('subprocess')
    assert mod is subprocess
    with pytest.raises(ImportError) as exc:
        deps.import_method_dependency('blah')
    if PY2:
        assert str(exc.value) == ("Cannot import blah required by the "
                                  "test_import_method_dependency() method: "
                                  "'No module named blah'")
    else:
        assert str(exc.value) == "No module named 'blah'"


def test_with_import():
    @deps.with_import('blah')
    def with_import_tester():
        pass

    # FIXME: this should really test the message
    with pytest.raises(ImportError) as exc:
        with_import_tester()
