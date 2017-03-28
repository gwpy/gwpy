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

import os
import subprocess

import pytest

from compat import unittest

from gwpy.utils import shell

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class UtilsTestCase(unittest.TestCase):
    """`TestCase` for the utils module
    """
    def test_shell_call(self):
        out, err = shell.call(["echo", "This works"])
        self.assertEqual(out, "This works\n")
        self.assertEqual(err, '')
        out2, err2 = shell.call("echo 'This works'")
        self.assertEqual(out, out2)
        self.assertEqual(err, err2)
        self.assertRaises(OSError, shell.call, ['this-command-doesnt-exist'])
        self.assertRaises(subprocess.CalledProcessError, shell.call,
                          'this-command-doesnt-exist')
        self.assertRaises(subprocess.CalledProcessError, shell.call, 'false')
        with pytest.warns(UserWarning):
            shell.call('false', on_error='warn')

    def test_which(self):
        try:
            result, _ = shell.call('which true')
        except Exception as e:
            self.skipTest(str(e))
        else:
            result = result.decode('utf-8').rstrip('\n')
        self.assertEqual(shell.which('true'), result)
