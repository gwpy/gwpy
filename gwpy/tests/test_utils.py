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
from importlib import import_module
from math import sqrt

from six import PY2

import pytest

import numpy

from astropy import units

from gwpy.utils import (shell, mp as utils_mp)
from gwpy.utils import deps  # deprecated

import utils

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


def test_compat_ordereddict():
    with pytest.warns(DeprecationWarning):
        from gwpy.utils.compat import OrderedDict as CompatOrderedDict
    from collections import OrderedDict
    assert CompatOrderedDict is OrderedDict


# -- gwpy.utils.lal -----------------------------------------------------------

@utils.skip_missing_dependency('lal')
class TestUtilsLal(object):
    """Tests of :mod:`gwpy.utils.lal`
    """
    def setup_class(self):
        self.lal = import_module('lal')
        self.utils_lal = import_module('gwpy.utils.lal')

    def test_to_lal_type_str(self):
        assert self.utils_lal.to_lal_type_str(float) == 'REAL8'
        assert self.utils_lal.to_lal_type_str(
            numpy.dtype('float64')) == 'REAL8'
        assert self.utils_lal.to_lal_type_str(11) == 'REAL8'
        with pytest.raises(ValueError):
            self.utils_lal.to_lal_type_str('blah')
        with pytest.raises(ValueError):
            self.utils_lal.to_lal_type_str(numpy.int8)
        with pytest.raises(ValueError):
            self.utils_lal.to_lal_type_str(20)

    def test_find_typed_function(self):
        assert self.utils_lal.find_typed_function(
            'REAL8', 'Create', 'Sequence') is self.lal.CreateREAL8Sequence

        try:
            import lalframe
        except ImportError:  # no lalframe
            pass
        else:
            self.utils_lal.find_typed_function(
                'REAL4', 'FrStreamRead', 'TimeSeries',
                module=lalframe) is lalframe.FrStreamReadREAL4TimeSeries

    def test_to_lal_unit(self):
        assert self.utils_lal.to_lal_unit('m') == self.lal.MeterUnit
        assert self.utils_lal.to_lal_unit('Farad') == self.lal.Unit(
            'm^-2 kg^-1 s^4 A^2')
        with pytest.raises(ValueError):
            self.utils_lal.to_lal_unit('rad/s')

    def test_from_lal_unit(self):
        assert self.utils_lal.from_lal_unit(
            self.lal.MeterUnit / self.lal.SecondUnit) == (
            units.Unit('m/s'))
        assert self.utils_lal.from_lal_unit(self.lal.StrainUnit) == (
            units.Unit('strain'))

    def test_to_lal_ligotimegps(self):
        assert self.utils_lal.to_lal_ligotimegps(123.456) == (
            self.lal.LIGOTimeGPS(123, 456000000))


# -- gwpy.utils.mp ------------------------------------------------------------

class TestUtilsMp(object):

    @pytest.mark.parametrize('verbose', [False, True, 'Test'])
    @pytest.mark.parametrize('nproc', [1, 2])
    def test_multiprocess_with_queues(self, capsys, nproc, verbose):
        inputs = [1, 4, 9, 16, 25]
        out = utils_mp.multiprocess_with_queues(
            nproc, sqrt, inputs, verbose=verbose,
        )
        assert out == [1, 2, 3, 4, 5]

        # assert progress bar prints correctly
        cap = capsys.readouterr()
        if verbose is True:
            assert cap.out.startswith('\rProcessing: ')
        elif verbose is False:
            assert cap.out == ''
        else:
            assert cap.out.startswith('\r{}: '.format(verbose))

        with pytest.raises(ValueError):
            utils_mp.multiprocess_with_queues(nproc, sqrt, [-1],
                                              verbose=verbose)

    def test_multiprocess_with_queues_raise(self):
        with pytest.warns(DeprecationWarning):
            utils_mp.multiprocess_with_queues(1, sqrt, [1],
                                              raise_exceptions=True)
