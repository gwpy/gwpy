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

"""Utilties for the GWpy test suite
"""

import os.path
import subprocess
import tempfile
from contextlib import contextmanager
from distutils.version import LooseVersion
from importlib import import_module
from itertools import zip_longest
from pathlib import Path

import pytest

import numpy
from numpy.testing import (assert_array_equal, assert_allclose)

from astropy.time import Time

from ..utils.decorators import deprecated_function

# -- useful constants ---------------------------------------------------------

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_GWF_FILE = os.path.join(TEST_DATA_DIR, 'HLV-HW100916-968654552-1.gwf')
TEST_HDF5_FILE = os.path.join(TEST_DATA_DIR, 'HLV-HW100916-968654552-1.hdf')


# -- dependencies -------------------------------------------------------------

@deprecated_function(message=(
    "gwpy.testing.utils.has is deprecated and will "
    "be removed in GWpy 3.1.0",
))
def has(module):  # pragma: no cover
    """Test whether a module is available

    Returns `True` if `import module` succeeded, otherwise `False`
    """
    try:
        import_module(module)
    except ImportError:
        return False
    else:
        return True


@deprecated_function(message=(
    "gwpy.testing.utils.skip_missing_dependency is deprecated and will "
    "be removed in GWpy 3.1.0, please update your code to use "
    "pytest.mark.requires from the pytest-requires package",
))
def skip_missing_dependency(module):  # pragma: no cover
    """Returns a mark generator to skip a test if the dependency is missing

    .. deprecated:: 3.0.0
       Use `pytest.mark.requires` from pytest-requires instead.
    """
    return pytest.mark.skipif(not has(module),
                              reason='No module named %s' % module)


@deprecated_function(message=(
    "gwpy.testing.utils.module_older_than is deprecated and will "
    "be removed in GWpy 3.1.0",
))
def module_older_than(module, minversion):  # pragma: no cover
    mod = import_module(module)
    return LooseVersion(mod.__version__) < LooseVersion(minversion)


@deprecated_function(message=(
    "gwpy.testing.utils.skip_minimum_version is deprecated and will "
    "be removed in GWpy 3.1.0",
))
def skip_minimum_version(module, minversion):  # pragma: no cover
    """Returns a mark generator to skip a test if the dependency is too old

    .. deprecated:: 3.0.0
    """
    return pytest.mark.skipif(
        module_older_than(module, minversion),
        reason='requires {} >= {}'.format(module, minversion))


def _has_kerberos_credential():
    """Return `True` if the current user has a valid kerberos credential

    This function just calls ``klist -s`` and returns `True` if the
    command returns a zero exit code, and `False` if it doesn't, or
    the call fails in any other way.
    """
    try:
        subprocess.check_call(
            ["klist", "-s"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


#: skip a test function if no kerberos credential is found
skip_kerberos_credential = pytest.mark.skipif(
    not _has_kerberos_credential(),
    reason="no kerberos credential",
)


# -- assertions ---------------------------------------------------------------

def assert_quantity_equal(q1, q2):
    """Assert that two `~astropy.units.Quantity` objects are the same
    """
    _assert_quantity(q1, q2, array_assertion=assert_array_equal)


def assert_quantity_almost_equal(q1, q2):
    """Assert that two `~astropy.units.Quantity` objects are almost the same

    This method asserts that the units are the same and that the values are
    equal within precision.
    """
    _assert_quantity(q1, q2, array_assertion=assert_allclose)


def _assert_quantity(q1, q2, array_assertion=assert_array_equal):
    assert q1.unit == q2.unit, "%r != %r" % (q1.unit, q2.unit)
    array_assertion(q1.value, q2.value)


def assert_quantity_sub_equal(a, b, *attrs, **kwargs):
    """Assert that two `~gwpy.types.Array` objects are the same (or almost)

    Parameters
    ----------
    a, b : `~gwpy.types.Array`
        the arrays two be tested (can be subclasses)

    *attrs
        the list of attributes to test, defaults to all

    almost_equal : `bool`, optional
        allow the numpy array's to be 'almost' equal, default: `False`,
        i.e. require exact matches

    exclude : `list`, optional
        a list of attributes to exclude from the test
    """
    # get value test method
    if kwargs.pop('almost_equal', False):
        assert_array = assert_allclose
    else:
        assert_array = assert_array_equal

    # parse attributes to be tested
    if not attrs:
        attrs = a._metadata_slots
    exclude = kwargs.pop('exclude', [])
    attrs = [attr for attr in attrs if attr not in exclude]

    # don't assert indexes that don't exist for both
    def _check_index(dim):
        index = "{}index".format(dim)
        _index = "_" + index
        if (
            index in attrs
            and getattr(a, _index, "-") == "-"
            and getattr(b, _index, "-") == "-"
        ):
            attrs.remove(index)
    _check_index("x")
    _check_index("y")

    # test data
    assert_attributes(a, b, *attrs)
    assert_array(a.value, b.value)


def assert_attributes(a, b, *attrs):
    """Assert that the attributes for two objects match

    `attrs` should be `list` of attribute names that can be accessed
    with `getattr`
    """
    for attr in attrs:
        x = getattr(a, attr, None)
        y = getattr(b, attr, None)
        if isinstance(x, numpy.ndarray) and isinstance(b, numpy.ndarray):
            assert_array_equal(x, y)
        elif isinstance(x, Time) and isinstance(y, Time):
            assert x.gps == y.gps
        else:
            assert x == y


def assert_table_equal(a, b, is_copy=True, meta=False, check_types=True,
                       almost_equal=False):
    """Assert that two tables store the same information
    """
    # check column names are the same
    assert sorted(a.colnames) == sorted(b.colnames)

    # check that the metadata match
    if meta:
        assert a.meta == b.meta

    if almost_equal:
        assert_array = assert_allclose
    else:
        assert_array = assert_array_equal

    # actually check the data
    for name in a.colnames:
        cola = a[name]
        colb = b[name]
        if check_types:
            assert cola.dtype == colb.dtype
        assert_array(cola, colb)

    # check that the tables are copied or the same data
    for name in a.colnames:
        # check may_share_memory is True when copy is False and so on
        assert numpy.may_share_memory(a[name], b[name]) is not is_copy


def assert_segmentlist_equal(a, b):
    """Assert that two `SegmentList`s contain the same data
    """
    for aseg, bseg in zip_longest(a, b):
        assert aseg == bseg


def assert_flag_equal(a, b, attrs=['name', 'ifo', 'tag', 'version']):
    """Assert that two `DataQualityFlag`s contain the same data
    """
    assert_segmentlist_equal(a.active, b.active)
    assert_segmentlist_equal(a.known, b.known)
    for attr in attrs:
        assert getattr(a, attr) == getattr(b, attr)


def assert_dict_equal(a, b, assert_value, *args, **kwargs):
    """Assert that two `dict`s contain the same data

    Parameters
    ----------
    a, b
        two objects to compare

    assert_value : `callable`
        method to compare that two dict entries are the same

    *args, **kargs
        positional and keyword arguments to pass to ``assert_value``
    """
    assert a.keys() == b.keys()
    for key in a:
        assert_value(a[key], b[key], *args, **kwargs)


def assert_zpk_equal(a, b, almost_equal=False):
    if almost_equal:
        assert_array = assert_allclose
    else:
        assert_array = assert_array_equal
    for x, y in zip(a, b):  # zip through zeros, poles, gain
        assert_array(x, y)


# -- I/O helpers --------------------------------------------------------------

@contextmanager
@deprecated_function
def TemporaryFilename(*args, **kwargs):  # pylint: disable=invalid-name
    """Create and return a temporary filename

    Calls `tempfile.mktemp` to create a temporary filename, and deletes
    the named file (if it exists) when the context ends.

    This method **does not create the named file**.

    Examples
    --------
    >>> with TemporaryFilename(suffix='.txt') as tmp:
    ...     print(tmp)
    '/var/folders/xh/jdrqg2bx3s5f4lkq0rf2903c0000gq/T/tmpnNxivL.txt'
    """
    name = tempfile.mktemp(*args, **kwargs)
    try:
        yield name
    finally:
        if os.path.isfile(name):
            os.remove(name)


def test_read_write(data, format,
                    extension=None, autoidentify=True,
                    read_args=[], read_kw={},
                    write_args=[], write_kw={},
                    assert_equal=assert_array_equal, assert_kw={}):
    """Test that data can be written to and read from a file in some format

    Parameters
    ----------
    data : some type with `.read()` and `.write()` methods
        the data to be written

    format : `str`
        the name of the file format (as registered with `astropy.io.registry`

    extension : `str`, optional
        the name of the file extension, defaults to ``.<format>``

    autoidenfity : `bool`, optional
        attempt to auto-identify when reading writing by not specifying
        ``format``

    read_args : `list`, optional
        positional arguments to pass to ``type(data).read()``

    read_kwargs : `dict`, optional
        keyword arguments to pass to ``type(data).read()``

    write_args : `list`, optional
        positional arguments to pass to ``data.write()``

    write_kwargs : `dict`, optional
        keyword arguments to pass to ``data.write()``

    assert_equal : `callable`, optional
        the function to assert that the object read back from file matches
        the original ``data``

    assert_kwargs : `dict`, optional
        keyword arguments to pass to ``assert_equal``
    """
    # parse extension and add leading period
    if extension is None:
        extension = format
    extension = '.%s' % extension.lstrip('.')

    DataClass = type(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "test.{}".format(extension)

        data.write(tmp, *write_args, format=format, **write_kw)

        # try again with automatic format identification
        if autoidentify:
            data.write(str(tmp), *write_args, **write_kw)

        # read the data back and check that its the same
        new = DataClass.read(tmp, *read_args, format=format, **read_kw)
        assert_equal(new, data, **assert_kw)

        # try again with automatic format identification
        if autoidentify:
            new = DataClass.read(str(tmp), *read_args, **read_kw)
            assert_equal(new, data, **assert_kw)
