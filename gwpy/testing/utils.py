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

"""Utilties for the GWpy test suite."""

# ruff: noqa: PT028, S101

from __future__ import annotations

import subprocess
import tempfile
from itertools import zip_longest
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

import numpy
import pytest
from astropy.time import Time
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
)

from ..io.cache import file_segment

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Container,
        Iterable,
    )
    from typing import Any

    from astropy.units import Quantity

    from ..segments import (
        DataQualityFlag,
        Segment,
    )
    from ..signal.filter_design import ZpkType
    from ..types import Array

KLIST = which("klist")

# -- useful constants ----------------

TEST_DATA_PATH = Path(__file__).parent / "data"
TEST_DATA_DIR = str(TEST_DATA_PATH)
TEST_GWF_FILE = str(TEST_DATA_PATH / "HLV-HW100916-968654552-1.gwf")
TEST_GWF_SPAN = file_segment(TEST_GWF_FILE)
TEST_HDF5_FILE = str(TEST_DATA_PATH / "HLV-HW100916-968654552-1.hdf")


# -- dependencies --------------------

def _has_kerberos_credential() -> bool:
    """Return `True` if the current user has a valid kerberos credential.

    This function just calls ``klist -s`` and returns `True` if the
    command returns a zero exit code, and `False` if it doesn't, or
    the call fails in any other way.
    """
    if KLIST is None:
        return False
    try:
        subprocess.check_call(
            [KLIST, "-s"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (
        subprocess.CalledProcessError,  # klist failed (no credential)
        FileNotFoundError,  # klist isn't there
    ):
        return False
    return True


#: skip a test function if no kerberos credential is found
skip_kerberos_credential = pytest.mark.skipif(
    not _has_kerberos_credential(),
    reason="no kerberos credential",
)


# -- assertions ----------------------

def assert_quantity_equal(
    q1: Quantity,
    q2: Quantity,
) -> None:
    """Assert that two `~astropy.units.Quantity` objects are the same."""
    _assert_quantity(
        q1,
        q2,
        array_assertion=assert_array_equal,
    )


def assert_quantity_almost_equal(
    q1: Quantity,
    q2: Quantity,
) -> None:
    """Assert that two `~astropy.units.Quantity` objects are almost the same.

    This method asserts that the units are the same and that the values are
    equal within precision.
    """
    _assert_quantity(
        q1,
        q2,
        array_assertion=assert_allclose,
    )


def _assert_quantity(
    q1: Quantity,
    q2: Quantity,
    array_assertion: Callable = assert_array_equal,
) -> None:
    assert q1.unit == q2.unit, f"'{q1.unit}' != '{q2.unit}'"
    array_assertion(q1.value, q2.value)


def assert_quantity_sub_equal(
    a: Array,
    b: Array,
    *attrs: str,
    almost_equal: bool = False,
    exclude: Container[str] | None = None,
    **kwargs,
) -> None:
    """Assert that two `~gwpy.types.Array` objects are the same (or almost).

    Parameters
    ----------
    a, b : `~gwpy.types.Array`
        The arrays to be tested (can be subclasses).

    *attrs
        The list of attributes to test, defaults to all.

    almost_equal : `bool`, optional
        Allow the numpy array's to be 'almost' equal, default: `False`,
        i.e. require exact matches.

    exclude : `list`, optional
        A list of attributes to exclude from the test.

    kwargs
        Other keyword arguments are passed to the array comparison operator
        `numpy.testing.assert_array_equal` or `numpy.testing.assert_allclose`.

    See Also
    --------
    numpy.testing.assert_array_equal
    numpy.testing.assert_allclose
    """
    # get value test method
    assert_array: Callable
    if almost_equal:
        assert_array = assert_allclose
    else:
        assert_array = assert_array_equal

    # parse attributes to be tested
    if not attrs:
        attrs = a._metadata_slots
    checkattrs = [attr for attr in attrs if attr not in (exclude or [])]

    # don't assert indexes that don't exist for both
    def _check_index(dim: str) -> None:
        index = f"{dim}index"
        _index = "_" + index
        if (
            index in attrs
            and getattr(a, _index, "-") == "-"
            and getattr(b, _index, "-") == "-"
        ):
            checkattrs.remove(index)
    _check_index("x")
    _check_index("y")

    # test data
    assert_attributes(a, b, *checkattrs)
    assert_array(a.value, b.value, **kwargs)


def assert_attributes(
    a: Array,
    b: Array,
    *attrs: str,
) -> None:
    """Assert that the attributes for two objects match.

    `attrs` should be `list` of attribute names that can be accessed
    with `getattr`
    """
    for attr in attrs:
        x = getattr(a, attr, None)
        y = getattr(b, attr, None)
        if isinstance(x, numpy.ndarray) and isinstance(y, numpy.ndarray):
            assert_array_equal(x, y)
        elif isinstance(x, Time) and isinstance(y, Time):
            assert x.gps == y.gps
        else:
            assert x == y


def assert_table_equal(
    a: Array,
    b: Array,
    *,
    is_copy: bool = True,
    meta: bool = False,
    check_types: bool = True,
    almost_equal: bool = False,
) -> None:
    """Assert that two tables store the same information."""
    # check column names are the same
    assert sorted(a.colnames) == sorted(b.colnames)

    # check that the metadata match
    if meta:
        assert a.meta == b.meta

    assert_array: Callable
    if almost_equal:
        assert_array = assert_allclose
    else:
        assert_array = assert_array_equal

    # actually check the data
    for name in a.colnames:
        cola = a[name]
        colb = b[name]
        if check_types:
            assert cola.dtype == colb.dtype, \
                f"{name} dtype mismatch: {cola.dtype} != {colb.dtype}"
        assert_array(cola, colb)

    # check that the tables are copied or the same data
    for name in a.colnames:
        # check may_share_memory is True when copy is False and so on
        assert numpy.may_share_memory(a[name], b[name]) is not is_copy


def assert_segmentlist_equal(
    a: Iterable[Segment],
    b: Iterable[Segment],
) -> None:
    """Assert that two `SegmentList`s contain the same data."""
    for aseg, bseg in zip_longest(a, b):
        assert aseg == bseg


def assert_flag_equal(
    a: DataQualityFlag,
    b: DataQualityFlag,
    attrs: Iterable[str] = ("name", "ifo", "tag", "version"),
) -> None:
    """Assert that two `DataQualityFlag`s contain the same data."""
    assert_segmentlist_equal(a.active, b.active)
    assert_segmentlist_equal(a.known, b.known)
    for attr in attrs:
        assert getattr(a, attr) == getattr(b, attr)


def assert_dict_equal(
    a: dict[Any, Any],
    b: dict[Any, Any],
    assert_value: Callable,
    **kwargs,
) -> None:
    """Assert that two `dict`s contain the same data.

    Parameters
    ----------
    a, b
        Two objects to compare.

    assert_value : `callable`
        Method to compare that two dict entries are the same.

    kwargs
        Keyword arguments to pass to ``assert_value``.
    """
    assert a.keys() == b.keys()
    for key in a | b:
        assert_value(a[key], b[key], **kwargs)


def assert_zpk_equal(
    a: ZpkType,
    b: ZpkType,
    *,
    almost_equal: bool = False,
) -> None:
    """Assert that two ZPK filters are the same (or almost)."""
    assert_array: Callable
    if almost_equal:
        assert_array = assert_allclose
    else:
        assert_array = assert_array_equal
    z1, p1, k1 = a
    z2, p2, k2 = b
    assert_array(z1, z2)  # zeros
    assert_array(p1, p2)  # poles
    assert_array(k1, k2)  # gain


# -- I/O helpers ---------------------

def test_read_write(
    data: Array,
    fmt: str,
    extension: str | None = None,
    *,
    autoidentify: bool = True,
    read_args: Iterable[Any] = [],
    read_kw: dict[str, Any] | None = None,
    write_args: Iterable[Any] = [],
    write_kw: dict[str, Any] | None = None,
    assert_equal: Callable = assert_quantity_sub_equal,
    assert_kw: dict[str, Any] | None = None,
) -> None:
    """Test that data can be written to and read from a file in some format.

    Parameters
    ----------
    data : some type with `.read()` and `.write()` methods
        the data to be written

    fmt : `str`
        the name of the file format (as registered with `astropy.io.registry`

    extension : `str`, optional
        the name of the file extension, defaults to ``.<fmt>``

    autoidentify : `bool`, optional
        attempt to auto-identify when reading writing by not specifying
        ``format``

    read_args : `list`, optional
        positional arguments to pass to ``type(data).read()``

    read_kw : `dict`, optional
        keyword arguments to pass to ``type(data).read()``

    write_args : `list`, optional
        positional arguments to pass to ``data.write()``

    write_kw : `dict`, optional
        keyword arguments to pass to ``data.write()``

    assert_equal : `callable`, optional
        the function to assert that the object read back from file matches
        the original ``data``

    assert_kw : `dict`, optional
        keyword arguments to pass to ``assert_equal``
    """
    # parse extension and add leading period
    if assert_kw is None:
        assert_kw = {}
    if write_kw is None:
        write_kw = {}
    if read_kw is None:
        read_kw = {}
    if extension is None:
        extension = fmt
    extension = extension.lstrip(".")

    DataClass = type(data)  # noqa: N806

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / f"test.{extension}"

        data.write(tmp, *write_args, format=fmt, **write_kw)

        # try again with automatic format identification
        if autoidentify:
            data.write(str(tmp), *write_args, **write_kw)

        # read the data back and check that its the same
        new = DataClass.read(tmp, *read_args, format=fmt, **read_kw)
        assert_equal(new, data, **assert_kw)

        # try again with automatic format identification
        if autoidentify:
            new = DataClass.read(str(tmp), *read_args, **read_kw)
            assert_equal(new, data, **assert_kw)
