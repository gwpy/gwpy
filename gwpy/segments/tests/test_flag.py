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

"""Tests for :mod:`gwpy.segments`."""

from __future__ import annotations

import re
import warnings
from contextlib import nullcontext
from http import HTTPStatus
from io import BytesIO
from typing import TYPE_CHECKING
from unittest import mock

import h5py
import pytest
from matplotlib import rc_context
from requests.exceptions import HTTPError
from requests.models import Response
from requests.status_codes import codes as status_codes

from ...plot import SegmentAxes
from ...segments import (
    DataQualityDict,
    DataQualityFlag,
    Segment,
    SegmentList,
)
from ...testing import utils
from ...testing.errors import pytest_skip_flaky_network

if TYPE_CHECKING:
    from typing import SupportsFloat

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- veto definer fixture ------------

VETO_DEFINER_FILE = """
<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt">
<LIGO_LW>
    <Table Name="veto_definer:table">
        <Column Type="int_4s" Name="category"/>
        <Column Type="lstring" Name="comment"/>
        <Column Type="int_4s" Name="end_pad"/>
        <Column Type="int_8s" Name="process:process_id"/>
        <Column Type="lstring" Name="name"/>
        <Column Type="int_4s" Name="version"/>
        <Column Type="int_4s" Name="start_pad"/>
        <Column Type="int_4s" Name="start_time"/>
        <Column Type="lstring" Name="ifo"/>
        <Column Type="int_4s" Name="end_time"/>
        <Stream Delimiter="," Type="Local" Name="veto_definer:table">
            1,"Test flag 1",2,,"TEST-FLAG",1,-1,100,"X1",0,
            2,"Test flag 1",2,,"TEST-FLAG_2",1,1,100,"X1",200,
            2,"Test flag 1",2,,"TEST-FLAG_2",2,-2,200,"X1",0,
            2,"Test flag 1",2,,"TEST-FLAG_2",2,-2,100,"Y1",0
        </Stream>
    </Table>
</LIGO_LW>
""".strip()


@pytest.fixture
def veto_definer(tmp_path):
    """Write the ``VETO_DEFINER_FILE`` to a temporary file and return."""
    tmp = tmp_path / "veto-definer.xml"
    tmp.write_text(VETO_DEFINER_FILE)
    return tmp


def veto_def(ifo, name, version, **kwargs):
    """Create a `igwn_ligolw.lsctables.VetoDef`."""
    from igwn_ligolw.lsctables import VetoDef

    vdef = VetoDef()
    kwargs["ifo"] = ifo
    kwargs["name"] = name
    kwargs["version"] = version
    for key in VetoDef.__slots__:
        setattr(vdef, key, kwargs.get(key))
    return vdef

# -- XML without _ns columns ---------

LIGOLW_NO_NS = """
<?xml version="1.0"?>
<!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt">
<LIGO_LW>
  <Table Name="segment_definergroup:segment_definer:table">
    <Column Name="segment_def_id" Type="int_8s"/>
    <Column Name="ifos" Type="lstring"/>
    <Column Name="name" Type="lstring"/>
    <Column Name="version" Type="int_4s"/>
    <Column Name="comment" Type="lstring"/>
    <Stream Name="segment_definergroup:segment_definer:table" Type="Local" Delimiter=",">
      0,"X1","TEST_FLAG",1,"Test flag",
    </Stream>
  </Table>
  <Table Name="segment_summarygroup:segment_summary:table">
    <Column Name="segment_sum_id" Type="int_8s"/>
    <Column Name="start_time" Type="int_4s"/>
    <Column Name="end_time" Type="int_4s"/>
    <Column Name="comment" Type="lstring"/>
    <Column Name="segment_definer:segment_def_id" Type="int_8s"/>
    <Stream Name="segment_summarygroup:segment_summary:table" Type="Local" Delimiter=",">
      0,1366644592,1366644608,"",0,
    </Stream>
  </Table>
  <Table Name="segmentgroup:segment:table">
    <Column Name="segment_id" Type="int_8s"/>
    <Column Name="start_time" Type="int_4s"/>
    <Column Name="end_time" Type="int_4s"/>
    <Column Name="segment_definer:segment_def_id" Type="int_8s"/>
    <Stream Name="segmentgroup:segment:table" Type="Local" Delimiter=",">
      0,1366644592,1366644593,0,
    </Stream>
  </Table>
</LIGO_LW>
""".strip()  # noqa: E501


@pytest.fixture
def ligolw_no_ns(tmp_path):
    """Write the LIGO_LW segment file without nanoseconds columns to a file."""
    tmp = tmp_path / "test.xml"
    tmp.write_text(LIGOLW_NO_NS)
    return tmp


# -- test data -----------------------

def _as_segmentlist(*segments: tuple[SupportsFloat, SupportsFloat]) -> SegmentList:
    """Return ``segments`` as a `SegmentList`."""
    return SegmentList([Segment(a, b) for a, b in segments])


NAME = "X1:TEST-FLAG_NAME:0"

# simple list of 'known' segments
KNOWN = _as_segmentlist(
    (0, 3), (6, 7))

# simple  list of 'active' segments
ACTIVE = _as_segmentlist(
    (1, 2), (3, 4), (5, 7))

# intersection of above 'known' and 'active' segments
KNOWNACTIVE = _as_segmentlist(
    (1, 2), (6, 7))

# 'active' set contracted by 0.1 seconds
ACTIVE_CONTRACTED = _as_segmentlist(
    (1.1, 1.9), (3.1, 3.9), (5.1, 6.9))

# 'active' seg protracted by 0.1 seconts
ACTIVE_PROTRACTED = _as_segmentlist(
    (.9, 2.1), (2.9, 4.1), (4.9, 7.1))

# some more segments
KNOWN2 = _as_segmentlist(
    (100, 150))

ACTIVE2 = _as_segmentlist(
    (100, 101), (110, 120))

# padding
PADDING = (-0.5, 1)

# padded version of above 'known' segments
KNOWNPAD = _as_segmentlist(
    (-.5, 4), (5.5, 8))

# padded version of above 'active' segments
ACTIVEPAD = _as_segmentlist(
    (.5, 3), (2.5, 5), (4.5, 8))

# padded, coalesed version of above 'active' segments
ACTIVEPADC = _as_segmentlist(
    (.5, 4), (5.5, 8))


# -- query helpers -------------------

QUERY_FLAGS = ["X1:TEST-FLAG:1", "Y1:TEST-FLAG2:4"]

QUERY_RESULT = DataQualityDict()

QUERY_RESULT["X1:TEST-FLAG:1"] = DataQualityFlag(
    "X1:TEST-FLAG:1",
    known=[(0, 10)],
    active=[(0, 1), (1, 2), (3, 4), (6, 9)])

QUERY_RESULT["Y1:TEST-FLAG2:4"] = DataQualityFlag(
    "Y1:TEST-FLAG2:4",
    known=[(0, 5), (9, 10)],
    active=[])

QUERY_RESULTC = type(QUERY_RESULT)({x: y.copy().coalesce() for
                                    x, y in QUERY_RESULT.items()})


def mock_query_segments(flag, start, end, **kwargs):
    """Mock `dqsegdb2.query.query_segments`."""
    try:
        ifo, name, version = flag.split(":")
        version = int(version)
    except ValueError:
        ifo, name = flag.split(":", 1)
        version = None
    span = SegmentList([Segment(start, end)])
    reflag = re.compile(flag)
    try:
        actual = next(filter(reflag.match, QUERY_RESULT))
    except StopIteration:  # empty
        response = Response()
        response.status_code = status_codes.NOT_FOUND
        response.reason = HTTPStatus(response.status_code).phrase
        response.raw = BytesIO()
        response.raw.write(br"<h1>{response.statue_code} {response.reason}</h1>")
        msg = (
            f"{response.status_code} Client Error: "
            f"{response.reason} for url: something"
        )
        raise HTTPError(msg, response=response)
    return {
        "ifo": ifo,
        "name": name,
        "version": version,
        "known": list(map(tuple, QUERY_RESULT[actual].known & span)),
        "active": list(map(tuple, QUERY_RESULT[actual].active & span)),
        "query_information": {},
        "metadata": kwargs,
    }


# -- DataQualityFlag -----------------

class TestDataQualityFlag:
    """Test `gwpy.segments.DataQualityFlag."""

    TEST_CLASS = DataQualityFlag

    @classmethod
    def create(cls, name=NAME, known=KNOWN, active=ACTIVE, **kwargs):
        """Create a new `DataQualityFlag` for testing."""
        return cls.TEST_CLASS(name=name, known=known, active=active, **kwargs)

    @pytest.fixture
    @classmethod
    def flag(cls):
        """Return a new `DataQualityFlag` for testing."""
        return cls.create()

    @pytest.fixture
    @classmethod
    def empty(cls):
        """Return a new, empty `DataQualityFlag` for testing."""
        return cls.TEST_CLASS()

    # -- test properties -------------

    def test_name(self, empty, flag):
        """Test ``name`` property handling."""
        assert empty.name is None

        assert flag.name == NAME
        assert flag.ifo == NAME.split(":")[0]
        assert flag.tag == NAME.split(":")[1]
        assert flag.version == int(NAME.split(":")[2])

    def test_known(self, flag):
        """Test ``known`` property handling."""
        utils.assert_segmentlist_equal(flag.known, KNOWN)

    def test_known_empty(self, empty):
        """Test empty ``known`` property handling."""
        assert isinstance(empty.known, SegmentList)
        assert empty.known == []

    def test_known_cast(self):
        """Test that the ``known`` property casts to `SegmentList`."""
        new = self.TEST_CLASS()
        new.known = [(1, 2), (3, 4)]  # type: ignore[assignment]
        assert isinstance(new.known, SegmentList)
        assert all(isinstance(seg, Segment) for seg in new.known)

    def test_active(self, flag):
        """Test ``active`` property handling."""
        utils.assert_segmentlist_equal(flag.active, ACTIVE)

    def test_active_empty(self, empty):
        """Test empty ``active`` property handling."""
        assert isinstance(empty.active, SegmentList)
        assert empty.active == []

    def test_active_cast(self):
        """Test that the ``active`` property casts to `SegmentList`."""
        new = self.TEST_CLASS()
        new.active = [(1, 2), (3, 4)]  # type: ignore[assignment]
        assert isinstance(new.active, SegmentList)
        assert all(isinstance(seg, Segment) for seg in new.active)

    @pytest.mark.parametrize(("name", "texname"), [
        (None, None),
        ("test", "test"),
        ("test_one_two", r"test\_one\_two"),
    ])
    def test_texname(self, name, texname):
        """Test ``texname`` property handling."""
        flag = self.TEST_CLASS(name=name)
        assert flag.texname == texname

    def test_extent(self):
        """Test ``extent`` property handling."""
        flag = self.TEST_CLASS(known=[(4, 6), (100, 101)])
        assert flag.extent == (4, 101)

    def test_extent_empty(self, empty):
        """Test ``extent`` property error handling."""
        with pytest.raises(
            ValueError,
            match="empty list",
        ):
            empty.extent  # noqa: B018

    @pytest.mark.parametrize(("active", "livetime"), [
        pytest.param([], 0, id="empty"),
        pytest.param([(1, 2), (3, 10)], 8, id="empty"),
    ])
    def test_livetime(self, active, livetime):
        """Test ``livetime`` property handling."""
        flag = self.TEST_CLASS(active=active)
        assert flag.livetime == livetime

    @pytest.mark.parametrize(("active", "known", "regular"), [
        pytest.param(
            [(0, 1), (3, 4)],
            [(0, 1), (2, 3), (3, 4)],
            True,
            id="regular",
        ),
        pytest.param(
            [(0, 1), (2, 3), (3, 4)],
            [(0, 1), (3, 4)],
            False,
            id="irregular",
        ),
    ])
    def test_regular(self, active, known, regular):
        """Test the ``regular`` property."""
        flag = self.TEST_CLASS(active=active, known=known)
        assert flag.regular is regular

    @pytest.mark.parametrize(("padding", "result"), [
        (None, (0, 0)),
        ((0, 0), (0, 0)),
        ([-1, 2], (-1, 2)),
    ])
    def test_padding(self, padding, result):
        """Test the ``padding`` property."""
        flag = self.TEST_CLASS(padding=padding)
        assert isinstance(flag.padding, tuple)
        assert flag.padding == result

        flag.padding = [-1, 2]  # type: ignore[assignment]
        assert isinstance(flag.padding, tuple)
        assert flag.padding == (-1, 2)

        flag.padding = None  # type: ignore[assignment]
        assert flag.padding == (0, 0)

    def test_padding_deleter(self):
        """Test ``del flag.padding``."""
        flag = self.TEST_CLASS(padding=(1, 2))
        assert flag.padding == (1, 2)
        del flag.padding
        assert flag.padding == (0, 0)

    # -- test methods ----------------

    @pytest.mark.parametrize(("flag", "name", "ifo", "tag", "version"), [
        (None, None, None, None, None),
        ("test", "test", None, None, None),
        ("L1:test", "L1:test", "L1", "test", None),
        ("L1:test:1", "L1:test:1", "L1", "test", 1),
    ])
    def test_parse_name(self, flag, name, ifo, tag, version):
        """Test name parsing."""
        flag = self.TEST_CLASS(flag)
        assert flag.name == name
        assert flag.ifo == ifo
        assert flag.tag == tag
        assert flag.version == version

    def test_plot(self, flag):
        """Test `DataQualityFlag.plot()`."""
        flag.label = "Test label"
        with rc_context(rc={"text.usetex": False}):
            plot = flag.plot(figsize=(6.4, 3.8))
            assert isinstance(plot.gca(), SegmentAxes)
            assert plot.gca().get_epoch() == flag.known[0][0]
            assert len(plot.gca().collections) == 2
            assert len(plot.gca().collections[1].get_paths()) == len(KNOWN)
            assert len(plot.gca().collections[0].get_paths()) == len(ACTIVE)
            assert plot.gca().collections[0].get_label() == flag.label

            plot.save(BytesIO(), format="png")
            plot.close()

    def test_plot_texname(self, flag):
        """Test `DataQualityFlag.plot()`."""
        flag.label = None
        with rc_context(rc={"text.usetex": True}):
            plot = flag.plot(figsize=(6.4, 3.8))
            assert plot.gca().collections[0].get_label() == flag.texname
            plot.close()

    def test_and(self):
        """Test ``a & b``."""
        a = self.TEST_CLASS(active=ACTIVE[:2], known=KNOWN)
        b = self.TEST_CLASS(active=ACTIVE[2:], known=KNOWN)
        c = a & b
        utils.assert_segmentlist_equal(c.active, a.active & b.active)
        utils.assert_segmentlist_equal(c.known, a.known & b.known)

    def test_sub(self):
        """Test ``a - b``."""
        a = self.TEST_CLASS(active=ACTIVE[:2], known=KNOWN)
        b = self.TEST_CLASS(active=ACTIVE[2:], known=KNOWN)
        c = a - b
        utils.assert_segmentlist_equal(c.known, a.known & b.known)
        utils.assert_segmentlist_equal(c.active, (a.active - b.active) & c.known)

    def test_or(self):
        """Test ``a | b``."""
        a = self.TEST_CLASS(active=ACTIVE[:2], known=KNOWN)
        b = self.TEST_CLASS(active=ACTIVE[2:], known=KNOWN)
        c = a | b
        utils.assert_segmentlist_equal(c.active, a.active | b.active)
        utils.assert_segmentlist_equal(c.known, a.known | b.known)

    def test_invert(self):
        """Test ``~a``."""
        a = self.TEST_CLASS(active=ACTIVE[:2], known=KNOWN)
        b = ~a
        utils.assert_segmentlist_equal(b.active, a.known & ~a.active)
        utils.assert_segmentlist_equal(b.known, a.known)

    def test_difference_simple(self):
        """Test that subtract works as intended for a simple case.

        Tests regression against https://github.com/gwpy/gwpy/issues/1700.
        """
        known1 = _as_segmentlist((0, 2), (3, 7))
        active1 = _as_segmentlist((1, 2), (3, 4), (5, 7))

        known2 = _as_segmentlist((3, 7), (8, 10))
        active2 = _as_segmentlist((4, 7), (9, 10))

        a = self.TEST_CLASS(active=active1, known=known1)
        b = self.TEST_CLASS(active=active2, known=known2)

        diff = a - b

        expected_known = _as_segmentlist((3, 7))
        expected_active = _as_segmentlist((3, 4))

        expected_diff = self.TEST_CLASS(
            active=expected_active,
            known=expected_known,
        )

        utils.assert_flag_equal(diff, expected_diff)

    def test_coalesce(self):
        """Test `DataQualityFlag.coalesce()`."""
        flag = self.create()
        flag.coalesce()
        utils.assert_segmentlist_equal(flag.known, KNOWN)
        utils.assert_segmentlist_equal(flag.active, KNOWNACTIVE)
        assert flag.regular is True

    def test_contract(self):
        """Test `DataQualityFlag.contract()`."""
        flag = self.create()
        flag.contract(.1)
        utils.assert_segmentlist_equal(flag.known, KNOWN)
        utils.assert_segmentlist_equal(flag.active, ACTIVE_CONTRACTED)

    def test_protract(self):
        """Test `DataQualityFlag.protract()`."""
        flag = self.create(active=ACTIVE_CONTRACTED)
        flag.protract(.1)
        utils.assert_segmentlist_equal(flag.known, KNOWN)
        utils.assert_segmentlist_equal(flag.active, ACTIVE)

    @pytest.mark.parametrize(("contract", "active"), [
        (False, ACTIVE_CONTRACTED),
        (True, ACTIVE_PROTRACTED),
    ])
    def test_round(self, contract, active):
        """Test `DataQualityFlag.round()`."""
        flag = self.create(active=active)
        r = flag.round(contract=contract)
        utils.assert_segmentlist_equal(r.known, KNOWN)
        utils.assert_segmentlist_equal(r.active, KNOWNACTIVE)

    def test_pad_noop(self, flag):
        """Test `DataQualityFlag.pad()`."""
        # test with no arguments (and no padding)
        padded = flag.pad()
        utils.assert_flag_equal(flag, padded)

    def test_pad_noargs(self, flag):
        """Test `DataQualityFlag.pad()` using ``flag.padding``."""
        flag.padding = PADDING
        padded = flag.pad()
        utils.assert_segmentlist_equal(padded.known, KNOWNPAD)
        utils.assert_segmentlist_equal(padded.active, ACTIVEPAD)

    def test_pad_args(self, flag):
        """Test `DataQualityFlag.pad()` with arguments."""
        flag.padding = (-100, 100)
        padded = flag.pad(*PADDING)
        utils.assert_segmentlist_equal(padded.known, KNOWNPAD)
        utils.assert_segmentlist_equal(padded.active, ACTIVEPAD)

    def test_pad_inplace(self, flag):
        """Test `DataQualityFlag.pad(..., inplace=...)`."""
        padded = flag.pad(*PADDING, inplace=False)
        assert padded is not flag

        padded = flag.pad(*PADDING, inplace=True)
        assert padded is flag
        utils.assert_segmentlist_equal(flag.known, KNOWNPAD)
        utils.assert_segmentlist_equal(flag.active, ACTIVEPAD)

        # check that other keyword arguments get rejected appropriately
        with pytest.raises(TypeError):
            flag.pad(*PADDING, kwarg="test")

    @pytest.mark.requires("igwn_ligolw.lsctables")
    def test_from_veto_def(self):
        """Test `DataQualityFlag.from_veto_def()`."""
        a = veto_def(
            "X1",
            "TEST-FLAG",
            1,
            start_time=0,
            end_time=0,
            start_pad=-2,
            end_pad=2,
            comment="Comment",
        )
        f = self.TEST_CLASS.from_veto_def(a)
        assert f.name == "X1:TEST-FLAG:1"
        assert f.category is None
        assert f.padding == (-2, 2)
        assert f.description == "Comment"
        utils.assert_segmentlist_equal(f.known, [(0, float("inf"))])

    @pytest.mark.requires("igwn_ligolw.lsctables")
    def test_from_veto_def_version(self):
        """Test `DataQualityFlag.from_veto_def()` with missing version."""
        a = veto_def("X1", "TEST-FLAG", None, start_time=0, end_time=1)
        f = self.TEST_CLASS.from_veto_def(a)
        assert f.name == "X1:TEST-FLAG"
        assert f.version is None

    @mock.patch("gwpy.segments.flag.query_segments", mock_query_segments)
    def test_populate(self):
        """Test `DataQualityFlag.populate()`."""
        name = QUERY_FLAGS[0]
        flag = self.TEST_CLASS(name, known=QUERY_RESULT[name].known)
        flag.populate()
        utils.assert_flag_equal(flag, QUERY_RESULTC[name])

    # -- test I/O --------------------

    @pytest.mark.parametrize(("format_", "ext", "rw_kwargs", "simple"), [
        ("hdf5", "hdf5", {"path": "test-dqflag"}, False),
        ("hdf5", "h5", {"path": "test-dqflag"}, False),
        ("json", "json", {}, True),
    ])
    def test_read_write(self, flag, format_, ext, rw_kwargs, simple):
        """Test writing and reading back a `DataQualityFlag`."""
        # simplify calling read/write tester
        def _read_write(**kwargs):
            read_kw = rw_kwargs.copy()
            read_kw.update(kwargs.pop("read_kw", {}))
            write_kw = rw_kwargs.copy()
            write_kw.update(kwargs.pop("write_kw", {}))
            return utils.test_read_write(
                flag,
                format_,
                extension=ext,
                assert_equal=utils.assert_flag_equal,
                read_kw=read_kw,
                write_kw=write_kw,
                **kwargs,
            )

        # perform simple test
        if simple:
            _read_write()

        # perform complicated test
        else:
            _read_write(autoidentify=False)
            with pytest.raises(
                OSError,
                match="test",
            ):
                _read_write(autoidentify=True)
            _read_write(autoidentify=True, write_kw={"overwrite": True})

    def test_read_write_hdf5(self, flag, tmp_path):
        """Test writing and reading back a `DataQualityFlag` in HDF5 format."""
        tmp = tmp_path / "test.h5"
        flag.write(tmp, path="test")
        f2 = self.TEST_CLASS.read(tmp, path="test", format="hdf5")
        utils.assert_flag_equal(f2, flag)

        # test direct access from dataset
        with h5py.File(tmp, "r") as h5f:
            f2 = self.TEST_CLASS.read(h5f["test"])
        utils.assert_flag_equal(f2, flag)

        # test auto-discover of single dataset
        f2 = self.TEST_CLASS.read(tmp)
        utils.assert_flag_equal(f2, flag)

    @pytest.mark.requires("igwn_ligolw.lsctables")
    def test_read_write_ligolw(self, flag):
        """Test writing and reading back a `DataQualityFlag` in LIGO_LW format."""
        utils.test_read_write(
            flag,
            "ligolw",
            extension="xml",
            assert_equal=utils.assert_flag_equal,
            autoidentify=False,
            read_kw={},
        )

    @pytest.mark.requires("igwn_ligolw.lsctables")
    def test_write_ligolw_attrs(self, tmp_path, flag):
        """Test writing a `DataQualityFlag` in LIGO_LW format with attributes."""
        from gwpy.io.ligolw import read_table
        tmp = tmp_path / "tmp.xml"
        flag.write(
            tmp,
            format="ligolw",
            attrs={"process_id": 100},
        )
        segdeftab = read_table(tmp, "segment_definer")
        assert int(segdeftab[0].process_id) == 100

    @pytest.mark.requires("igwn_ligolw.lsctables")
    def test_read_ligolw_no_ns(self, ligolw_no_ns):
        """Test writing and reading back a `DataQualityFlag` in LIGO_LW format.

        Specifically when the file doesn't populate nanosecond columns.
        """
        flag = self.TEST_CLASS.read(ligolw_no_ns, format="ligolw")
        assert flag.name == "X1:TEST_FLAG:1"
        assert flag.known == [(1366644592, 1366644608)]
        assert flag.active == [(1366644592, 1366644593)]

    # -- test queries ----------------

    @mock.patch("gwpy.segments.flag.query_segments", mock_query_segments)
    def test_query(self):
        """Test `DataQualityFlag.query()`."""
        result = self.TEST_CLASS.query(QUERY_FLAGS[0], 0, 10)
        assert isinstance(result, self.TEST_CLASS)
        expected = QUERY_RESULT[QUERY_FLAGS[0]].copy().coalesce()
        utils.assert_segmentlist_equal(result.known, expected.known)
        utils.assert_segmentlist_equal(result.active, expected.active)

    @pytest.mark.parametrize(("name", "flag"), [
        pytest.param(
            QUERY_FLAGS[0],
            QUERY_FLAGS[0],
            id="normal",
        ),
        pytest.param(
            QUERY_FLAGS[0].rsplit(":", 1)[0],
            QUERY_FLAGS[0],
            id="versionless",
        ),
    ])
    @mock.patch("gwpy.segments.flag.query_segments", mock_query_segments)
    def test_query_dqsegdb(self, name, flag):
        """Test `DataQualityFlag.query_dqsegdb()`."""
        # standard query
        result = self.TEST_CLASS.query_dqsegdb(name, 0, 10)
        expected = QUERY_RESULTC[flag]
        assert isinstance(result, self.TEST_CLASS)
        utils.assert_segmentlist_equal(result.known, expected.known)
        utils.assert_segmentlist_equal(result.active, expected.active)

        # segment as tuple
        result2 = self.TEST_CLASS.query_dqsegdb(name, (0, 10))
        utils.assert_flag_equal(result, result2)

        # segmentlist
        result2 = self.TEST_CLASS.query_dqsegdb(name, SegmentList([(0, 10)]))
        utils.assert_flag_equal(result, result2)

    def test_query_dqsegdb_error_flag_name(self):
        """Test `DataQualityFlag.query_dqsegdb()` flag name parsing error."""
        with pytest.raises(
            ValueError,
            match=r"Cannot parse ifo or tag \(name\) for flag 'BAD-FLAG_NAME'",
        ):
            self.TEST_CLASS.query_dqsegdb(
                "BAD-FLAG_NAME",
                SegmentList([(0, 10)]),
            )

    @mock.patch("gwpy.segments.flag.query_segments", mock_query_segments)
    def test_query_dqsegdb_not_found(self):
        """Test `DataQualityFlag.query_dqsegdb()` 404 annotation."""
        # flag not in database
        with pytest.raises(
            HTTPError,
            match=r"^404 Client Error: Not Found .* \[X1:GWPY-TEST:0\]$",
        ):
            self.TEST_CLASS.query_dqsegdb("X1:GWPY-TEST:0", 0, 10)

    @pytest.mark.parametrize("args", [
        (1, 2, 3),
        ((1, 2, 3),),
    ])
    def test_query_dqsegdb_args(self, args):
        """Test `DataQualityFlag.query_dqsegdb()` response to bad arguments."""
        # bad syntax
        with pytest.raises(
            ValueError,
            match=r"query_dqsegdb\(\) takes",
        ):
            self.TEST_CLASS.query_dqsegdb(QUERY_FLAGS[0], *args)

    @mock.patch("gwpy.segments.flag.query_segments", mock_query_segments)
    def test_query_dqsegdb_multi(self):
        """Test `DataQualityFlag.query_dqsegdb()` with multiple segments."""
        segs = SegmentList([Segment(0, 2), Segment(8, 10)])
        result = self.TEST_CLASS.query_dqsegdb(QUERY_FLAGS[0], segs)
        expected = QUERY_RESULTC[QUERY_FLAGS[0]]

        assert isinstance(result, self.TEST_CLASS)
        utils.assert_segmentlist_equal(result.known, expected.known & segs)
        utils.assert_segmentlist_equal(result.active, expected.active & segs)

    @pytest_skip_flaky_network
    def test_fetch_open_data(self):
        """Test `DataQualityFlag.fetch_open_data()`."""
        segs = self.TEST_CLASS.fetch_open_data(
            "H1_DATA",
            946339215,
            946368015,
        )
        assert segs.ifo == "H1"
        assert segs.name == "H1:DATA"
        assert segs.label == "H1_DATA"
        utils.assert_segmentlist_equal(segs.known, [(946339215, 946368015)])
        utils.assert_segmentlist_equal(segs.active, [
            (946340946, 946351799),
            (946356479, 946360619),
            (946362652, 946368015),
        ])


# -- DataQualityDict -----------------

class TestDataQualityDict:
    """Test `gwpy.segments.DataQualityDict."""

    TEST_CLASS = DataQualityDict
    ENTRY_CLASS = DataQualityFlag

    @classmethod
    def create(cls):
        """Create a `DataQualityDict`."""
        flgd = cls.TEST_CLASS()
        flgd["X1:TEST-FLAG:1"] = cls.ENTRY_CLASS(name="X1:TEST-FLAG:1",
                                                 active=ACTIVE, known=KNOWN)
        flgd["Y1:TEST-FLAG:2"] = cls.ENTRY_CLASS(name="Y1:TEST-FLAG:2",
                                                 active=ACTIVE2, known=KNOWN2)
        return flgd

    @pytest.fixture
    @classmethod
    def instance(cls):
        """Create a `DataQualityDict` (fixture)."""
        return cls.create()

    @pytest.fixture
    @classmethod
    def reverse(cls):
        """Create a `DataQualityDict` that is the reverse of ``instance``."""
        inst = cls.create()
        rev = type(inst)()
        keys = list(inst.keys())
        rev[keys[0]] = inst[keys[1]]
        rev[keys[1]] = inst[keys[0]]
        return rev

    # -- test logic ------------------

    def test_iand(self, instance, reverse):
        """Test ``a &= b``."""
        a = instance.copy()
        a &= reverse
        keys = list(a.keys())
        for key in keys:
            utils.assert_flag_equal(a[key], instance[key] & reverse[key])

    def test_and(self, instance, reverse):
        """Test ``a & b``."""
        a = instance.copy()
        a &= reverse
        utils.assert_dict_equal(a, instance & reverse, utils.assert_flag_equal)

    def test_ior(self, instance, reverse):
        """Test ``a |= b``."""
        a = instance.copy()
        a |= reverse
        keys = list(a.keys())
        for key in keys:
            utils.assert_flag_equal(a[key], instance[key] | reverse[key])

    def test_or(self, instance, reverse):
        """Test ``a | b``."""
        a = instance.copy()
        a |= reverse
        utils.assert_dict_equal(a, instance | reverse, utils.assert_flag_equal)

    def test_isub(self, instance, reverse):
        """Test ``a -= b``."""
        a = instance.copy()
        a -= reverse
        keys = list(a.keys())
        for key in keys:
            utils.assert_flag_equal(a[key], instance[key] - reverse[key])

    def test_sub(self, instance, reverse):
        """Test ``a - b``."""
        a = instance.copy(deep=True)
        a -= reverse
        utils.assert_dict_equal(a, instance - reverse, utils.assert_flag_equal)

    def test_ixor(self, instance, reverse):
        """Test ``a ^= b``."""
        a = instance.copy(deep=True)
        a ^= reverse
        utils.assert_dict_equal(a, instance ^ reverse, utils.assert_flag_equal)

    def test_invert(self, instance):
        """Test ``~a``."""
        inverse = type(instance)()
        for key in instance:
            inverse[key] = ~instance[key]
        utils.assert_dict_equal(~instance, inverse, utils.assert_flag_equal)

    # -- test methods ----------------

    def test_union(self, instance):
        """Test `DataQualityDict.union()`."""
        union = instance.union()
        assert isinstance(union, self.ENTRY_CLASS)
        utils.assert_segmentlist_equal(union.known, KNOWN + KNOWN2)
        utils.assert_segmentlist_equal(union.active, ACTIVE + ACTIVE2)

    def test_intersection(self, instance):
        """Test `DataQualityDict.intersection()`."""
        intersection = instance.intersection()
        assert isinstance(intersection, self.ENTRY_CLASS)
        utils.assert_segmentlist_equal(intersection.known, KNOWN & KNOWN2)
        utils.assert_segmentlist_equal(intersection.active, ACTIVE & ACTIVE2)

    def test_plot(self, instance):
        """Test `DataQualityDict.plot()`."""
        with rc_context(rc={"text.usetex": False}):
            plot = instance.plot(figsize=(6.4, 3.8))
            assert isinstance(plot.gca(), SegmentAxes)
            assert plot.get_figwidth() == 6.4
            assert plot.get_figheight() == 3.8
            plot.save(BytesIO(), format="png")
            plot.close()

    def test_plot_known_label(self, instance):
        """Test `DataQualityDict.plot(..., known=None, label=...)`."""
        with rc_context(rc={"text.usetex": False}):
            plot = instance.plot(
                known=None,
                figsize=(6.4, 3.8),
            )
            ax = plot.gca()
            for key, artist in zip(instance, ax.collections, strict=True):
                assert artist.get_label() == instance[key].name
            plot.save(BytesIO(), format="png")
            plot.close()

    def test_plot_label(self, instance):
        """Test `DataQualityDict.plot(..., label=...)`."""
        with rc_context(rc={"text.usetex": False}):
            plot = instance.plot(
                label="Fixed label",
                figsize=(6.4, 3.8),
            )
            ax = plot.gca()
            for artist in ax.collections:
                assert artist.get_label() == "Fixed label"
            plot.save(BytesIO(), format="png")
            plot.close()

    # -- test I/O --------------------

    @pytest.mark.requires("igwn_ligolw.lsctables")
    def test_from_veto_definer_file(self, veto_definer):
        """Test `DataQualityDict.from_veto_definer_file()`."""
        # read veto definer
        vdf = self.TEST_CLASS.from_veto_definer_file(veto_definer)
        assert len(vdf.keys()) == 4

        # test one flag to make sure it is well read
        name = "X1:TEST-FLAG:1"
        assert name in vdf
        utils.assert_segmentlist_equal(vdf[name].known,
                                       [(100, float("inf"))])
        assert vdf[name].category == 1
        assert vdf[name].padding == (-1, 2)

        # test ifo kwarg
        vdf = self.TEST_CLASS.from_veto_definer_file(veto_definer, ifo="X1")
        assert len(vdf.keys()) == 3
        assert "Y1:TEST-FLAG_2:2" not in vdf

        # test start and end kwargs
        vdf = self.TEST_CLASS.from_veto_definer_file(veto_definer,
                                                     start=200, end=300)
        assert len(vdf.keys()) == 3
        assert "X1:TEST-FLAG_2:1" not in vdf

    @pytest.mark.parametrize(("format_", "ext", "rw_kwargs"), [
        ("hdf5", "hdf5", {}),
        ("hdf5", "h5", {}),
        ("hdf5", "hdf5", {"path": "test-dqdict"}),
    ])
    def test_read_write(self, instance, format_, ext, rw_kwargs):
        """Test writing and reading back a `DataQualityDict`."""
        # define assertion
        def _assert(a, b):
            return utils.assert_dict_equal(a, b, utils.assert_flag_equal)

        # simplify calling read/write tester
        def _read_write(**kwargs):
            read_kw = rw_kwargs.copy()
            read_kw.update(kwargs.pop("read_kw", {}))
            write_kw = rw_kwargs.copy()
            write_kw.update(kwargs.pop("write_kw", {}))
            return utils.test_read_write(
                instance,
                format_,
                extension=ext,
                assert_equal=_assert,
                read_kw=read_kw,
                write_kw=write_kw,
                **kwargs,
            )

        _read_write(autoidentify=False)

        # test that attempting to read/write again fails
        with pytest.raises(
            OSError,
            match="File exists:",
        ):
            _read_write(autoidentify=True)

        # unless overwrite=True is given
        _read_write(autoidentify=True, write_kw={"overwrite": True})

    @pytest.mark.requires("igwn_ligolw.lsctables")
    def test_read_write_ligolw(self, instance):
        """Test writing and reading back a `DataQualityDict` in LIGO_LW format."""
        def _assert(a, b):
            return utils.assert_dict_equal(a, b, utils.assert_flag_equal)

        utils.test_read_write(
            instance,
            "ligolw",
            extension="xml",
            assert_equal=_assert,
            autoidentify=False,
            read_kw={},
        )

    def test_read_on_missing(self, instance):
        """Test `DataQualityDict.read(..., on_missing=...)`."""
        with h5py.File(
                "test",
                mode="w-",
                driver="core",
                backing_store=False,
        ) as h5f:
            instance.write(h5f)
            names = ["randomname"]

            def _read(**kwargs):
                return self.TEST_CLASS.read(
                    h5f,
                    names=names,
                    format="hdf5",
                    **kwargs,
                )

            # check on_missing='error' (default) raises ValueError
            with pytest.raises(
                ValueError,
                match=r"^'randomname' not found in any input file$",
            ):
                _read()

            # check on_missing='warn' prints warning
            with pytest.warns(UserWarning):
                _read(on_missing="warn")

            # check on_missing='ignore' does nothing
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _read(on_missing="ignore")

            # check on_missing=<anything else> raises exception
            with pytest.raises(
                ValueError,
                match=r"^'randomname' not found in any input file$",
            ):
                _read(on_missing="blah")

    @pytest.mark.requires("igwn_ligolw.lsctables")
    def test_to_ligolw_tables(self, instance):
        """Test `DataQualityDict.to_ligolw_tables()`."""
        tables = instance.to_ligolw_tables()
        assert len(tables[0]) == len(instance)  # segdef
        assert len(tables[1]) == sum(len(x.known) for x in instance.values())
        assert len(tables[2]) == sum(len(x.active) for x in instance.values())

    # -- test queries ----------------

    @mock.patch("gwpy.segments.flag.query_segments", mock_query_segments)
    def test_query(self):
        """Test `DataQualityDict.query()`."""
        result = self.TEST_CLASS.query(QUERY_FLAGS, 0, 10)
        expected = QUERY_RESULT.copy().coalesce()

        assert isinstance(result, self.TEST_CLASS)
        utils.assert_dict_equal(result, expected, utils.assert_flag_equal)

    @mock.patch("gwpy.segments.flag.query_segments", mock_query_segments)
    def test_query_dqsegdb(self):
        """Test `DataQualityDict.query_dqsegdb()`."""
        result = self.TEST_CLASS.query_dqsegdb(QUERY_FLAGS, 0, 10)
        assert isinstance(result, self.TEST_CLASS)
        utils.assert_dict_equal(
            result,
            QUERY_RESULTC,
            utils.assert_flag_equal,
        )

    @pytest.mark.parametrize(("on_error", "ctx"), [
        # does nothing
        ("ignore", nullcontext()),
        # emits a warning
        ("warn", pytest.warns(UserWarning)),
        # propagates the exception from the thread
        ("raise", pytest.raises(
            HTTPError,
            match=r"404 Client Error: Not Found for .* \[X1:BLAHBLAH:1\]",
        )),
        # invalid value
        ("blah", pytest.raises(ValueError, match="on_error must be one of")),
    ])
    @mock.patch("gwpy.segments.flag.query_segments", mock_query_segments)
    def test_query_dqsegdb_on_error(self, on_error, ctx):
        """Test `DataQualityDict.query_dqsegdb(..., on_error=...)`."""
        with ctx as record:
            result = self.TEST_CLASS.query_dqsegdb(
                [*QUERY_FLAGS, "X1:BLAHBLAH:1"],
                0,
                10,
                on_error=on_error,
                nthreads=2,
            )
        if on_error == "warn":
            assert len(record) == 1  # check on_error='ignore' didn't warn
        elif on_error != "ignore":
            return  # exception has been asserted
        assert isinstance(result, self.TEST_CLASS)
        utils.assert_dict_equal(
            result,
            QUERY_RESULTC,
            utils.assert_flag_equal,
        )

    @mock.patch("gwpy.segments.flag.query_segments", mock_query_segments)
    def test_populate(self):
        """Test `DataQualityDict.populate()`."""
        def fake():
            return self.TEST_CLASS({
                x: self.ENTRY_CLASS(name=x, known=y.known)
                for x, y in QUERY_RESULT.items()
            })

        # build fake veto definer file
        vdf = fake()
        vdf2 = fake()
        vdf3 = fake()

        flag = QUERY_FLAGS[0]
        vdf2[flag].padding = (-1, 1)

        span = SegmentList([Segment(0, 2)])

        # and populate using a mocked query
        vdf.populate()
        vdf2.populate()
        vdf3.populate(segments=span)

        # test warnings on bad entries
        vdf["TEST"] = self.ENTRY_CLASS("X1:BLAHBLAHBLAH:1", known=[(0, 1)])
        with pytest.warns(UserWarning) as record:
            vdf.populate(on_error="warn")
            vdf.populate(on_error="ignore")
        assert len(record) == 1
        vdf.pop("TEST")

        with pytest.raises(
            ValueError,
            match="on_error must be one of",
        ):
            vdf.populate(on_error="blah")

        # check basic populate worked
        utils.assert_dict_equal(vdf, QUERY_RESULTC, utils.assert_flag_equal)

        # check padded populate worked
        utils.assert_flag_equal(vdf2[flag], QUERY_RESULTC[flag].pad(-1, 1))

        # check segment-restricted populate worked
        for flag in vdf3:
            utils.assert_segmentlist_equal(
                vdf3[flag].known, QUERY_RESULTC[flag].known & span)
            utils.assert_segmentlist_equal(
                vdf3[flag].active, QUERY_RESULTC[flag].active & span)

    def test_coalesce(self):
        """Test `DataQualityDict.coalesce()`."""
        instance = self.create()
        instance.coalesce()
        value = instance["X1:TEST-FLAG:1"]
        utils.assert_segmentlist_equal(value.active, KNOWNACTIVE)
