# -*- coding: utf-8 -*-
# Copyright (C) California Institute of Technology (2022)
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

"""Tests for :mod:`gwpy.table.io.gstlal`
"""

import pytest

from numpy import testing as nptest
from numpy import float32

from ..io import gstlal as gstlalio
from gwpy.table import EventTable

__author__ = 'Derek Davis <derek.davis@ligo.org>'


# -- gstlal file fixture -----------------------------------------------------


GSTLAL_FILE = """<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt">
<LIGO_LW>
        <Table Name="coinc_inspiral:table">
                <Column Name="coinc_event:coinc_event_id" Type="int_8s"/>
                <Column Name="combined_far" Type="real_8"/>
                <Column Name="end_time" Type="int_4s"/>
                <Column Name="end_time_ns" Type="int_4s"/>
                <Column Name="false_alarm_rate" Type="real_8"/>
                <Column Name="ifos" Type="lstring"/>
                <Column Name="mass" Type="real_8"/>
                <Column Name="mchirp" Type="real_8"/>
                <Column Name="minimum_duration" Type="real_8"/>
                <Column Name="snr" Type="real_8"/>
                <Stream Name="coinc_inspiral:table" Delimiter="," Type="Local">
                        1,1,100,0,1,"H1,L1",1,1,1,1,
                        2,1,100,0,1,"H1,L1",1,1,1,1,
                </Stream>
        </Table>
        <Table Name="coinc_event:table">
                <Column Name="coinc_definer:coinc_def_id" Type="int_8s"/>
                <Column Name="coinc_event_id" Type="int_8s"/>
                <Column Name="instruments" Type="lstring"/>
                <Column Name="likelihood" Type="real_8"/>
                <Column Name="nevents" Type="int_4u"/>
                <Column Name="process:process_id" Type="int_8s"/>
                <Column Name="time_slide:time_slide_id" Type="int_8s"/>
                <Stream Name="coinc_event:table" Delimiter="," Type="Local">
                        0,1,"H1,L1",1,1,0,0,
                        1,1,"H1,L1",1,1,0,0
                </Stream>
        </Table>
        <Table Name="sngl_inspiral:table">
                <Column Name="process:process_id" Type="int_8s"/>
                <Column Name="ifo" Type="lstring"/>
                <Column Name="end_time" Type="int_4s"/>
                <Column Name="end_time_ns" Type="int_4s"/>
                <Column Name="eff_distance" Type="real_4"/>
                <Column Name="coa_phase" Type="real_4"/>
                <Column Name="mass1" Type="real_4"/>
                <Column Name="mass2" Type="real_4"/>
                <Column Name="snr" Type="real_4"/>
                <Column Name="chisq" Type="real_4"/>
                <Column Name="chisq_dof" Type="int_4s"/>
                <Column Name="bank_chisq" Type="real_4"/>
                <Column Name="bank_chisq_dof" Type="int_4s"/>
                <Column Name="sigmasq" Type="real_8"/>
                <Column Name="spin1x" Type="real_4"/>
                <Column Name="spin1y" Type="real_4"/>
                <Column Name="spin1z" Type="real_4"/>
                <Column Name="spin2x" Type="real_4"/>
                <Column Name="spin2y" Type="real_4"/>
                <Column Name="spin2z" Type="real_4"/>
                <Column Name="template_duration" Type="real_8"/>
                <Column Name="event_id" Type="int_8s"/>
                <Column Name="Gamma0" Type="real_4"/>
                <Column Name="Gamma1" Type="real_4"/>
                <Column Name="Gamma2" Type="real_4"/>
                <Stream Name="sngl_inspiral:table" Delimiter="," Type="Local">
                        1,"L1",100,0,nan,0,10,10,3,2,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,
                        1,"H1",100,0,nan,0,10,10,3,2,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,
                        1,"V1",100,0,nan,0,10,10,3,2,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1
                </Stream>
        </Table>
</LIGO_LW>
"""  # noqa: E501


@pytest.fixture
def gstlal_table(tmp_path):
    tmp = tmp_path / "H1L1V1-LLOID-1-1.xml.gz"
    tmp.write_text(GSTLAL_FILE)
    return tmp


# -- test data ----------------------------------------------------------------

SNGL_LEN = 3
COINC_LEN = 2


@pytest.mark.requires("ligo.lw.lsctables")
def test_sngl_function(gstlal_table):
    table = gstlalio.read_gstlal_sngl(gstlal_table)
    assert len(table) == SNGL_LEN


@pytest.mark.requires("ligo.lw.lsctables")
def test_read_sngl_eventtable(gstlal_table):
    table = EventTable.read(gstlal_table, format='ligolw.gstlal',
                            triggers='sngl')
    assert len(table) == SNGL_LEN


@pytest.mark.requires("ligo.lw.lsctables")
def test_read_sngl_format(gstlal_table):
    table = EventTable.read(gstlal_table, format='ligolw.gstlal.sngl')
    assert len(table) == SNGL_LEN


@pytest.mark.requires("ligo.lw.lsctables")
def test_read_sngl_columns(gstlal_table):
    table = EventTable.read(gstlal_table, format='ligolw.gstlal.sngl',
                            columns=['snr', 'end_time'])
    assert list(table.keys()) == ['snr', 'end_time']


@pytest.mark.requires("ligo.lw.lsctables")
def test_coinc_function(gstlal_table):
    table = gstlalio.read_gstlal_coinc(gstlal_table)
    assert len(table) == COINC_LEN


@pytest.mark.requires("ligo.lw.lsctables")
def test_read_coinc_eventtable(gstlal_table):
    table = EventTable.read(gstlal_table, format='ligolw.gstlal',
                            triggers='coinc')
    assert len(table) == COINC_LEN


@pytest.mark.requires("ligo.lw.lsctables")
def test_read_coinc_format(gstlal_table):
    table = EventTable.read(gstlal_table, format='ligolw.gstlal.coinc')
    assert len(table) == COINC_LEN


@pytest.mark.requires("ligo.lw.lsctables")
def test_read_coinc_columns(gstlal_table):
    table = EventTable.read(gstlal_table, format='ligolw.gstlal.coinc',
                            columns=['snr', 'end_time'])
    assert list(table.keys()) == ['snr', 'end_time']


@pytest.mark.requires("ligo.lw.lsctables")
def test_derived_values(gstlal_table):
    table = EventTable.read(gstlal_table, format='ligolw.gstlal',
                            triggers='sngl',
                            columns=['snr_chi', 'chi_snr', 'mchirp'])
    nptest.assert_almost_equal(
        table['snr_chi'][0], 4.5)
    nptest.assert_almost_equal(
        table['chi_snr'][0], 1./4.5)
    nptest.assert_almost_equal(
        table['mchirp'][0], float32(8.705506))


@pytest.mark.requires("ligo.lw.lsctables")
def test_incorrect_sngl_column(gstlal_table):
    with pytest.raises(
         ValueError,
         match="is not a valid column name",
         ):
        EventTable.read(gstlal_table, format='ligolw.gstlal.sngl',
                        columns=['nan'])


@pytest.mark.requires("ligo.lw.lsctables")
def test_incorrect_coinc_column(gstlal_table):
    with pytest.raises(
         ValueError,
         match="is not a valid column name",
         ):
        EventTable.read(gstlal_table, format='ligolw.gstlal.coinc',
                        columns=['nan'])


@pytest.mark.requires("ligo.lw.lsctables")
def test_incorrect_trigger_name(gstlal_table):
    with pytest.raises(
         ValueError,
         match="^The 'triggers' argument",
         ):
        EventTable.read(gstlal_table, format='ligolw.gstlal',
                        triggers='nan')
