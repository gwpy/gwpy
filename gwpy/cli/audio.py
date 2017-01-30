
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) Derek Davis (2017)
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
#

""" Audio Files
"""
from cliproduct import CliProduct


class Audio(CliProduct):

    def get_action(self):
        """Return the string used as "action" on command line."""
        return 'audio'

    def init_cli(self, parser):
        """Set up the argument list for this product"""
        self.arg_chan1(parser)
        return

    def get_max_datasets(self):
        """Audio only handles 1 at a time""" #Currently
        return 1

    def get_title(self):
        """Start of default super title, first channel is appended to it"""
        return 'Audio : '

    def gen_plot(self, args):
        """Generate the plot from time series and arguments
           In this 'plot' is an audio file that is saved
        """

        if arg_list.out:
            out_file = arg_list.out
        else:
            out_file = "./gwpy.wav"       

	self.timeseries[0].wavwrite(out_file)


        return

