# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017)
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

"""Extend :mod:`astropy.table` with the `GravitySpyTable`
"""

import os, sys

import pandas as pd
from sqlalchemy.engine import create_engine

from .table import EventTable
from astropy.table import Table

from ..utils import mp as mp_utils

from xml.sax import SAXException

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'
__all__ = ['GravitySpyTable']

class GravitySpyTable(EventTable):
    """A container for a table of Gravity Spy Events (as well as
    Events from the O1 Glitch Classification Paper whcih includes

    - PCAT
    - PC-LIB
    - WDF
    - WDNN
    - Karoo GP

    This differs from the basic `~astropy.table.Table` in two ways

    - GW-specific file formats are registered to use with
      `GravitySpyTable.fetch`
    - columns of this table are of the `EventColumn` type, which provides
      methods for filtering based on a `~gwpy.segments.SegmentList` (not
      specifically time segments)

    See also
    --------
    astropy.table.Table
        for details on parameters for creating an `GravitySpyTable`
    """

    # -- i/o ------------------------------------

    @classmethod
    def fetch(cls, table, *args, **kwargs):
        """Fetch data into an `EventTable`

        Parameters
        ----------
        table : `str`,
            The name of table you are attempting to receive triggers
            from.

        *args
            other filters you would like to supply
            underlying reader method for the given format

        .. note::

           For now it will attempt to autmatically connect you
           to a specific DB. In the future, this may be an input
           argument.

        Returns
        -------
        table : `EventTable`

        Raises
        ------
        astropy.io.registry.IORegistryError
            if the `format` cannot be automatically identified

        Notes"""

        try:
            engine = create_engine('postgresql://{0}:{1}@gravityspy.ciera.northwestern.edu:5432/gravityspy'.format(os.environ['QUEST_SQL_USER'], os.environ['QUEST_SQL_PASSWORD']))
        except:
            raise ValueError('Remember to set export QUEST_SQL_USER and export QUEST_SQL_PASSWORD in order to access the Gravity Spy Data: https://secrets.ligo.org/secrets/144/ description is username and secret is password.')

        try:
            tab = pd.read_sql(table, engine)
        except:
            raise ValueError('I am sorry could not retrive triggers from that table. The following our acceptible table names {0}'.format(engine.table_names()))

        tab = Table.from_pandas(tab)

        # and return
        return GravitySpyTable(EventTable(tab.filled()))


    def download(self, **kwargs):
        """If table contains Gravity Spy triggers `EventTable`

        Parameters
        ----------
        nproc : `int`, optional, default: 1
            number of CPUs to use for parallel file reading

        kwargs: Optional TrainingSet and LabelledSamples args
            that will download images in a specila way
            ./"Label"/"SampleType"/"image"

        Returns
        -------
        Folder containing omega scans sorted by label
        """

        # back to pandas
        imagesDB = self.to_pandas()
        imagesDB = imagesDB.loc[imagesDB.imgUrl1 != '?']

        TrainingSet = kwargs.pop('TrainingSet', 0)
        LabelledSamples = kwargs.pop('LabelledSamples', 0)

        if TrainingSet:
            for iLabel in imagesDB.Label.unique():
                if LabelledSamples:
                    for iType in imagesDB.SampleType.unique():
                        if not os.path.isdir(iLabel + '/' + iType):
                            os.makedirs(iLabel + '/' + iType)
                else:
                    if not os.path.isdir(iLabel):
                        os.makedirs(iLabel)

        def get_image(url, **kwargs):
            from ligo.org import request
            directory = './'
            if kwargs.pop('TrainingSet', 0):
                print('I worked 1')
                if kwargs.pop('LabelledSamples', 0):
                    print('I worked 2')
            with open(directory + url.split('/')[-1], 'wb') as f:
                f.write(request(url))

        imagesDB = imagesDB[['imgUrl1','imgUrl2','imgUrl3','imgUrl4']]
        images = imagesDB.as_matrix().flatten().tolist()

        # calculate maximum number of processes
        nproc = min(kwargs.pop('nproc', 1), len(images))

        # define multiprocessing method
        def _download_single_image(f, **kwargs):
            try:
                return f, get_image(f, **kwargs)
            except Exception as e:
                if nproc == 1:
                    raise
                elif isinstance(e, SAXException):  # SAXExceptions don't pickle
                    return f, e.getException()
                else:
                    return f, e

        # read files
        output = mp_utils.multiprocess_with_queues(
            nproc, _download_single_image, images, raise_exceptions=False)

        # raise exceptions (from multiprocessing, single process raises inline)
        for f, x in output:
            if isinstance(x, Exception):
                x.args = ('Failed to read %s: %s' % (f, str(x)),)
                raise x

