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

import os
import sys

from .table import EventTable
from astropy.table import Table

import six

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
        try:
            import pandas as pd
        except ImportError as e:
            e.args = ('pandas is required to download triggers',)
            raise

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

        def get_image(url):
            from ligo.org import request
            directory = './' + url[1] + '/' + url[2] + '/'
            with open(directory + url[0].split('/')[-1], 'wb') as f:
                f.write(request(url[0]))

        imagesURL = imagesDB[['imgUrl1', 'imgUrl2', 'imgUrl3', 'imgUrl4']]
        imagesURL = imagesURL.as_matrix().flatten().tolist()
        if TrainingSet:
            labels = imagesDB.Label.as_matrix().flatten().tolist()
            if LabelledSamples:
                sampletype = imagesDB.SampleType.as_matrix().flatten().tolist()
                images = six.itertools.izip_longest(
                         imagesURL, labels, sampletype)
            else:
                images = six.itertools.izip_longest(imagesURL, labels, [])
        else:
            images = six.itertools.izip_longest(imagesURL, [], [])

        images = list(images)

        # calculate maximum number of processes
        nproc = min(kwargs.pop('nproc', 1), len(images))

        # define multiprocessing method
        def _download_single_image(f):
            try:
                return f, get_image(f)
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
