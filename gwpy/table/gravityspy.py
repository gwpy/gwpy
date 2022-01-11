# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017-2021)
#               Cardiff University (2021)
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

import re
from json import JSONDecodeError
from pathlib import Path
from urllib.parse import urlencode

import requests

from ..utils import mp as mp_utils
from .table import EventTable

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'
__all__ = ['GravitySpyTable']

JSON_CONTENT_TYPE = "application/json"

# default gravity spy remote host
DEFAULT_HOST = "https://gravityspytools.ciera.northwestern.edu"

# API paths
SEARCH_PATH = "/search"
SIMILARITY_SEARCH_PATH = "/similarity_search_restful_API"

# ordered list of image durations supported by gravityspy.
# NOTE: the index of each element here is the map between
#       the 'urlX' column in the GravitySpyTable and the
#       duration of the relevant image, for some reason.
DURATIONS = (
    0.5,
    1.0,
    2.0,
    4.0,
)

# list of ERAs supported by similarity search API
ERA = {
    "ALL": (1126400000, 1584057618),
    "O1": (1126400000, 1137250000),
    "ER10": (1161907217, 1164499217),
    "O2a": (1164499217, 1219276818),
    "ER13": (1228838418, 1229176818),
    "ER14": (1235750418, 1238112018),
    "O3": (1238166018, 1272326418),
    "O3a": (1238166018, 1254009618),
    "O3b": (1256655642, 1272326418),
}

# regex to filter columns containing URLs
URL_COLUMN = re.compile(r"\Aurl[0-9]+\Z")


def _image_download_target(
    row,
    outdir,
    training_set=False,
    labelled_samples=False,
):
    """Construct where to download a GravitySpy image

    This returns a `str.format`-style template that just needs
    the duration to form a complete target path.
    """
    ifo = row["ifo"]
    id_ = row["gravityspy_id"]
    label = row["ml_label"] if training_set else ""
    stype = row["sample_type"] if labelled_samples else ""
    return str(
        outdir
        / label
        / stype
        / "{}_{}_spectrogram_{{}}.png".format(ifo, id_)
    )


def _download_image(bundle):
    url, target = bundle
    return target.write_bytes(requests.get(url).content)


class GravitySpyTable(EventTable):
    """A container for a table of Gravity Spy Events (as well as
    Events from the O1 Glitch Classification Paper whcih includes

    - PCAT
    - PC-LIB
    - WDF
    - WDNN
    - Karoo GP

    See also
    --------
    astropy.table.Table
        for details on parameters for creating a `GravitySpyTable`
    """

    # -- i/o ------------------------------------

    def download(
        self,
        download_path="download",
        nproc=1,
        download_durs=DURATIONS,
        training_set=False,
        labelled_samples=False,
    ):
        """Download image files associated with entries in a `GravitySpyTable`.

        Parameters
        ----------
        nproc : `int`, optional, default: 1
            number of CPUs to use for parallel file reading

        download_path : `str` optional, default: 'download'
            Specify where the images end up.

        download_durs : `list` optional, default: [0.5, 1.0, 2.0, 4.0]
            Specify exactly which durations you want to download
            default is to download all the avaialble GSpy durations.

        training_set : `bool`, optional
            if `True` download training set data

        labelled_samples : `bool`, optional
            if `True` download only labelled samples
        """
        download_path = Path(download_path)

        # work out which columns contain URLs
        url_columns = list(filter(URL_COLUMN.match, self.colnames))

        # construct a download target location for each requested image
        urls = dict()
        for row in self:
            if row["url1"] in {None, "", "?"}:  # skip rows without URLs
                continue

            # construct the abstract download target
            _download_target = _image_download_target(
                row,
                download_path,
                training_set=training_set,
                labelled_samples=labelled_samples,
            )

            # and map the remote URL to a concrete target
            # for each duration requested
            for col in url_columns:
                idx = int(col[3:])  # urlXXXX number
                duration = DURATIONS[idx-1]
                if duration in download_durs:  # image requested
                    urls[row[col]] = Path(_download_target.format(duration))

        # create all of the directories needed up front
        for path in urls.values():
            path.parent.mkdir(exist_ok=True, parents=True)

        # download the images
        mp_utils.multiprocess_with_queues(
            nproc,
            _download_image,
            urls.items(),
        )

        # and return the list of new files we have
        return sorted(map(str, urls.values()))

    @classmethod
    def search(
        cls,
        gravityspy_id,
        howmany=10,
        era="ALL",
        ifos=("H1", "L1"),
        database="similarity_index_o3",
        host=DEFAULT_HOST,
        **kwargs,
    ):
        """Perform a GravitySpy 'Similarity Search' for the given ID.

        Parameters
        ----------
        gravityspy_id : `str`
            the unique 10 character hash that identifies a Gravity Spy image

        howmany : `int`, optional
            number of similar images you would like

        era : `str`, optional
            which era to search, see online for more information

        ifos : `tuple`, str`, optional
            the list of interferometers to include in the search

        database : `str`, optional
            the database to query

        host : `str`, optional
            the URL (scheme and FQDN) of the Gravity Spy host to query

        **kwargs
            all other kwargs are passed to `requests.get` to perform
            the actual remote communication.

        Returns
        -------
        table : `GravitySpyTable`
            a `GravitySpyTable` containing similar events based on
            an evaluation of the Euclidean distance of the input image
            to all other images in some Feature Space

        Notes
        -----
        For an online version, and documentation of the search, see

        https://gravityspytools.ciera.northwestern.edu/search/
        """
        base = host + SEARCH_PATH + SIMILARITY_SEARCH_PATH

        if isinstance(ifos, str):  # 'H1L1' -> ['H1', 'L1']
            ifos = [ifos[i:i+2] for i in range(0, len(ifos), 2)]

        query = urlencode({
            'howmany': howmany,
            'imageid': gravityspy_id,
            'era': "event_time BETWEEN {} AND {}".format(*ERA[era]),
            'ifo': ", ".join(map(repr, ifos)),
            'database': database,
        })
        url = '{}/?{}'.format(base, query)

        response = requests.get(url, **kwargs)
        response.raise_for_status()  # check the response
        try:
            return GravitySpyTable(response.json())
        except JSONDecodeError:  # that didn't work
            # if the response was actually HTML, something terrible happened
            if response.headers["Content-Type"] != JSON_CONTENT_TYPE:
                raise requests.HTTPError(
                    "response from {} was '200 OK' but the content is HTML, "
                    "please check the request parameters".format(
                        response.url,
                    ),
                    response=response,
                )
            raise
