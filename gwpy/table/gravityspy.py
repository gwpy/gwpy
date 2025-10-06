# Copyright (c) 2017-2021 Scott Coughlin
#               2021 Cardiff University
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

"""Extend :mod:`astropy.table` with the `GravitySpyTable`."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import requests

from .table import EventTable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from astropy.table import Row

__author__ = "Scott Coughlin <scott.coughlin@ligo.org>"
__all__ = ["GravitySpyTable"]

logger = logging.getLogger(__name__)

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
    row: Row,
    outdir: Path,
    *,
    training_set: bool = False,
    labelled_samples: bool = False,
) -> str:
    """Construct where to download a GravitySpy image.

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
        / f"{ifo}_{id_}_spectrogram_{{}}.png",
    )


def _download_image(
    bundle: tuple[str, Path],
    timeout: float | None = 30.0,
    **kwargs,
) -> None:
    url, target = bundle
    logger.debug("Downloading %s to %s", url, target)
    resp = requests.get(
        url,
        timeout=timeout,
        **kwargs,
    )
    resp.raise_for_status()
    target.write_bytes(resp.content)
    logger.debug("Wrote %s", target)


class GravitySpyTable(EventTable):
    """A container for a table of Gravity Spy Events.

    As well as Events from the O1 Glitch Classification Paper which includes:

    - PCAT
    - PC-LIB
    - WDF
    - WDNN
    - Karoo GP

    See Also
    --------
    astropy.table.Table
        for details on parameters for creating a `GravitySpyTable`
    """

    # -- i/o -------------------------

    def download(
        self,
        download_path: str | Path = "download",
        parallel: int = 1,
        download_durs: Sequence[float] = DURATIONS,
        *,
        training_set: bool = False,
        labelled_samples: bool = False,
    ) -> list[str]:
        """Download image files associated with entries in a `GravitySpyTable`.

        Parameters
        ----------
        parallel : `int`, optional
            Number of parallel threads to use for parallel file reading.

        download_path : `str` optional
            Target directory for downloaded images.

        download_durs : `list` optional
            List of durations to download, must be a subset of
            ``[0.5, 1.0, 2.0, 4.0]``.
            Default is to download all the available GSpy durations.

        training_set : `bool`, optional
            If `True` download training set data.

        labelled_samples : `bool`, optional
            If `True` download only labelled samples.
        """
        download_path = Path(download_path)

        # work out which columns contain URLs
        url_columns = list(filter(URL_COLUMN.match, self.colnames))

        # construct a download target location for each requested image
        urls: dict[str, Path] = {}
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
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            executor.map(
                _download_image,
                list(urls.items()),
            )

        # and return the list of new files we have
        return sorted(map(str, urls.values()))

    @classmethod
    def search(
        cls,
        gravityspy_id: str,
        howmany: int = 10,
        era: str = "ALL",
        ifos: tuple[str, ...] | str = ("H1", "L1"),
        database: str = "similarity_index_o3",
        host: str = DEFAULT_HOST,
        timeout: float | None = 30.0,
        **kwargs,
    ) -> Self:
        """Perform a GravitySpy 'Similarity Search' for the given ID.

        Parameters
        ----------
        gravityspy_id : `str`
            The unique 10 character hash that identifies a Gravity Spy image.

        howmany : `int`, optional
            The number of similar images you would like.

        era : `str`, optional
            Which era to search, see online for more information.

        ifos : `tuple`, str`, optional
            The list of interferometers to include in the search.

        database : `str`, optional
            The database to query.

        host : `str`, optional
            The URL (scheme and FQDN) of the Gravity Spy host to query

        timeout : `float`, optional
            How long to wait for the server to respond before giving up.

        **kwargs
            Other kwargs are passed to `requests.get` to perform
            the actual remote communication.

        Returns
        -------
        table : `GravitySpyTable`
            A `GravitySpyTable` containing similar events based on
            an evaluation of the Euclidean distance of the input image
            to all other images in some Feature Space.

        Notes
        -----
        For an online version, and documentation of the search, see

        https://gravityspytools.ciera.northwestern.edu/search/
        """
        if isinstance(ifos, str):  # 'H1L1' -> ['H1', 'L1']
            ifo_list = [ifos[i:i+2] for i in range(0, len(ifos), 2)]
        else:
            ifo_list = list(ifos)

        query = urlencode({
            "howmany": howmany,
            "imageid": gravityspy_id,
            "era": "event_time BETWEEN {} AND {}".format(*ERA[era]),
            "ifo": ", ".join(map(repr, ifo_list)),
            "database": database,
        })
        url = host + SEARCH_PATH + SIMILARITY_SEARCH_PATH + "/?" + query

        logger.debug("Executing GravitySpy similarity search: %s", url)
        response = requests.get(url, timeout=timeout, **kwargs)
        logger.debug("Response: %s", response)
        response.raise_for_status()  # check the response
        try:
            return GravitySpyTable(response.json())
        except JSONDecodeError:  # that didn't work
            # if the response was actually HTML, something terrible happened
            if response.headers["Content-Type"] != JSON_CONTENT_TYPE:
                msg = (
                    f"response from {response.url} was '200 OK' but "
                    "the content is HTML, please check the request parameters"
                )
                raise requests.HTTPError(
                    msg,
                    response=response,
                ) from None
            raise
