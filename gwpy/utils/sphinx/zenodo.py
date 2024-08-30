# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2018-2023)
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

import argparse
import sys

import requests

DEFAULT_ZENODO_URL = "https://zenodo.org"
DEFAULT_HITS = 10


def format_citations(
    zid,
    url=DEFAULT_ZENODO_URL,
    hits=10,
    tag_prefix="v",
):
    """Query and format a citations page from Zenodo entries.

    Parameters
    ----------
    zid : `int`, `str`
        The Zenodo ID (``conceptrecid``) of the parent target record.

    url : `str`, optional
        The base URL of the Zenodo host, defaults to ``https://zenodo.org``.

    hits : `int`, optional
        The maximum number of results to show, default: ``10``.

    tag_prefix : `str`, optional
        The prefix for git tags. This is removed to generate the section
        headers in the output RST.

    Returns
    -------
    rst : `str`
        An RST-formatted string of DOI badges with URLs.
    """
    # query for metadata
    apiurl = f"{url.rstrip('/')}/api/records"
    params = {
        "q": f"conceptrecid:{zid}",
        "allversions": True,
        "sort": "version",
        "page": 1,
        "size": int(hits),
    }
    resp = requests.get(  # make the request
        apiurl,
        params=params,
        timeout=30,
    )
    resp.raise_for_status()  # make sure it worked
    records = resp.json()  # parse the response

    lines = []
    for i, rec in enumerate(records["hits"]["hits"]):
        # print RST-format header
        version = str(rec['metadata']['version'])[len(tag_prefix):]
        head = "-" * len(version)
        lines.extend([
            head,
            version,
            head,
            "",
        ])

        # add DOI badge
        badge = f"{url}/badge/doi/{rec['doi']}.svg"
        lines.extend([
            f".. image:: {badge}",
            f"    :alt: {rec['title']} Zenodo DOI badge",
            f"    :target: {rec['doi_url']}",
        ])

        # add break before next record
        lines.append("")

    return '\n'.join(lines).strip()


# -- command-line usage ---------------

def create_parser():
    """Create an `argparse.ArgumentParser` for this tool.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "id",
        type=int,
        help="Zenodo concept ID for package",
    )
    parser.add_argument(
        "-u",
        "--url",
        default=DEFAULT_ZENODO_URL,
        help="Base URL of API to query",
    )
    parser.add_argument(
        "-n",
        "--hits",
        default=DEFAULT_HITS,
        type=int,
        help="Number of versions to display",
    )
    parser.add_argument(
        "-p",
        "--tag-prefix",
        default="v",
        help="Prefix for version tags",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="stdout",
        help="Output file path",
    )
    return parser


def main(args=None):
    """Run this tool as a command-line script.
    """
    # parse arguments
    parser = create_parser()
    opts = parser.parse_args(args=args)

    # generate RST
    citing = format_citations(
        opts.id,
        url=opts.url,
        hits=opts.hits,
        tag_prefix=opts.tag_prefix,
    )

    # print
    if opts.output_file in {None, "stdout"}:
        f = sys.stdout
    else:
        f = open(opts.output_file, 'w')
    with f:
        print(citing, file=f)


if __name__ == '__main__':
    sys.exit(main())
