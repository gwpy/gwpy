# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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
import requests
import sys


def parse_command_line():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('id', type=int, help='Zenodo ID for package')
    parser.add_argument('-u', '--url', default='https://zenodo.org/',
                        help='%(metavar)s to query')
    parser.add_argument('-n', '--hits', default=10, type=int,
                        help='number of versions to display')
    parser.add_argument('-p', '--tag-prefix', default='v',
                        help='prefix for git version tags')
    parser.add_argument('-o', '--output-file', help='output file path')

    if len(sys.argv) == 1:  # print --help message for no arguments
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


def format_citations(zid, url='https://zenodo.org/', hits=10, tag_prefix='v'):
    """Query and format a citations page from Zenodo entries

    Parameters
    ----------
    zid : `int`, `str`
        the Zenodo ID of the target record

    url : `str`, optional
        the base URL of the Zenodo host, defaults to ``https://zenodo.org``

    hist : `int`, optional
        the maximum number of hits to show, default: ``10``

    tag_prefix : `str`, optional
        the prefix for git tags. This is removed to generate the section
        headers in the output RST

    Returns
    -------
    rst : `str`
        an RST-formatted string of DOI badges with URLs
    """
    # query for metadata
    url = ('{url}/api/records/?'
           'page=1&'
           'size={hits}&'
           'q=conceptrecid:"{id}"&'
           'sort=-version&'
           'all_versions=True'.format(id=zid, url=url, hits=hits))
    resp = requests.get(url)  # make the request
    resp.raise_for_status()  # make sure it worked
    metadata = resp.json()  # parse the response

    lines = []
    for i, hit in enumerate(metadata['hits']['hits']):
        version = hit['metadata']['version'][len(tag_prefix):]
        lines.append('-' * len(version))
        lines.append(version)
        lines.append('-' * len(version))
        lines.append('')
        lines.append('.. image:: {badge}\n'
                     '   :target: {doi}'.format(**hit['links']))
        if i < hits - 1:
            lines.append('')

    return '\n'.join(lines)


if __name__ == '__main__':
    args = parse_command_line()

    if args.output_file:
        f = open(args.output_file, 'w')
    else:
        f = sys.stdout

    citing = format_citations(args.id, url=args.url, hits=args.hits,
                              tag_prefix=args.tag_prefix)

    with f:
        print(citing, file=f)
