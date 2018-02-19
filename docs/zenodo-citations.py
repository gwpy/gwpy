#!/usr/bin/env python

from __future__ import print_function

import argparse
import requests
import sys

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
args = parser.parse_args()

# query for metadata
url = ('{url}/api/records/?'
       'page=1&'
       'size={hits}&'
       'q=conceptrecid:"{id}"&'
       'sort=-version&'
       'all_versions=True'.format(**vars(args)))
metadata = requests.get(url).json()

weburl = url.replace('api/records/', 'search').replace('page=1', 'page=2')

if args.output_file:
    f = open(args.output_file, 'w')
else:
    f = sys.stdout


def _print(*args):
    return print(*args, file=f)


with f:
    _print("""
###########
Citing GWpy
###########

If you have used GWpy as part of a project that leads to a scientific
publication, please acknowledge this by citing the DOI for the version of
GWpy that you have used.

Each of the DOIs below resolves a Zenodo record for that version.
See the _Export_ section on each page for formatted citations in a number
of common styles.

The list below includes only the {hits} most recent releases of GWpy.
For older versions, please `click here <{weburl}>`__.
""".format(hits=args.hits, weburl=weburl).lstrip())

    for i, hit in enumerate(metadata['hits']['hits']):
        version = hit['metadata']['version'].strip(args.tag_prefix)
        _print('-' * len(version))
        _print(version)
        _print('-' * len(version))
        _print('')
        _print('.. image:: {badge}\n'
               '   :target: {doi}'.format(**hit['links']))
        if i < args.hits - 1:
            _print('')
