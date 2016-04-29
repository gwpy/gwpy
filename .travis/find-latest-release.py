#!/usr/bin/env python

"""Find the latest release of a package by parsing an apache directory listing.

This code requires the release numbers to use semantic versioning.
"""

import argparse
import re
import os.path
from distutils.version import LooseVersion

from six.moves.urllib.request import urlopen

from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('url', help='URL of package host')
parser.add_argument('package', help='name of package to match, used to build '
                                    'regex as \'<package>-*.tar.gz\'')
parser.add_argument('-x', '--extension', default='tar.gz',
                    help='extension of package files hosted at URL, '
                         'default: %(default)')
parser.add_argument('-o', '--output-format', default='url',
                    choices=['url', 'version', 'filename'],
                    help='output format')
args = parser.parse_args()

listing = BeautifulSoup(urlopen(args.url), 'html.parser')
match = re.compile('%s-(?P<version>(.*)).%s' % (args.package, args.extension))

latest = LooseVersion('0.0.0')

for a in listing.find_all('a', text=match):
    filename = a.attrs['href']
    version = LooseVersion(match.search(filename).groups()[0])
    if version > latest:
        latest = version
        target = os.path.join(args.url, filename)

if args.output_format == 'url':
    print(target)
elif args.output_format == 'filename':
    print(os.path.basename(target))
else:
    print(latest)
