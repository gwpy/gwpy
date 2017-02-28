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
parser.add_argument('-x', '--extension', default=['tar.gz', 'tar.xz'],
                    action='append',
                    help='extension(s) of package files hosted at URL, '
                         'default: %(default)s')
args = parser.parse_args()

listing = BeautifulSoup(urlopen(args.url), 'html.parser')
extensions = '(%s)' % '|'.join(set(args.extension))
match = re.compile('%s-(?P<version>([0-9]+.[0-9]+(.*))).%s\Z'
                   % (args.package, extensions))

latest = LooseVersion('0.0.0')

for a in listing.find_all('a', text=match):
    filename = a.attrs['href']
    version = LooseVersion(match.search(filename).groups()[0])
    if version > latest:
        latest = version
        target = os.path.join(args.url, filename)

print('%s %s' % (latest, target))
