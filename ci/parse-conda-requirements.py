#!/usr/bin/env python

"""Parse a requirements.txt-format file for use with Conda
"""

from __future__ import print_function

import argparse
import atexit
import json
import os
import re
import subprocess
import sys
import tempfile

try:
    from pip._internal import req as pip_req
except ImportError:  # pip < 10.0.0
    from pip import req as pip_req

VERSION_OPERATOR = re.compile('[><=!]')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('filename', help='path of requirments file to parse')
parser.add_argument('-p', '--python-version',
                    default='{0.major}.{0.minor}'.format(sys.version_info),
                    help='python version to use (default: %(default)s)')
parser.add_argument('-o', '--output', help='path of output file, '
                                           'defaults to stdout')
args = parser.parse_args()

requirements = ["python={0.python_version}.*".format(args)]
for item in pip_req.parse_requirements(args.filename, session='gwpyci'):
    # if environment markers don't pass, skip
    if item.markers and not item.markers.evaluate():
        continue
    # if requirement is a URL, skip
    if item.original_link:
        continue
    requirements.append('{0.name}{0.req.specifier}'.format(item))

tmp = tempfile.mktemp()

# clean up when we're done
def _clean():
    if os.path.isfile(tmp):
        os.remove(tmp)
atexit.register(_clean)

# print requirements to temp file
with open(tmp, 'w') as reqfile:
    for req in requirements:
        print(req, file=reqfile)

# find all packages with conda
pfind = subprocess.Popen(
    ['conda', 'install', '--quiet', '--dry-run', '--file', tmp, '--json'],
    stdout=subprocess.PIPE)
out, err = pfind.communicate()

# conda search failed, which means one or more packages are missing
if pfind.returncode:
    if isinstance(out, bytes):
        out = out.decode('utf-8')
    missing = [pkg.split('[', 1)[0].lower() for
               pkg in json.loads(out)['packages']]
    requirements = [
        req for req in requirements if
        VERSION_OPERATOR.split(req)[0].strip().lower() not in missing]

# print output to file or stdout
if args.output:
    fout = open(args.output, 'w')
else:
    fout = sys.stdout
try:
    for req in requirements:
        print(req, file=fout)
finally:
    fout.close()
