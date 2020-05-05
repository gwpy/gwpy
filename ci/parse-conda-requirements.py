#!/usr/bin/env python

"""Parse a requirements.txt-format file for use with Conda
"""

import argparse
import atexit
import json
import os
import re
import subprocess
import sys
import tempfile
from distutils.spawn import find_executable

import pkg_resources

CONDA_PACKAGE_MAP = {
    "matplotlib": "matplotlib-base",
}


def parse_requirements(file):
    for line in file:
        if line.startswith("-r "):
            name = line[3:].rstrip()
            with open(name, "r") as file2:
                yield from parse_requirements(file2)
        else:
            yield from pkg_resources.parse_requirements(line)


CONDA = find_executable("conda") or os.environ.get("CONDA_EXE", "conda")

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
with open(args.filename, "r") as reqf:
    for item in parse_requirements(reqf):
        # if environment markers don't pass, skip
        if item.marker and not item.marker.evaluate():
            continue
        # if requirement is a URL, skip
        if item.url:
            continue
        name = CONDA_PACKAGE_MAP.get(item.name, item.name)
        requirements.append('{}{}'.format(name, item.specifier))

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
cmd = [CONDA, 'install', '--quiet', '--dry-run', '--file', tmp, '--json']
pfind = subprocess.Popen(cmd, stdout=subprocess.PIPE)
out, err = pfind.communicate()

# conda search failed, which means one or more packages are missing
if pfind.returncode:
    if isinstance(out, bytes):
        out = out.decode('utf-8')
    try:
        missing = [pkg.split('[', 1)[0].lower() for
                   pkg in json.loads(out)['packages']]
    except json.JSONDecodeError:
        # run it all again so that it fails out in the open
        subprocess.check_call(cmd)
        raise
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
