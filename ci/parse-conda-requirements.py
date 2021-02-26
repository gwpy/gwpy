#!/usr/bin/env python

"""Parse a requirements.txt-format file for use with Conda
"""

import argparse
import atexit
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from distutils.spawn import find_executable

import pkg_resources

# configure logging
LOGGING_FORMAT = "[%(asctime)s] %(levelname)+8s: %(message)s"
try:
    import coloredlogs
except ImportError:
    logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
else:
    coloredlogs.install(fmt=LOGGING_FORMAT, level=logging.INFO)

CONDA = find_executable("conda") or os.environ.get("CONDA_EXE", "conda")
CONDA_PACKAGE_MAP = {
    "matplotlib": "matplotlib-base",
}
LOGGER = logging.getLogger("condaparser")
VERSION_OPERATOR = re.compile('[><=!]')


def parse_requirements(file):
    for line in file:
        if line.startswith("-r "):
            name = line[3:].rstrip()
            with open(name, "r") as file2:
                yield from parse_requirements(file2)
        else:
            yield from pkg_resources.parse_requirements(line)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'filename',
        nargs='+',
        help='path of requirments file to parse',
        )
    parser.add_argument(
        '-p',
        '--python-version',
        default='{0.major}.{0.minor}'.format(sys.version_info),
        help='python version to use (default: %(default)s)',
        )
    parser.add_argument(
        '-o',
        '--output',
        help='path of output file, defaults to stdout',
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    requirements = ["python={0.python_version}.*".format(args)]
    for filename in args.filename:
        LOGGER.info("Processing {}".format(filename))
        with open(filename, "r") as reqf:
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
    LOGGER.info("Finding packages with conda:")
    cmd = [CONDA, 'install', '--quiet', '--dry-run', '--file', tmp, '--json']
    LOGGER.info("$ {}".format(" ".join(cmd)))
    pfind = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, err = pfind.communicate()

    # conda search failed, which means one or more packages are missing
    if pfind.returncode:
        LOGGER.warning(
            "conda install failed, attempting to filter out messages",
        )
        if isinstance(out, bytes):
            out = out.decode('utf-8')
        # attempt to parse the list of missing packages
        try:
            missing = [pkg.split('[', 1)[0].lower() for
                       pkg in json.loads(out)['packages']]
        except json.JSONDecodeError:
            # run it all again so that it fails out in the open
            LOGGER.critical("filtering failed...")
            cmd.remove("--json")
            LOGGER.info("$ {}".format(" ".join(cmd)))
            subprocess.check_call(cmd)
            raise
        # filter out the missing packages
        for req in list(requirements):
            if (
                req.replace(' ', '') in missing
                or VERSION_OPERATOR.split(req)[0].strip().lower() in missing
            ):
                LOGGER.warning("removing {!r}".format(req))
                requirements.remove(req)

    # print output to file or stdout
    if args.output:
        fout = open(args.output, 'w')
    else:
        LOGGER.info("Packages:")
        fout = sys.stdout
    try:
        for req in sorted(requirements):
            print(req, file=fout)
    finally:
        fout.close()
