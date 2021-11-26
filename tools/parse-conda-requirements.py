#!/usr/bin/env python

"""Parse setup.cfg for package requirements and print out a list of
packages that can be installed using conda from the conda-forge channel.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from configparser import ConfigParser
from pathlib import Path
from shutil import which

import pkg_resources

# conda config
CONDA = (
    which("conda")
    or os.environ.get("CONDA_EXE", "conda")
)
CONDA_OR_MAMBA = which("mamba") or CONDA

# configure logging
LOGGER = logging.getLogger(__name__)
try:
    from coloredlogs import ColoredFormatter as _Formatter
except ImportError:
    _Formatter = logging.Formatter
if not LOGGER.hasHandlers():
    _LOG_HANDLER = logging.StreamHandler()
    _LOG_HANDLER.setFormatter(_Formatter(
        fmt="[%(asctime)s] %(levelname)+8s: %(message)s",
    ))
    LOGGER.addHandler(_LOG_HANDLER)

# regex to match version spec characters
VERSION_OPERATOR = re.compile('[><=!]')


def load_conda_forge_name_map():
    """Load the PyPI <-> conda-forge package name map from grayskull

    See https://github.com/conda-incubator/grayskull/blob/main/grayskull/pypi/config.yaml
    """  # noqa: E501
    try:
        from grayskull.pypi import PyPi
    except ModuleNotFoundError:
        LOGGER.warning(
            "failed to import grayskull, will not be able to translate "
            "pypi names into conda-forge names",
        )
        return {}

    from ruamel.yaml import YAML

    # parse the config file and return (pypi_name: conda_forge_name) pairs
    with open(PyPi.PYPI_CONFIG, "r") as conf:
        return {
            x: y["conda_forge"]
            for x, y in YAML().load(conf).items()
        }


def parse_requirements(lines, conda_forge_map=dict()):
    """Parse requirement specs from a list of lines

    Yields
    ------
    spec : `pkg_resources.Requirement`
        a formatted requirement for each line
    """
    for line in lines:
        for item in pkg_resources.parse_requirements(line):
            # if environment markers don't pass, skip
            if item.marker and not item.marker.evaluate():
                continue
            # if requirement is a URL, skip
            if item.url:
                continue
            # format as 'name{>=version}'
            yield (
                conda_forge_map.get(item.name, item.name)
                + str(item.specifier)
            ).strip()


def parse_all_requirements(python, file, extras):
    """Parse all requirements from a setup.cfg file

    Parameters
    ----------
    python : `str`
        the ``'X.Y'`` python version to use

    file : `str`, `pathlib.Path`
        the setuptools config file to read from

    extras : `list` of `str` or ``'ALL'``
        the list of extras to parse from the ``'options.extras_require'``
        key, or ``'ALL'`` to read all of them

    Yields
    ------
    requirements : `str`
        a requirement spec str compatible with conda
    """
    def _log(req):
        LOGGER.debug(f"  parsed {req}")
        return req

    # load the map from grayskull
    conda_forge_map = load_conda_forge_name_map()

    # add python first
    LOGGER.info(f"Using Python {python}")
    yield f"python={python}.*"

    # read requirements from setup.cfg
    conf = ConfigParser()
    conf.read(file)
    if extras == "ALL":  # use all extras
        extras = conf["options.extras_require"].keys()
    options = [
        ("options", "setup_requires"),
        ("options", "install_requires"),
    ] + [("options.extras_require", extra) for extra in extras]

    for sect, opt in options:
        LOGGER.info(f"Processing {sect}/{opt}")
        lines = conf[sect][opt].strip().splitlines()
        for req in parse_requirements(lines, conda_forge_map=conda_forge_map):
            LOGGER.debug(f"  parsed {req}")
            yield req


def find_packages(requirements, use_mamba=True):
    """Run conda/mamba to resolve an environment

    This does not actually create an environment, but is called so
    that if it fails because packages are missing, they can be
    identified.
    """
    prefix = tempfile.mktemp(prefix=Path(__file__).stem)
    EXE = CONDA_OR_MAMBA if use_mamba else CONDA
    use_mamba = "mamba" in EXE
    cmd = [
        EXE,
        "create",  # solve for a new environment
        "--dry-run",  # don't actually do anything but solve
        "--json",  # print JSON-format output
        "--quiet",  # don't print logging info
        "--yes",  # don't ask questions
        "--override-channels",  # ignore user's conda config
        "--channel", "conda-forge",  # only look at conda-forge
        "--prefix", prefix,  # don't overwrite existing env by mistake!
    ]

    # we use weird quoting here so that when the command is printed
    # to the log, PowerShell users can copy it and run it verbatim
    # without ps seeing '>' and piping output
    cmd.extend((f'"""{req}"""' for req in requirements))

    LOGGER.debug(f"$ {' '.join(cmd)}")
    pfind = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        universal_newlines=True,  # rename to 'text' for python >=3.7
    )

    # mamba search failed, but mamba doesn't output JSON on failure
    # so we have to try again using conda to strip out the failures
    if pfind.returncode and json and use_mamba:
        LOGGER.debug(f"mamba install failed:\n{pfind.stdout}".rstrip())
        LOGGER.debug("trying again with conda")
        return(find_packages(requirements, use_mamba=False))

    return pfind


def filter_requirements(requirements):
    """Filter requirements by running conda/mamba to see what is missing
    """
    requirements = list(requirements)

    # find all packages with conda
    LOGGER.info("Finding packages with conda/mamba")
    pfind = find_packages(requirements, use_mamba=True)

    if pfind.returncode:  # something went wrong
        # parse the JSON report
        report = json.loads(pfind.stdout)

        # report isn't a simple 'missing package' error
        if report["exception_name"] != "PackagesNotFoundError":
            LOGGER.critical("\n".join((
                "conda/mamba failed to resolve packages:",
                report['error'],
            )))
            pfind.check_returncode()  # raises exception

        # one or more packages are missing
        LOGGER.warning(
            "conda/mamba failed to find some packages, "
            "attempting to parse what's missing",
        )
        missing = {
            pkg.split('[', 1)[0].lower()  # strip out build info
            for pkg in report["packages"]
        }

        # filter out the missing packages
        for req in list(requirements):
            guesses = {
                # name with version (no whitespace)
                req.replace(' ', ''),
                # name only
                VERSION_OPERATOR.split(req)[0].strip().lower(),
            }
            if guesses & missing:  # package is missing
                LOGGER.warning(f"  removing {req!r}")
                requirements.remove(req)

    return requirements


def create_parser():
    """Create a command-line `ArgumentParser` for this tool
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'extras_name',
        nargs='*',
        default=[],
        help='name of setuptools \'extras\' to parse',
    )
    parser.add_argument(
        '-a',
        '--all',
        action="store_true",
        default=False,
        help='include all extras',
    )
    parser.add_argument(
        '-c',
        '--config-file',
        default='setup.cfg',
        help='path of setup.cfg file',
    )
    parser.add_argument(
        '-p',
        '--python-version',
        default='{0.major}.{0.minor}'.format(sys.version_info),
        help='python version to use',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        help='path of output file, defaults to stdout',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help='print verbose logging',
    )
    return parser


def main(args=None):
    """Run the thing
    """
    parser = create_parser()
    args = parser.parse_args(args=args)

    # set verbose logging
    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    LOGGER.debug(f"found conda in {CONDA}")
    if CONDA_OR_MAMBA != CONDA:
        LOGGER.debug(f"found mamba in {CONDA_OR_MAMBA}")

    # parse requirements
    requirements = parse_all_requirements(
        args.python_version,
        args.config_file,
        "ALL" if args.all else args.extras_name,
    )

    # filter out requirements that aren't available in conda-forge
    requirements = filter_requirements(requirements)

    LOGGER.info("Package finding complete")

    # print output to file or stdout
    out = "\n".join(sorted(requirements))
    if args.output:
        args.output.write_text(out)
    else:
        print(out)


if __name__ == "__main__":
    sys.exit(main())
