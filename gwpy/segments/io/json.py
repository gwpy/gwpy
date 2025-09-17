# Copyright (c) 2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Read/write segments and flags from DQSEGDB-format JSON."""

import json
from typing import IO

from ...io.registry import (
    default_registry,
    identify_factory,
)
from ...io.utils import (
    with_open,
)
from .. import DataQualityFlag

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- read ----------------------------

@with_open
def read_json_flag(fobj: IO) -> DataQualityFlag:
    """Read a `DataQualityFlag` from a segments-web.ligo.org JSON file."""
    data = json.load(fobj)

    # format flag
    name = "{ifo}:{name}:{version}".format(**data)
    out = DataQualityFlag(name, active=data["active"],
                          known=data["known"])

    # parse 'metadata'
    try:
        out.description = data["metadata"].get("flag_description", None)
    except KeyError:  # no metadata available, but that's ok
        pass
    else:
        out.isgood = not data["metadata"].get(
            "active_indicates_ifo_badness", False)

    return out


# -- write ---------------------------

@with_open(mode="w", pos=1)
def write_json_flag(
    flag: DataQualityFlag,
    fobj: IO,
    **kwargs,
):
    """Write a `DataQualityFlag` to a JSON file.

    Parameters
    ----------
    flag : `DataQualityFlag`
        Data to write.

    fobj : `file`
        Target file (or filename) to write.

    **kwargs
        other keyword arguments to pass to :func:`json.dump`

    See Also
    --------
    json.dump
        for details on acceptable keyword arguments
    """
    # build json packet
    data = {}
    data["ifo"] = flag.ifo
    data["name"] = flag.tag
    data["version"] = flag.version
    data["active"] = flag.active
    data["known"] = flag.known
    data["metadata"] = {}
    data["metadata"]["active_indicates_ifo_badness"] = not flag.isgood
    data["metadata"]["flag_description"] = flag.description

    # write
    json.dump(data, fobj, **kwargs)


# -- register ------------------------

default_registry.register_reader("json", DataQualityFlag, read_json_flag)
default_registry.register_writer("json", DataQualityFlag, write_json_flag)
default_registry.register_identifier(
    "json",
    DataQualityFlag,
    identify_factory("json"),
)
