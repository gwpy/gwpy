# Copyright (c) 2014-2017 Louisiana State University
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

"""The `DataQualityFlag` and `DataQualityDict`.

The `DataQualityFlag` represents an annotated set of data-quality segments
indicating something about the state of a laser-interferometer
gravitational-wave detector in a given time interval.

The `DataQualityDict` is just a `dict` of flags, provided as a convenience
for handling multiple flags over the same global time interval.
"""

from __future__ import annotations

import contextlib
import json
import operator
import os
import re
import textwrap
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import (
    copy as shallowcopy,
    deepcopy,
)
from functools import (
    reduce,
    wraps,
)
from io import BytesIO
from math import (
    ceil,
    floor,
)
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from astropy.table import Row as AstropyTableRow
from dqsegdb2.query import query_segments
from dqsegdb2.utils import get_default_host
from gwosc import timeline
from numpy import inf
from requests import codes as http_codes
from requests.exceptions import (
    HTTPError,
    RequestException,
)

from ..io.registry import UnifiedReadWriteMethod
from ..time import (
    LIGOTimeGPS,
    to_gps,
)
from ..utils.misc import if_not_none
from .connect import (
    DataQualityDictRead,
    DataQualityDictWrite,
    DataQualityFlagRead,
    DataQualityFlagWrite,
)
from .segments import (
    Segment,
    SegmentList,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )
    from typing import (
        ParamSpec,
        Self,
        SupportsFloat,
        TypeVar,
    )

    import astropy.table
    import igwn_ligolw

    from ...plot import Plot
    from ...time import SupportsToGps

    P = ParamSpec("P")
    R = TypeVar("R")

    SegmentListLike = Iterable[tuple[SupportsFloat, SupportsFloat]]

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = [
    "DataQualityDict",
    "DataQualityFlag",
]

DEFAULT_SEGMENT_SERVER = get_default_host()

IFO_TAG_VERSION_REGEX = re.compile(
    r"\A(?P<ifo>[A-Z]\d):(?P<tag>[^/]+):(?P<version>\d+)\Z")
IFO_TAG_REGEX = re.compile(r"\A(?P<ifo>[A-Z]\d):(?P<tag>[^/]+)\Z")
TAG_VERSION_REGEX = re.compile(r"\A(?P<tag>[^/]+):(?P<version>\d+)\Z")


# -- utilities -----------------------

def _parse_query_segments(
    args: tuple[
        SegmentList | tuple[SupportsToGps, SupportsToGps]
    ] | tuple[SupportsToGps, SupportsToGps],
    func: Callable,
) -> SegmentList:
    """Parse ``args`` for query_dqsegdb() or query_segdb().

    Returns a SegmentList in all cases
    """
    # user passed SegmentList
    if len(args) == 1 and isinstance(args[0], SegmentList):
        return args[0]

    # otherwise unpack two arguments as a segment
    if len(args) == 1:
        args = args[0]

    # if not two arguments, panic
    try:
        start, end = args
    except ValueError as exc:
        exc.args = (f"{func.__name__}() takes 2 arguments for start and end GPS time, "
                    "or 1 argument containing a Segment or SegmentList",)
        raise

    # return list with one Segment
    return SegmentList([Segment(to_gps(start), to_gps(end))])


def _check_on_error(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        on_error = kwargs.get("on_error", "raise")
        # check on_error flag
        if on_error not in {"raise", "warn", "ignore"}:
            msg = (
                "on_error must be one of 'raise', 'warn', or 'ignore', "
                f"not '{on_error}'"
            )
            raise ValueError(msg)
        return func(*args, **kwargs)

    return wrapper



# -- DataQualityFlag -----------------

class DataQualityFlag:
    """A representation of a named set of segments.

    Parameters
    ----------
    name : str, optional
        The name of this flag.
        This should be of the form {ifo}:{tag}:{version}, e.g.
        'H1:DMT-SCIENCE:1'. Use `label` for human-readable names.

    active : `SegmentList`, optional
        A list of active segments for this flag

    known : `SegmentList`, optional
        A list of known segments for this flag

    label : `str`, optional
        Human-readable name for this flag, e.g. ``'Science-mode'``

    category : `int`, optional
        Veto category for this flag.

    description : `str`, optional
        Human-readable description of why this flag was created.

    isgood : `bool`, optional
        Do active segments mean the IFO was in a good state?
    """

    _EntryClass: type[Segment] = Segment
    _ListClass: type[SegmentList] = SegmentList

    def __init__(
        self,
        name: str | None = None,
        active: SegmentListLike | None = None,
        known: SegmentListLike | None = None,
        *,
        label: str | None = None,
        category: int | None = None,
        description: str | None = None,
        isgood: bool = True,
        padding: tuple[float | None, float | None] | None = None,
    ) -> None:
        """Define a new DataQualityFlag."""
        self.name = name
        self.known = known
        self.active = active
        self.label = label
        self.category = category
        self.description = description
        self.isgood = isgood
        self.padding = padding

    # -- utilities -------------------

    def _to_segmentlist(
        self,
        segmentlist: SegmentListLike | None,
    ) -> SegmentList:
        """Convert input to SegmentList of the correct type."""
        if segmentlist is None:
            return self._ListClass()
        return self._ListClass(self._EntryClass(*seg) for seg in segmentlist)

    # -- properties ------------------

    @property
    def name(self) -> str | None:
        """The name associated with this flag.

        This normally takes the form {ifo}:{tag}:{version}. If found,
        each component is stored separately the associated attributes.

        :type: `str`
        """
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        self._name = name
        try:
            self._parse_name(name)
        except ValueError:
            self._parse_name(None)

    @property
    def ifo(self) -> str | None:
        """The interferometer associated with this flag.

        This should be a single uppercase letter and a single number,
        e.g. ``'H1'``.
        """
        return self._ifo

    @ifo.setter
    def ifo(self, ifoname: str | None) -> None:
        self._ifo = ifoname

    @property
    def tag(self) -> str | None:
        """The tag (name) associated with this flag.

        This should take the form ``'AAA-BBB_CCC_DDD'``, i.e. where
        each component is an uppercase acronym of alphanumeric
        characters only, e.g. ``'DCH-IMC_BAD_CALIBRATION'`` or
        ``'DMT-SCIENCE'``.
        """
        return self._tag

    @tag.setter
    def tag(self, tag: str | None) -> None:
        self._tag = tag

    @property
    def version(self) -> int | None:
        """The version number of this flag.

        Each flag in the segment database is stored with a version
        integer, with each successive version representing a more
        accurate dataset for its known segments than any previous.
        """
        return self._version

    @version.setter
    def version(self, v: int | None) -> None:
        self._version = if_not_none(int, v)

    @property
    def label(self) -> str | None:
        """A human-readable label for this flag.

        For example: ``'Science-mode'``.
        """
        return self._label

    @label.setter
    def label(self, lab: str | None) -> None:
        self._label = lab

    @property
    def active(self) -> SegmentList:
        """The set of segments during which this flag was active."""
        return self._active

    @active.setter
    def active(self, segmentlist: SegmentListLike | None) -> None:
        self._active = self._to_segmentlist(segmentlist)

    @active.deleter
    def active(self) -> None:
        self._active = self._ListClass()

    @property
    def known(self) -> SegmentList:
        """The segments during which this flag was known."""
        return self._known

    @known.setter
    def known(self, segmentlist: SegmentListLike | None) -> None:
        self._known = self._to_segmentlist(segmentlist)

    @known.deleter
    def known(self) -> None:
        self._known = self._ListClass()

    @property
    def category(self) -> int | None:
        """Veto category for this flag."""
        return self._category

    @category.setter
    def category(self, cat: int | None) -> None:
        self._category = if_not_none(int, cat)

    @property
    def description(self) -> str | None:
        """Description of why/how this flag was generated."""
        return self._description

    @description.setter
    def description(self, desc: str | None) -> None:
        self._description = desc

    @property
    def isgood(self) -> bool:
        """Whether `active` segments mean the instrument was in a good state."""
        return self._isgood

    @isgood.setter
    def isgood(self, good: bool) -> None:
        self._isgood = bool(good)

    @property
    def padding(self) -> tuple[float, float]:
        """(start, end) padding for this flag's active segments."""
        return self._padding

    @padding.setter
    def padding(self, pad: None | tuple[float | None, float | None]) -> None:
        if pad is None:
            self._padding = (0, 0)
        else:
            self._padding = (float(pad[0] or 0.), float(pad[1] or 0.))

    @padding.deleter
    def padding(self) -> None:
        self._padding = (0., 0.)

    # -- read-only properties --------

    @property
    def texname(self) -> str | None:
        """Name of this flag in LaTeX printable format."""
        if self.name is None:
            return None
        return self.name.replace("_", r"\_")

    @property
    def extent(self) -> Segment:
        """The GPS ``[start, stop)`` enclosing segment of this `DataQualityFlag`."""
        return self.known.extent()

    @property
    def livetime(self) -> float:
        """Amount of time this flag was `active`."""
        return abs(self.active)

    @property
    def regular(self) -> bool:
        """`True` if the `active` segments are a proper subset of the `known`."""
        return abs(self.active - self.known) == 0

    # -- i/o -------------------------

    read = UnifiedReadWriteMethod(DataQualityFlagRead)
    write = UnifiedReadWriteMethod(DataQualityFlagWrite)

    # -- classmethods ----------------

    @classmethod
    def query_dqsegdb(
        cls,
        flag: str,
        *args: SupportsToGps | Segment | SegmentList,
        host: str | None = DEFAULT_SEGMENT_SERVER,
        **kwargs,
    ) -> Self:
        """Query a DQSegDB server for a flag.

        Parameters
        ----------
        flag : `str`
            The name of the flag for which to query

        args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a `SegmentList`
            defining a number of summary segments

        host : `str`, optional
            Name or URL of the DQSegDB instance to talk to.
            Defaults to :func:`dqsegdb2.utils.get_default_host`.

        kwargs
            All other keyword arguments are passed to
            :func:`dqsegdb2.query.query_segments`.

        Returns
        -------
        flag : `DataQualityFlag`
            A new `DataQualityFlag`, with the `known` and `active` lists
            filled appropriately.

        Examples
        --------
        The GPS interval(s) of interest can be passed as two arguments
        specifing the start and end of a single interval:

        >>> DataQualityDict.query_dqsegdb(["X1:OBSERVING:1", "Y1:OBSERVING:1"], start, end)

        Or, as a single `Segment`:

        >>> DataQualityDict.query_dqsegdb(["X1:OBSERVING:1", "Y1:OBSERVING:1"], interval)

        Or, as a `SegmentList` specifying multiple intervals.

        >>> DataQualityDict.query_dqsegdb(["X1:OBSERVING:1", "Y1:OBSERVING:1"], intervals)
        """  # noqa: E501
        # parse arguments
        qsegs = _parse_query_segments(args, cls.query_dqsegdb)

        # parse deprecated 'url' keyword as 'host'
        url = kwargs.pop("url", None)
        if url:
            warnings.warn(
                "the `url` keyword argument for `query_dqsegdb` "
                "has been renamed `host`; this warning will become "
                "an error in the future",
                DeprecationWarning,
                stacklevel=2,
            )
            host = url

        # parse flag
        out = cls(name=flag)
        if out.ifo is None or out.tag is None:
            msg = f"Cannot parse ifo or tag (name) for flag '{flag}'"
            raise ValueError(msg)

        # process query
        for start, end in qsegs:
            # handle infinities
            if float(end) == +inf:
                end = int(to_gps("now"))  # noqa: PLW2901

            # query
            try:
                data = query_segments(
                    flag,
                    int(start),
                    int(end),
                    host=host,
                    **kwargs,
                )
            except HTTPError as exc:
                if exc.response.status_code == http_codes.NOT_FOUND:
                    exc.args = (exc.args[0] + f" [{flag}]",)
                raise

            # read from json buffer
            new = cls.read(
                BytesIO(json.dumps(data).encode("utf-8")),
                format="json",
            )

            # restrict to query segments
            segl = SegmentList([Segment(start, end)])
            new.known &= segl
            new.active &= segl
            out += new
            # replace metadata
            out.description = new.description
            out.isgood = new.isgood

        return out

    # alias for compatibility
    query = query_dqsegdb

    @classmethod
    def fetch_open_data(
        cls,
        flag: str,
        start: SupportsToGps,
        end: SupportsToGps,
        **kwargs,
    ) -> Self:
        """Fetch Open Data timeline segments into a flag.

        flag : `str`
            The name of the flag to query

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS end time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        timeout : `int`, optional
            Timeout for download (seconds).

        host : `str`, optional
            URL of GWOSC host, default: ``'https://gwosc.org'``.

        Returns
        -------
        flag : `DataQualityFlag`
            a new flag with `active` segments filled from Open Data

        Examples
        --------
        >>> from gwpy.segments import DataQualityFlag
        >>> print(DataQualityFlag.fetch_open_data(
        ...     "H1_DATA",
        ...     "Sep 14 2015",
        ...     "Sep 15 2015",
        ... ))
        <DataQualityFlag('H1:DATA',
                         known=[[1126224017 ... 1126310417)]
                         active=[[1126251604 ... 1126252133)
                                 [1126252291 ... 1126274322)
                                 [1126276234 ... 1126281754)
                               ...
                                 [1126308670 ... 1126309577)
                                 [1126309637 ... 1126309817)
                                 [1126309877 ... 1126310417)]
                         description=None)>
        """
        start = int(to_gps(start))
        end = ceil(to_gps(end))
        known = [(start, end)]
        active = timeline.get_segments(flag, start, end, **kwargs)
        return cls(
            flag.replace("_", ":", 1),
            known=known,
            active=active,
            label=flag,
        )

    @classmethod
    def from_veto_def(
        cls,
        veto: igwn_ligolw.lsctables.VetoDef | astropy.table.Row,
    ) -> Self:
        """Define a `DataQualityFlag` from a `VetoDef`.

        Parameters
        ----------
        veto : `~igwn_ligolw.lsctables.VetoDef`, `~astropy.table.Row`.
            Veto definition to convert from.
        """
        # handle getting by item name (astropy) or attribute name (igwn_ligolw)
        if isinstance(veto, AstropyTableRow):
            get = veto.get
        else:
            def get(name, default=None):  # noqa: ANN001,ANN202
                return getattr(veto, name, default)

        name = f"{get('ifo')}:{get('name')}"
        with contextlib.suppress(TypeError):
            name += f":{int(get('version'))}"
        known = Segment(get("start_time"), get("end_time") or +inf)
        pad = (get("start_pad") or 0, get("end_pad") or 0)
        return cls(
            name=name,
            known=[known],
            category=get("category", None),
            description=get("comment", None),
            padding=pad,
        )

    # -- methods ---------------------

    def populate(
        self,
        source: str | None = DEFAULT_SEGMENT_SERVER,
        segments: SegmentList | None = None,
        *,
        pad: bool = True,
        **kwargs,
    ) -> Self:
        """Query the segment database for this flag's active segments.

        This method assumes all of the metadata for each flag have been
        filled. Minimally, the following attributes must be filled

        .. autosummary::

           ~DataQualityFlag.name
           ~DataQualityFlag.known

        Segments will be fetched from the database, with any
        :attr:`~DataQualityFlag.padding` added on-the-fly.

        This `DataQualityFlag` will be modified in-place.

        Parameters
        ----------
        source : `str`
            Source of segments for this flag. This must be
            either a URL for a segment database or a path to a file on disk.

        segments : `SegmentList`, optional
            A list of segments during which to query, if not given,
            existing known segments for this flag will be used.

        pad : `bool`, optional, default: `True`
            apply the `~DataQualityFlag.padding` associated with this
            flag, default: `True`.

        **kwargs
            any other keyword arguments to be passed to
            :meth:`DataQualityFlag.query` or :meth:`DataQualityFlag.read`.

        Returns
        -------
        self : `DataQualityFlag`
            a reference to this flag
        """
        tmp = DataQualityDict()
        tmp[self.name] = self
        tmp.populate(source=source, segments=segments, pad=pad, **kwargs)
        return tmp[self.name]

    def contract(self, x: float) -> SegmentList:
        """Contract each of the ``active`` segments by ``x`` seconds.

        This method adds ``x`` to each segment's lower bound, and subtracts
        ``x`` from the upper bound.

        The :attr:`~DataQualityFlag.active` `SegmentList` is modified
        in place.

        Parameters
        ----------
        x : `float`
            Number of seconds by which to contract each `Segment`.
        """
        self.active = self.active.contract(x)
        return self.active

    def protract(self, x: float) -> SegmentList:
        """Protract each of the ``active`` segments by ``x`` seconds.

        This method subtracts ``x`` from each segment's lower bound,
        and adds ``x`` to the upper bound, while maintaining that each
        `Segment` stays within the `known` bounds.

        The :attr:`~DataQualityFlag.active` `SegmentList` is modified
        in place.

        Parameters
        ----------
        x : `float`
            Number of seconds by which to protact each `Segment`.
        """
        self.active = self.active.protract(x)
        return self.active

    def pad(
        self,
        *args: float,
        inplace: bool = False,
    ) -> Self:
        """Apply a padding to each segment in this `DataQualityFlag`.

        This method either takes no arguments, in which case the value of
        the :attr:`~DataQualityFlag.padding` attribute will be used,
        or two values representing the padding for the start and end of
        each segment.

        For both the `start` and `end` paddings, a positive value means
        pad forward in time, so that a positive `start` pad or negative
        `end` padding will contract a segment at one or both ends,
        and vice-versa.

        This method will apply the same padding to both the
        `~DataQualityFlag.known` and `~DataQualityFlag.active` lists,
        but will not :meth:`~DataQualityFlag.coalesce` the result.

        Parameters
        ----------
        args : `float`, optional
            Two floats giving the start and end padding to apply.
            If not given, `self.padding` will be used.

        inplace : `bool`, optional
            Modify this object in-place.
            Default is `False`, i.e. return a copy of the original
            object with padded segments.

        Returns
        -------
        paddedflag : `DataQualityFlag`
            A view of the modified flag.
        """
        if not args:
            start, end = self.padding
        else:
            start, end = args

        if inplace:
            new = self
        else:
            new = self.copy()

        def _pad(s):
            return type(s)(s[0]+start, s[1]+end)

        new.known = type(self.known)(map(_pad, self.known))
        new.active = type(self.active)(map(_pad, self.active))

        return new

    def round(self, *, contract: bool = False) -> Self:
        """Round this flag to integer segments.

        Parameters
        ----------
        contract : `bool`, optional
            If `False` (default) expand each segment to the containing
            integer boundaries, otherwise contract each segment to the
            contained boundaries.

        Returns
        -------
        roundedflag : `DataQualityFlag`
            A copy of the original flag with the `active` and `known` segments
            padded out to integer boundaries..
        """
        def _round(seg: Segment) -> Segment:
            if contract:  # round inwards
                a = type(seg[0])(ceil(seg[0]))
                b = type(seg[1])(floor(seg[1]))
            else:  # round outwards
                a = type(seg[0])(floor(seg[0]))
                b = type(seg[1])(ceil(seg[1]))
            if a >= b:  # if segment is too short, return 'null' segment
                return type(seg)(0, 0)  # will get coalesced away
            return type(seg)(a, b)

        new = self.copy()
        new.active = type(new.active)(map(_round, new.active))
        new.known = type(new.known)(map(_round, new.known))
        return new.coalesce()

    def coalesce(self) -> Self:
        """Coalesce the segments for this flag.

        This method does two things:

        - `coalesces <SegmentList.coalesce>` the `~DataQualityFlag.known` and
          `~DataQualityFlag.active` segment lists
        - forces the `active` segments to be a proper subset of the `known`
          segments

        .. note::

            This operation is performed in-place.

        Returns
        -------
        self
            A view of this flag, not a copy.
        """
        self.known = self.known.coalesce()
        self.active = self.active.coalesce()
        self.active = (self.known & self.active).coalesce()
        return self

    def __repr__(self) -> str:
        """Return a representation of this flag."""
        prefix = f"<{type(self).__name__}("
        suffix = ")>"
        indent = " " * len(prefix)

        # format segment lists
        parts = {}
        for attr in ("known", "active"):
            parts[attr] = rep = textwrap.indent(
                str(getattr(self, attr)),
                f"{indent}      ",
            ).strip().splitlines()
            if len(rep) > 10:  # use ellipsis
                rep = [*rep[:3], f"{indent}      ...", *rep[-3:]]

        # print the thing
        return "".join((
            prefix,
            f"{os.linesep}{indent}".join([
                f"'{self.name}',",
                f"known={os.linesep.join(parts['known'])}",
                f"active={os.linesep.join(parts['active'])}",
                f"description='{self.description}'",
            ]),
            suffix,
        ))

    def copy(self) -> Self:
        """Build an exact copy of this flag.

        Returns
        -------
        flag : `DataQualityFlag`
            A deepcopy of the original flag.
        """
        return deepcopy(self)

    def plot(
        self,
        figsize: tuple[float, float] = (12, 4),
        xscale: str = "auto-gps",
        **kwargs,
    ) -> Plot:
        """Plot this flag on a segments projection.

        Parameters
        ----------
        figsize : `tuple` of `float`
            The size (width, height) of the figure to create.

        xscale: `str`
            The scaling to use for the X-axis (time axis).
            Default is ``"auto-gps"`` to dynamically choose the right scaling
            based on how much time is covered by the visible span.

        kwargs
            Other keyword arguments are passed to the
            :class:`~gwpy.plot.Plot` constructor.

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            The newly created figure, with populated Axes.

        See Also
        --------
        matplotlib.pyplot.figure
            For documentation of keyword arguments used to create the figure.

        matplotlib.figure.Figure.add_subplot
            For documentation of keyword arguments used to create the axes

        gwpy.plot.SegmentAxes.plot_segmentlist
            For documentation of keyword arguments used in rendering the data.
        """
        from matplotlib import rcParams

        from ..plot import Plot

        # get the right default label
        if "label" not in kwargs:
            if self.label:
                kwargs["label"] =  self.label
            elif rcParams["text.usetex"]:
                kwargs["label"] =  self.texname
            else:
                kwargs["label"] =  self.name

        return Plot(
            self,
            projection="segments",
            figsize=figsize,
            xscale=xscale,
            **kwargs,
        )

    def _parse_name(
        self,
        name: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        """Parse a flag name and set properties of this flag.

        Parameters
        ----------
        name : `str`, `None`
            The full name of a `DataQualityFlag` to parse, e.g.
            ``"H1:DMT-SCIENCE:1"``, or `None` to set all components
            to `None`.

        Returns
        -------
        ifo : `str` or `None`
            The IFO prefix for this flag.

        tag : `str` or `None`
            The name tag for this flag.

        version : `int` or `None`
            The version number for this flag.

        Raises
        ------
        ValueError
            If the input ``name`` cannot be parsed into
            {ifo}:{tag}:{version} format.
        """
        if name is None:
            self.ifo = None
            self.tag = None
            self.version = None
        elif match := IFO_TAG_VERSION_REGEX.match(name):
            self.ifo = match["ifo"]
            self.tag = match["tag"]
            self.version = int(match["version"])
        elif match := IFO_TAG_REGEX.match(name):
            self.ifo = match["ifo"]
            self.tag = match["tag"]
            self.version = None
        elif match := TAG_VERSION_REGEX.match(name):
            self.ifo = None
            self.tag = match["tag"]
            self.version = int(match["version"])
        else:
            msg = (
                f"no flag name structure detected in '{name}', flags "
                "should be named as '{ifo}:{tag}:{version}'; "
                "for arbitrary strings, use the "
                "`DataQualityFlag.label` attribute"
            )
            raise ValueError(msg)
        return self.ifo, self.tag, self.version

    def __and__(self, other: Self) -> Self:
        """Find the intersection of this one and ``other``."""
        return self.copy().__iand__(other)

    def __iand__(self, other: Self) -> Self:
        """Intersect this flag with ``other`` in-place."""
        self.known &= other.known
        self.active &= other.active
        return self

    def __sub__(self, other: Self) -> Self:
        """Find the difference between this flag and another."""
        return self.copy().__isub__(other)

    def __isub__(self, other: Self) -> Self:
        """Subtract the ``other`` `DataQualityFlag` from this one in-place."""
        self.known &= other.known
        self.active -= other.active
        self.active &= self.known
        return self

    def __or__(self, other: Self) -> Self:
        """Find the union of this flag and ``other``."""
        return self.copy().__ior__(other)

    def __ior__(self, other: Self) -> Self:
        """Add the ``other`` `DataQualityFlag` to this one in-place."""
        self.known |= other.known
        self.active |= other.active
        return self

    __add__ = __or__
    __iadd__ = __ior__

    def __xor__(self, other: Self) -> Self:
        """Find the exclusive OR of this one and ``other``."""
        return self.copy().__ixor__(other)

    def __ixor__(self, other: Self) -> Self:
        """Exclusive OR this flag with ``other`` in-place."""
        self.known &= other.known
        self.active ^= other.active
        return self

    def __invert__(self) -> Self:
        """Return the logical inverse of this flag."""
        new = self.copy()
        new.active = ~self.active
        new.active &= new.known
        return new


class DataQualityDict(dict):
    """An `dict` of (key, `DataQualityFlag`) pairs."""

    _EntryClass: type[DataQualityFlag] = DataQualityFlag

    # -- i/o -------------------------

    read = UnifiedReadWriteMethod(DataQualityDictRead)
    write = UnifiedReadWriteMethod(DataQualityDictWrite)

    # -- classmethods ----------------

    @classmethod
    @_check_on_error
    def query_dqsegdb(
        cls,
        flags: list[str],
        *args: SupportsToGps | Segment | SegmentList,
        host: str | None = DEFAULT_SEGMENT_SERVER,
        on_error: str = "raise",
        parallel: int = 10,
        **kwargs,
    ) -> Self:
        """Query the advanced LIGO DQSegDB for a list of flags.

        Parameters
        ----------
        flags : `iterable`
            A list of flag names for which to query.

        args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a `SegmentList`
            defining a number of summary segments.

        host : `str`, optional
            Name or URL of the DQSegDB instance to talk to.
            Defaults to :func:`dqsegdb2.utils.get_default_host`.

        on_error : `str`, optional
            how to handle an error querying for one flag, one of

            - `'raise'` (default): raise the Exception
            - `'warn'`: print a warning
            - `'ignore'`: move onto the next flag as if nothing happened

        parallel : `int`, optional
            Maximum number of threads to use for parallel connections to
            the DQSegDB host.

        kwargs
            All other keyword arguments are passed to
            :func:`dqsegdb2.query.query_segments`.

        Returns
        -------
        flagdict : `DataQualityDict`
            An ordered `DataQualityDict` of (name, `DataQualityFlag`)
            pairs.

        Examples
        --------
        The GPS interval(s) of interest can be passed as two arguments
        specifing the start and end of a single interval:

        >>> DataQualityDict.query_dqsegdb(["X1:OBSERVING:1", "Y1:OBSERVING:1"], start, end)

        Or, as a single `Segment`:

        >>> DataQualityDict.query_dqsegdb(["X1:OBSERVING:1", "Y1:OBSERVING:1"], interval)

        Or, as a `SegmentList` specifying multiple intervals.

        >>> DataQualityDict.query_dqsegdb(["X1:OBSERVING:1", "Y1:OBSERVING:1"], intervals)
        """  # noqa: E501
        # parse segments
        qsegs = _parse_query_segments(args, cls.query_dqsegdb)

        # thread function
        def _query(flag: str) -> DataQualityFlag | None:
            try:
                return cls._EntryClass.query_dqsegdb(
                    flag,
                    qsegs,
                    host=host,
                    **kwargs,
                )
            except Exception as exc:
                exc.args = (f"{exc} [{flag}]",)
                if on_error == "raise":
                    raise
                if on_error == "warn":
                    warnings.warn(str(exc), stacklevel=2)
                return None

        # execute queries in threads
        out = cls()
        parallel = min(parallel, len(flags))
        with ThreadPoolExecutor(
            max_workers=parallel,
            thread_name_prefix=f"{cls.__name__}.query_dqsegdb",
        ) as pool:
            for flag in filter(None, pool.map(_query, flags)):
                out[flag.name] = flag

        return out

    # alias for compatibility
    query = query_dqsegdb

    @classmethod
    def from_veto_definer_file(
        cls,
        source: str,
        start: SupportsToGps | None = None,
        end: SupportsToGps | None = None,
        ifo: str | None = None,
        **read_kw,
    ) -> Self:
        """Read a `DataQualityDict` from a LIGO_LW XML VetoDefinerTable.

        Parameters
        ----------
        source : `str`, `Path`, `file`
            Path or URL of veto definer file to read, or open file.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS end time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        ifo : `str`, optional
            Interferometer prefix whose flags you want to read.
            Default is `None` (read all flags).

        read_kw
            Other keyword arguments are passed to
            `~astropy.table.Table.read` when reading the veto definer file.

        Returns
        -------
        flags : `DataQualityDict`
            A `DataQualityDict` of flags parsed from the `veto_def_table`
            of the input file.

        Notes
        -----
        This method does not automatically `~DataQualityDict.populate`
        the `active` segment list of any flags, a separate call should
        be made for that as follows

        >>> flags = DataQualityDict.from_veto_definer_file('/path/to/file.xml')
        >>> flags.populate()
        """
        from ..table import Table

        # read veto definer file as a table
        read_kw.setdefault("tablename", "veto_definer")
        read_kw.setdefault("columns", [
            "ifo",
            "name",
            "version",
            "category",
            "comment",
            "start_time",
            "start_pad",
            "end_time",
            "end_pad",
        ])
        veto_def_table = Table.read(source, **read_kw)

        # handle GPS types
        startgps = None if start is None else to_gps(start)
        endgps = None if end is None else to_gps(end)

        def _keep(row: astropy.table.Row) -> bool:
            """Return `True` if this row is relevant."""
            if ifo and row["ifo"] != ifo:
                return False
            if startgps and 0 < row["end_time"] <= startgps:
                return False
            return not (endgps and row["start_time"] >= endgps)

        # parse flag definitions
        out = cls()
        for row in filter(_keep, veto_def_table):
            # pin the times
            if startgps:
                row["start_time"] = max(row["start_time"], startgps)
            if endgps and not row["end_time"]:
                row["end_time"] = endgps
            elif endgps:
                row["end_time"] = min(row["end_time"], endgps)
            flag = DataQualityFlag.from_veto_def(row)
            if flag.name in out:
                out[flag.name] |= flag
            else:
                out[flag.name] = flag
        return out

    @classmethod
    def from_ligolw_tables(
        cls,
        segmentdeftable: igwn_ligolw.lsctables.SegmentDefTable,
        segmentsumtable: igwn_ligolw.lsctables.SegmentSumTable,
        segmenttable: igwn_ligolw.lsctables.SegmentTable,
        names: list[str] | None = None,
        gpstype: type[SupportsFloat] | Callable[[float], SupportsFloat] = LIGOTimeGPS,
        on_missing: str = "error",
    ) -> Self:
        """Build a `DataQualityDict` from a set of LIGO_LW segment tables.

        Parameters
        ----------
        segmentdeftable : :class:`~igwn_ligolw.lsctables.SegmentDefTable`
            The ``segment_definer`` table to read.

        segmentsumtable : :class:`~igwn_ligolw.lsctables.SegmentSumTable`
            The ``segment_summary`` table to read.

        segmenttable : :class:`~igwn_ligolw.lsctables.SegmentTable`
            The ``segment`` table to read.

        names : `list` of `str`, optional
            A list of flag names to read.
            Default is to read all names.

        gpstype : `type`, `callable`, optional
            Class to use for GPS times in returned objects, can be a function
            to convert GPS time to something else.
            Default is `~gwpy.time.LIGOTimeGPS`.

        on_missing : `str`, optional
            Action to take when a one or more ``names`` are not found in
            the ``segment_definer`` table, one of

            - ``'ignore'`` : do nothing
            - ``'warn'`` : print a warning
            - ``error'`` : raise a `ValueError`

        Returns
        -------
        dqdict : `DataQualityDict`
            A dict of `DataQualityFlag` objects populated from the LIGO_LW tables.
        """
        out = cls()
        id_: dict[str, list[int]] = {}  # need to record relative IDs from LIGO_LW

        # read segment definers and generate DataQualityFlag object
        for row in segmentdeftable:
            ifos = sorted(row.instruments)
            ifo = "".join(ifos) if ifos else None
            tag = row.name
            version = row.version
            name = ":".join(
                str(k) for k in (ifo, tag, version) if k is not None
            )
            if names is None or name in names:
                out[name] = DataQualityFlag(name)
                thisid = int(row.segment_def_id)
                try:
                    id_[name].append(thisid)
                except KeyError:
                    id_[name] = [thisid]

        # verify all requested flags were found
        for flag in names or []:
            if flag not in out and on_missing != "ignore":
                msg = f"no segment definition found for flag='{flag}' in file"
                if on_missing == "warn":
                    warnings.warn(msg, stacklevel=2)
                else:
                    raise ValueError(msg)


        def _parse_segments(
            table: igwn_ligolw.ligolw.Table,
            listattr: str,
        ) -> None:
            """Parse a table into the target DataQualityDict."""
            # handle missing *_ns columns in LIGO_LW XML
            # (LIGO DMT doesn't/didn't write them)
            if "start_time_ns" in table.columnnames:
                row_segment = operator.attrgetter("segment")
            else:
                row_segment = operator.attrgetter("start_time", "end_time")

            for row in table:
                for flag in out:
                    # match row ID to list of IDs found for this flag
                    if int(row.segment_def_id) in id_[flag]:
                        getattr(out[flag], listattr).append(
                            Segment(*map(gpstype, row_segment(row))),
                        )
                        break

        # read segment summary table as 'known'
        _parse_segments(segmentsumtable, "known")

        # read segment table as 'active'
        _parse_segments(segmenttable, "active")

        return out

    def to_ligolw_tables(
        self,
        **attrs,
    ) -> tuple[
        igwn_ligolw.lsctables.SegmentDefTable,
        igwn_ligolw.lsctables.SegmentSumTable,
        igwn_ligolw.lsctables.SegmentTable,
    ]:
        """Convert this `DataQualityDict` into a trio of LIGO_LW segment tables.

        Parameters
        ----------
        attrs
            Other attributes to add to all rows in all tables
            (e.g. ``'process_id'``).

        Returns
        -------
        segmentdeftable : :class:`~igwn_ligolw.lsctables.SegmentDefTable`
            The ``segment_definer`` table.

        segmentsumtable : :class:`~igwn_ligolw.lsctables.SegmentSumTable`
            The ``segment_summary`` table.

        segmenttable : :class:`~igwn_ligolw.lsctables.SegmentTable`
            The ``segment`` table.
        """
        from igwn_ligolw import lsctables

        from ..io.ligolw import to_table_type as to_ligolw_table_type

        segdeftab_class = lsctables.SegmentDefTable
        segsumtab_class = lsctables.SegmentSumTable
        segtab_class = lsctables.SegmentTable
        segdeftab = segdeftab_class.new()
        segsumtab = segsumtab_class.new()
        segtab = segtab_class.new()

        def _write_attrs(
            table: igwn_ligolw.ligolw.Table,
            row: igwn_ligolw.ligolw.Row,
        ) -> None:
            """Write custom attributes to this row."""
            for key, val in attrs.items():
                setattr(row, key, to_ligolw_table_type(val, table, key))

        # write flags to tables
        for flag in self.values():
            # segment definer
            segdef = segdeftab.RowType()
            for col in segdeftab.columnnames:  # default all columns to None
                setattr(segdef, col, None)
            segdef.instruments = {flag.ifo}
            segdef.name = flag.tag
            segdef.version = flag.version
            segdef.comment = flag.description
            segdef.insertion_time = to_gps("now").gpsSeconds
            segdef.segment_def_id = segdeftab_class.get_next_id()
            _write_attrs(segdeftab, segdef)
            segdeftab.append(segdef)

            # write segment summary (known segments)
            for vseg in flag.known:
                segsum = segsumtab.RowType()
                for col in segsumtab.columnnames:  # default columns to None
                    setattr(segsum, col, None)
                segsum.segment_def_id = segdef.segment_def_id
                segsum.segment = map(LIGOTimeGPS, vseg)
                segsum.comment = None
                segsum.segment_sum_id = segsumtab_class.get_next_id()
                _write_attrs(segsumtab, segsum)
                segsumtab.append(segsum)

            # write segment table (active segments)
            for aseg in flag.active:
                seg = segtab.RowType()
                for col in segtab.columnnames:  # default all columns to None
                    setattr(seg, col, None)
                seg.segment_def_id = segdef.segment_def_id
                seg.segment = map(LIGOTimeGPS, aseg)
                seg.segment_id = segtab_class.get_next_id()
                _write_attrs(segtab, seg)
                segtab.append(seg)

        return segdeftab, segsumtab, segtab

    # -- methods ---------------------

    def coalesce(self) -> Self:
        """Coalesce all segments lists in this `DataQualityDict`.

        **This method modifies this object in-place.**

        Returns
        -------
        self
            A view of this flag, not a copy.
        """
        for flag in self:
            self[flag].coalesce()
        return self

    @_check_on_error
    def populate(
        self,
        source: str | None = DEFAULT_SEGMENT_SERVER,
        segments: SegmentList | None = None,
        *,
        pad: bool = True,
        on_error: str = "raise",
        **kwargs,
    ) -> Self:
        """Query the segment database for each flag's active segments.

        This method assumes all of the metadata for each flag have been
        filled. Minimally, the following attributes must be filled

        .. autosummary::

           ~DataQualityFlag.name
           ~DataQualityFlag.known

        Segments will be fetched from the database, with any
        :attr:`~DataQualityFlag.padding` added on-the-fly.

        Entries in this dict will be modified in-place.

        Parameters
        ----------
        source : `str`
            Source of segments for this flag. This must be
            either a URL for a segment database or a path to a file on disk.

        segments : `SegmentList`, optional
            A list of known segments during which to query, if not given,
            existing known segments for flags will be used.

        pad : `bool`, optional, default: `True`
            Apply the `~DataQualityFlag.padding` associated with each
            flag, default: `True`.

        on_error : `str`
            How to handle an error querying for one flag, one of

            - `'raise'` (default): raise the Exception
            - `'warn'`: print a warning
            - `'ignore'`: move onto the next flag as if nothing happened

        kwargs
            Any other keyword arguments to be passed to
            :meth:`DataQualityFlag.query` or :meth:`DataQualityFlag.read`.

        Returns
        -------
        self : `DataQualityDict`
            A reference to the modified DataQualityDict
        """
        # format source
        url = urlparse(source)
        # query DQSegDB if the URL is just a hostname
        use_dqsegdb = bool(url.netloc and url.path in ("", "/"))

        # if given the segments to use, query DQSegDB for everthing now
        if not use_dqsegdb:
            tmp = type(self).read(source, **kwargs)
        elif segments is not None:
            segments = SegmentList(map(Segment, segments))
            tmp = type(self).query_dqsegdb(
                self.keys(),
                segments,
                host=source,
                on_error=on_error,
                **kwargs,
            )

        # apply padding and wrap to given known segments
        for key in self:
            if use_dqsegdb and segments is None:
                # get segments for this flag now
                try:
                    tmp = {key: self[key].query_dqsegdb(
                        self[key].name,
                        self[key].known,
                        host=source,
                        **kwargs,
                    )}
                except RequestException as exc:
                    if on_error == "raise":
                        raise
                    if on_error == "warn":
                        warnings.warn(
                            f"Error querying for '{key}': {exc}",
                            stacklevel=2,
                        )
                    continue
            self[key].known &= tmp[key].known
            self[key].active = tmp[key].active
            if pad:
                self[key] = self[key].pad(inplace=True)
                if segments is not None:
                    self[key].known &= segments
                    self[key].active &= segments
        return self

    def copy(self, *, deep: bool = False) -> Self:
        """Build a copy of this dictionary.

        Parameters
        ----------
        deep : `bool`, optional, default: `False`
            perform a deep copy of the original dictionary with a fresh
            memory address

        Returns
        -------
        flag2 : `DataQualityFlag`
            a copy of the original dictionary
        """
        if deep:
            return deepcopy(self)
        return shallowcopy(self)

    def __iand__(self, other: Self) -> Self:
        """Intersect this dict with ``other`` in-place."""
        for key, value in other.items():
            if key in self:
                self[key] &= value
            else:
                self[key] = self._EntryClass()
        return self

    def __and__(self, other: Self) -> Self:
        """Find the intersection of this dict and ``other``."""
        if (
            sum(len(s.active) for s in self.values())
            <= sum(len(s.active) for s in other.values())
        ):
            return self.copy(deep=True).__iand__(other)
        return other.copy(deep=True).__iand__(self)

    def __ior__(self, other: Self) -> Self:
        """Add the ``other`` `DataQualityDict` to this one in-place."""
        for key, value in other.items():
            if key in self:
                self[key] |= value
            else:
                self[key] = shallowcopy(value)
        return self

    def __or__(self, other: Self) -> Self:
        """Find the union of this dict and ``other``."""
        if (
            sum(len(s.active) for s in self.values())
            >= sum(len(s.active) for s in other.values())
        ):
            return self.copy(deep=True).__ior__(other)
        return other.copy(deep=True).__ior__(self)

    __iadd__ = __ior__
    __add__ = __or__

    def __isub__(self, other: Self) -> Self:
        """Subtract the ``other`` `DataQualityDict` from this one in-place."""
        for key, value in other.items():
            if key in self:
                self[key] -= value
        return self

    def __sub__(self, other: Self) -> Self:
        """Find the difference between this dict and another."""
        return self.copy(deep=True).__isub__(other)

    def __ixor__(self, other: Self) -> Self:
        """Exclusive OR this dict with ``other`` in-place."""
        for key, value in other.items():
            if key in self:
                self[key] ^= value
        return self

    def __xor__(self, other: Self) -> Self:
        """Find the exclusive OR of this dict and ``other``."""
        return self.copy(deep=True).__ixor__(other)

    def __invert__(self) -> Self:
        """Return the logical inverse of this dict."""
        new = self.copy(deep=True)
        for key, value in new.items():
            new[key] = ~value
        return new

    def union(self) -> DataQualityFlag:
        """Return the union of all flags in this dict.

        Returns
        -------
        union : `DataQualityFlag`
            a new `DataQualityFlag` who's active and known segments
            are the union of those of the values of this dict
        """
        usegs = reduce(operator.or_, self.values())
        usegs.name = " | ".join(self.keys())
        return usegs

    def intersection(self) -> DataQualityFlag:
        """Return the intersection of all flags in this dict.

        Returns
        -------
        intersection : `DataQualityFlag`
            a new `DataQualityFlag` who's active and known segments
            are the intersection of those of the values of this dict
        """
        isegs = reduce(operator.and_, self.values())
        isegs.name = " & ".join(self.keys())
        return isegs

    def plot(
        self,
        label: str = "key",
        **kwargs,
    ) -> Plot:
        """Plot this dict on a segments projection.

        Parameters
        ----------
        label : `str`, optional
            Labelling system to use, or fixed label for all flags,
            special values include

            - ``'key'``: use the key of the `DataQualityDict`,
            - ``'name'``: use the :attr:`~DataQualityFlag.name` of the flag

            If anything else, that fixed label will be used for all lines.

        kwargs
            All keyword arguments are passed to the
            :class:`~gwpy.plot.Plot` constructor.

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            the newly created figure, with populated Axes.

        See Also
        --------
        matplotlib.pyplot.figure
            For documentation of keyword arguments used to create the figure.

        matplotlib.figure.Figure.add_subplot
            For documentation of keyword arguments used to create the axes.

        gwpy.plot.SegmentAxes.plot_segmentlist
            For documentation of keyword arguments used in rendering the data.
        """
        # make plot
        from ..plot import Plot
        return Plot(
            [self],
            projection="segments",
            label=label,
            **kwargs,
        )
