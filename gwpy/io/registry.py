# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2021)
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

"""I/O registry extensions on top of `astropy.io.registry`

This module imports a subset of the useful functions from
:mod:`astropy.io.registry` for convenience.
"""

import sys
import warnings
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import nullcontext
from functools import wraps

from astropy.io import registry as astropy_registry
from astropy.io.registry import (
    UnifiedReadWriteMethod,  # noqa: F401
    compat,
)
from astropy.utils.data import get_readable_fileobj

from ..utils.env import bool_env
from ..utils.progress import progress_bar
from .remote import open_remote_file
from .utils import (
    FILE_LIKE,
    file_list,
)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- utilities -----------------------

def list_identifier(
    identifier,
):
    """Decorate an I/O identifier to handle a list of files as input.

    This function tries to resolve a single file path as a `str` from any
    one or more file-like objects in the ``filepath`` or ``args`` inputs
    to pass to the underlying identifier for comparison.
    """
    @wraps(identifier)
    def decorated_func(origin, filepath, fileobj, *args, **kwargs):
        target = filepath  # thing to search
        if target is None and args:
            target = args[0]
        try:
            filepath = file_list(target)[0]
        except (
            IndexError,  # empty list
            ValueError,  # target can't be resolved as a list of file-like
        ):
            pass
        return identifier(origin, filepath, fileobj, *args, **kwargs)
    return decorated_func


# -- legacy format -------------------

@wraps(compat.default_registry.register_identifier)
def legacy_register_identifier(
    data_format,
    data_class,
    identifier,
    force=False,
):
    # pylint: disable=missing-docstring
    return compat.default_registry.register_identifier(
        data_format,
        data_class,
        list_identifier(identifier),
        force=force,
    )


compat.register_identifier = legacy_register_identifier


def get_read_format(cls, source, args, kwargs):
    """Determine the read format for a given input source
    """
    ctx = None
    if isinstance(source, FILE_LIKE):
        fileobj = source
        filepath = source.name if hasattr(source, "name") else None
    else:
        filepath = source
        try:
            ctx = get_readable_fileobj(filepath, encoding="binary")
            fileobj = ctx.__enter__()  # pylint: disable=no-member
        except OSError:
            raise
        except Exception:  # pylint: disable=broad-except
            fileobj = None
    try:
        return compat.default_registry._get_valid_format(
            "read",
            cls,
            filepath,
            fileobj,
            args,
            kwargs,
        )
    finally:
        if ctx is not None:
            ctx.__exit__(*sys.exc_info())  # pylint: disable=no-member


# -- Unified I/O format --------------

class UnifiedIORegistry(astropy_registry.UnifiedIORegistry):
    """UnifiedIORegistry that can handle reading files in parallel.
    """
    def identify_format(
        self,
        origin,
        data_class_required,
        path,
        fileobj,
        args,
        kwargs,
    ):
        if fileobj is None:
            try:
                path = file_list(path)[0]
            except (
                IndexError,  # list is empty
                ValueError,  # failed to parse as list-like
            ):
                pass
        return super().identify_format(
            origin,
            data_class_required,
            path,
            fileobj,
            args,
            kwargs,
        )

    def register_identifier(
        self,
        data_format,
        data_class,
        identifier,
        force=False,
    ):
        return super().register_identifier(
            data_format,
            data_class,
            list_identifier(identifier),
            force=force,
        )

    register_identifier.__doc__ = (
        astropy_registry.UnifiedIORegistry.register_identifier.__doc__
    )


default_registry = UnifiedIORegistry()


class UnifiedRead(astropy_registry.UnifiedReadWrite):
    """Base ``Class.read()`` implementation that handles parallel reads.
    """
    def __init__(
        self,
        instance,
        cls,
        registry=default_registry,
    ):
        super().__init__(
            instance,
            cls,
            "read",
            registry=registry,
        )

    def _read_single_file(
        self,
        *args,
        **kwargs,
    ):
        return self.registry.read(self._cls, *args, **kwargs)

    @staticmethod
    def _format_input_list(source):
        """Format the input arguments to include a list of files.
        """
        # parse input as a list of files
        try:  # try and map to a list of file-like objects
            return file_list(source)
        except ValueError:  # otherwise treat as single file
            return [source]

    def __call__(
        self,
        merge_function,
        source,
        *args,
        format=None,
        cache=None,
        parallel=1,
        verbose=False,
        **kwargs,
    ):
        """Execute ``cls.read()``.

        This method generalises parallel reading of lists of files for
        any input class. The output of each parallel read is then merged
        by ``merge_function`` which should have the following signature.

            def merge_function(cls: Type, instances: list[Type]) -> Type

        i.e take in the type object and a list of instances, and return
        a single instance of the same type.

        This method also generalises downloading files from remote URLs
        over HTTP or via `fsspec`.
        """
        cls = self._cls

        # handle deprecated keyword
        if nproc := kwargs.pop("nproc", None):
            warnings.warn(
                f"the 'nproc' keyword to {cls.__name__}.read was renamed "
                "parallel; this warning will be an error in the future.",
                DeprecationWarning,
            )
            parallel = nproc

        # set default cache based on environment
        if cache is None:
            cache = bool_env("GWPY_CACHE", False)

        # get the input as a list of inputs
        sources = self._format_input_list(source)

        # handle progress based on the number of inputs
        show_progress = False  # single file download progress
        if len(sources) == 1:
            show_progress = verbose
            verbose = False
        if verbose is True and format:
            verbose = f"Reading ({format})"
        elif verbose is True:
            verbose = "Reading"
        if verbose:
            progress = progress_bar(
                total=len(sources),
                desc=verbose,
            )
        else:
            progress = None

        remote_kwargs = {
            key: kwargs.pop(key) for key in (
                "sources",
                "http_headers",
                "use_fsspec",
                "fsspec_kwargs",
            ) if key in kwargs
        }

        # single file reader
        def _read(arg):
            # if arg is a str, presume it represents a URI, so try and open it
            if isinstance(arg, str):
                ctx = open_remote_file(
                    arg,
                    cache=cache,
                    show_progress=show_progress,
                    **remote_kwargs,
                )
            # otherwise just pass it along unmodified
            else:
                ctx = nullcontext(arg)

            with ctx as file:
                return self.registry.read(
                    self._cls,
                    file,
                    *args,
                    format=format,
                    cache=cache,
                    **kwargs,
                )

        # read all files in parallel threads
        outputs = []
        with ThreadPoolExecutor(
            max_workers=parallel,
        ) as pool:
            futures = [pool.submit(_read, source) for source in sources]
            for future in as_completed(futures):
                outputs.append(future.result())
                if progress:
                    progress.update(1)

        return merge_function(outputs)


class UnifiedWrite(astropy_registry.UnifiedReadWrite):
    """Base ``Class.write()`` implementation.
    """
    def __init__(
        self,
        instance,
        cls,
        registry=default_registry,
    ):
        super().__init__(
            instance,
            cls,
            "write",
            registry=registry,
        )

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        """Execute ``instance.write()``.
        """
        instance = self._instance
        return self.registry.write(instance, *args, **kwargs)


# -- utilities -----------------------

def inherit_unified_io(klass):
    """Re-register all Unified I/O readers/writes/identifiers from a parent
    class to a child.

    Only works with the first parent in the inheritance tree.

    This allows the Unified I/O registrations for the child class to be
    modified independently of the parent.
    """
    parent = klass.__mro__[1]
    parent_registry = parent.read.registry
    child_registry = klass.read.registry
    for row in parent_registry.get_formats(data_class=parent):
        name = row["Format"]

        # read
        if row["Read"].lower() == "yes":
            child_registry.register_reader(
                name,
                klass,
                parent_registry.get_reader(name, parent),
                force=False,
            )
        # write
        if row["Write"].lower() == "yes":
            child_registry.register_writer(
                name,
                klass,
                parent_registry.get_writer(name, parent),
                force=False,
            )
        # identify
        if row["Auto-identify"].lower() == "yes":
            child_registry.register_identifier(
                name,
                klass,
                parent_registry._identifiers[(name, parent)],
                force=False,
            )
    return klass
