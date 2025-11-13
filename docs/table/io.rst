:tocdepth: 3

.. currentmodule:: gwpy.table

.. _gwpy-table-io:

#################################################################################
Reading and writing :class:`~astropy.table.Table` and :class:`EventTable` objects
#################################################################################

.. note::

    This document complements the upstream Astropy documentation on
    reading/writing :class:`~astropy.table.Table` objects, please refer to
    :any:`astropy:read_write_tables`.

Astropy provides an excellent :any:`unified input/output <astropy:table_io>`
system for the `~astropy.table.Table` object, and GWpy extends upon that to
include common gravitational-wave file types, as well as providing
event-specific input/output registrations for event data.
In the most general case you can read a table of data as follows:

.. code-block:: python
    :name: gwpy-table-io-example
    :caption: Reading an `EventTable` from an ASCII file

    >>> from gwpy.table import EventTable
    >>> table = EventTable.read('mydata.txt')

:meth:`EventTable.read` will attempt to automatically identify the file
format based on the file extension and/or the contents of the file, however,
the ``format`` keyword argument can be used to manually identify the input
file-format.

The :meth:`~EventTable.read` and :meth:`~EventTable.write` methods take
different arguments and keywords based on the input/output file format,
see :ref:`gwpy-table-io-formats` for details on reading/writing for
each of the built-in formats.

.. _gwpy-table-gwosc:

************************************
Accessing Open Data event catalogues
************************************

|GWOSCl| publishes the Gravitational-Wave Transient Event Catalogues (GWTCs),
allowing public access to tables of gravitational wave events and their
parameters.

GWpy's :class:`EventTable` class comes with a
:meth:`~EventTable.fetch_open_data` method that queries the GWOSC database
to download data for the relevant catalogue.

.. _gwpy-table-gwosc-query:

==============
Simple queries
==============

The simplest query just requires the catalogue name, and will return all parameters
for all events in the catalogue:

.. code-block:: python
    :name: gwpy-table-io-fetch_open_data
    :caption: Fetching the GWTC-2 catalogue using :meth:`EventTable.fetch_open_data` (output correct as of |today|)

    >>> from gwpy.table import EventTable
    >>> events = EventTable.fetch_open_data("GWTC-2")
    >>> print(events)
           name        final_mass_source_lower ... mass_2_source_upper mass_1_source_upper
                                               ...
    ------------------ ----------------------- ... ------------------- -------------------
    GW190408_181802-v1                    -2.8 ...                 3.3                 5.1
           GW190412-v3                    -3.8 ...                 1.6                 4.7
    GW190413_052954-v1                    -9.2 ...                 7.3                12.6
    GW190413_134308-v1                   -11.4 ...                11.7                13.5
    GW190421_213856-v1                    -8.7 ...                 8.0                10.4
    GW190424_180648-v1                   -10.1 ...                 7.6                11.1
           GW190425-v2                    None ...                 0.3                 0.6
    GW190426_152155-v1                    None ...                 0.8                 3.9
    GW190503_185404-v1                    -7.7 ...                 7.7                 9.2
    GW190512_180714-v1                    -3.5 ...                 3.6                 5.3
    GW190513_205428-v1                    -5.8 ...                 7.7                 9.5
    GW190514_065416-v1                   -10.4 ...                 9.3                14.7
    GW190517_055101-v1                    -8.9 ...                 7.0                11.7
    GW190519_153544-v1                   -13.8 ...                11.0                10.7
           GW190521-v3                   -22.4 ...                22.7                28.7
    GW190521_074359-v1                    -4.4 ...                 5.4                 5.9
    GW190527_092055-v1                    -9.3 ...                10.5                16.4
    GW190602_175927-v1                   -14.9 ...                14.3                15.7
    GW190620_030421-v1                   -12.1 ...                12.2                16.0
    GW190630_185205-v1                    -4.6 ...                 5.2                 6.9
    GW190701_203306-v1                    -8.9 ...                 8.7                11.8
    GW190706_222641-v1                   -13.5 ...                14.6                14.6
    GW190707_093326-v1                    -1.3 ...                 1.4                 3.3
    GW190708_232457-v1                    -1.8 ...                 2.0                 4.7
    GW190719_215514-v1                   -10.2 ...                 9.0                18.0
    GW190720_000836-v1                    -2.2 ...                 2.3                 6.7
    GW190727_060333-v1                    -7.5 ...                 7.1                 9.5
    GW190728_064510-v1                    -1.3 ...                 1.7                 7.2
    GW190731_140936-v1                   -10.8 ...                 9.7                12.2
    GW190803_022701-v1                    -8.5 ...                 7.8                10.6
           GW190814-v2                    -0.9 ...                0.08                 1.1
    GW190828_063405-v1                    -4.3 ...                 4.6                 5.8
    GW190828_065509-v1                    -4.5 ...                 3.6                 7.0
    GW190909_114149-v1                   -16.8 ...                13.4                52.7
    GW190910_112807-v1                    -8.6 ...                 6.3                 7.6
    GW190915_235702-v1                    -6.0 ...                 5.6                 9.5
    GW190924_021846-v1                    -1.0 ...                 1.4                 7.0
    GW190929_012149-v1                   -25.3 ...                19.3                33.0
    GW190930_133541-v1                    -1.5 ...                 1.7                12.4

The full list of available columns can be queried as follows:

.. code-block:: python
    :name: gwpy-table-io-fetch_open_data-info
    :caption: Printing the columns of an `EventTable` (output correct as of |today|)

    >>> print(events.info)
    <EventTable masked=True length=39>
                  name                dtype    unit            description
    -------------------------------- ------- ------- --------------------------------
                               name   str18
            final_mass_source_lower  object                  final_mass_source_lower
            chirp_mass_source_lower float64                  chirp_mass_source_lower
            total_mass_source_upper float64                  total_mass_source_upper
                   chirp_mass_lower float64                         chirp_mass_lower
    ...

.. admonition:: GWOSC catalogues

    For more details on the GWOSC catalogues, see https://gwosc.org/eventapi/html/.

.. _gwpy-table-gwosc-filter:

================
Filtered queries
================

The columns returned can be selected using the ``column`` keyword,
and mathematical condition filters can be applied on-the-fly
using the ``where`` keyword:

.. code-block:: python
    :name: gwpy-table-io-fetch_open_data-filtered
    :caption: Downloading a sub-set of a catalogue from GWOSC (output correct as of |today|)

    >>> t = EventTable.fetch_open_data(
    ...     "GWTC-2",
    ...     where="mass_1_source < 4",
    ...     columns=["name", "mass_1_source", "mass_2_source", "luminosity_distance"],
    ... )
    >>> print(t)
        name    mass_1_source mass_2_source luminosity_distance
                   solMass       solMass            Mpc
    ----------- ------------- ------------- -------------------
    GW190425-v2           2.0           1.4               160.0

.. admonition:: More on filtering an `EventTable`

    For more information on filtering, see :ref:`gwpy-table-filter`.

.. _gwpy-table-io-formats:

***************************
Accessing GravitySpy events
***************************

|GravitySpy|_ is a citizen-science project that enables the public to
characterize and classify glitches in IGWN detector data.
The :class:`GravitySpyTable` subclass of :class:`EventTable` provides
methods to query GravitySpy for various tables of classified events.

=====================
Full database queries
=====================

The :meth:`GravitySpyTable.fetch` method (inherited directly from
`EventTable <EventTable.fetch>`) enables querying the Gravity Spy database
directly:

.. code-block:: python

    >>> from gwpy.table import GravitySpyTable
    >>> blips = GravitySpyTable.fetch(
    ...     source="gravityspy",
    ...     tablename="glitches",
    ...     where="Label=Blip",
    ... )

.. warning::

    Login credentials are required to support this query.
    IGWN members with LIGO.ORG credentials can find the required
    credentials at https://secrets.ligo.org/secrets/144/.

===================
Similarity searches
===================

The :meth:`GravitySpyTable.search` method enables performing a
`Similarity Search <https://gravityspytools.ciera.northwestern.edu/search/>`__
given the ID of a Gravity Spy event:

.. code-block:: python

    >>> from gwpy.table import GravitySpyTable
    >>> similar = GravitySpyTable.search("8FHTgA8MEu", howmany=5)
    >>> print(similar)
    ifo  peak_frequency  links_subjects ml_label searchedID ...
    --- ---------------- -------------- -------- ---------- ...
     H1 84.4759674072266      5740011.0 Scratchy 8FHTgA8MEu ...
     L1   128.8896484375     20892636.0 Scratchy 8FHTgA8MEu ...
     L1 73.4049224853516     20892632.0 Scratchy 8FHTgA8MEu ...
     L1 75.5168914794922     20892526.0 Scratchy 8FHTgA8MEu ...
     L1 144.991333007812      8644242.0 Scratchy 8FHTgA8MEu ...

This has download 5 similarly *Scratchy* glitches from the LIGO-Hanford
(`'H1'`) and LIGO-Livingston (`'L1'`) observatories.

*********************
Built-in file formats
*********************

GWpy extends the Astropy functionality with readers for the following file
formats:

- :ref:`gwpy-table-io-ligolw`
- :ref:`gwpy-table-io-ascii-cwb`
- :ref:`gwpy-table-io-root`
- :ref:`gwpy-table-io-pycbc_live`
- :ref:`gwpy-table-io-gstlal`
- :ref:`gwpy-table-io-gwf`

Each of the sub-sections below outlines how to read and write in these
file formats, include the custom keyword arguments to pass to
:meth:`EventTable.read` and :meth:`EventTable.write`.

.. admonition:: Listing all available formats

    To list all available formats, consult the documentation for
    :meth:`EventTable.read`.

============================
Multi-processed file reading
============================

The :meth:`EventTable.read` method accepts the ``nproc``
keyword argument, allowing multi-processed reading of lists of files.
This argument can be used with any file-format, not just those defined below,
but is **not** backported to for use with :meth:`Table.read`.


.. _gwpy-table-io-ligolw:

===============
``LIGO_LW`` XML
===============

**Additional dependencies:** :doc:`igwn-ligolw <igwn-ligolw:index>`

The LIGO Scientific Collaboration uses a custom scheme of XML in which to
store tabular data, called ``LIGO_LW``.
Complementing the scheme is a python library - :doc:`igwn-ligolw <igwn-ligolw:index>` -
which allows users to read and write all of the different types of tabular data
produced by gravitational-wave searches.

Reading and writing tables in ``LIGO_LW`` XML format is supported with
``format='ligolw', tablename=<tablename>'`` where ``<tablename>`` can be
any of the supported LSC table names (see :mod:`igwn_ligolw.lsctables`).

Reading
-------

When reading, the ``tablename`` keyword argument should be given to identify
the table in the file, as follows:

.. code-block:: python
    :name: gwpy-table-io-ligolw-tablename
    :caption: Reading an `EventTable` from ``LIGO_LW`` XML.

    >>> t = EventTable.read('H1-LDAS_STRAIN-968654552-10.xml.gz', tablename='sngl_burst')

The result should be something similar to this:

.. code-block:: python
    :name: gwpy-table-io-ligolw-output
    :caption: Result of reading an `EventTable` from ``LIGO_LW`` XML.

    >>> print(t)
    ifo peak_time peak_time_ns start_time ... confidence chisq chisq_dof bandwidth
    --- --------- ------------ ---------- ... ---------- ----- --------- ---------
     H1 968654557    783203126  968654557 ...  16.811825   0.0     512.0     256.0
     H1 968654557    781250001  968654557 ...  16.816761   0.0     512.0     256.0
     H1 968654557    779296876  968654557 ...  16.696106   0.0     512.0     256.0
     H1 968654557    777343751  968654557 ...  16.739489   0.0     512.0     256.0
     H1 968654557    775390626  968654557 ...  16.802326   0.0     512.0     256.0
     H1 968654557    773437501  968654557 ...   16.30731   0.0     512.0     256.0
     H1 968654557    771484376  968654557 ...  16.307253   0.0     512.0     256.0
     H1 968654557    769531251  968654557 ...   16.35647   0.0     512.0     256.0
     H1 968654557    767578126  968654557 ...  16.561176   0.0     512.0     256.0
     H1 968654557    765625001  968654557 ...  16.393112   0.0     512.0     256.0
     H1 968654557    763671876  968654557 ...  16.404041   0.0     512.0     256.0
     H1 968654557    761718751  968654557 ...  16.405825   0.0     512.0     256.0
     H1 968654557    759765626  968654557 ...  16.715092   0.0     512.0     256.0
     H1 968654557    757812501  968654557 ...  17.512749   0.0     512.0     256.0
     H1 968654557    755859376  968654557 ...  17.347675   0.0     512.0     256.0
     H1 968654557    753906251  968654557 ...  17.077478   0.0     512.0     256.0
     H1 968654557    751953126  968654557 ...  16.742907   0.0     512.0     256.0
    ...       ...          ...        ... ...        ...   ...       ...       ...
     H1 968654560    342773438  968654559 ...  11.029792   0.0      16.0       8.0
     H1 968654559    280273438  968654558 ...  12.363036   0.0      16.0       8.0
     H1 968654559    217773438  968654558 ...  13.985101   0.0      16.0       8.0
     H1 968654559    155273438  968654558 ...  14.662391   0.0      16.0       8.0
     H1 968654559     92773438  968654558 ...  15.864924   0.0      16.0       8.0
     H1 968654559     30273438  968654558 ...  16.321821   0.0      16.0       8.0
     H1 968654558    967773438  968654558 ...  16.975931   0.0      16.0       8.0
     H1 968654558    905273438  968654558 ...  19.160393   0.0      16.0       8.0
     H1 968654560    811523438  968654560 ...  11.270205   0.0       8.0       4.0
     H1 968654560    686523438  968654560 ...  15.839205   0.0       8.0       4.0
     H1 968654560    561523438  968654560 ...  15.944695   0.0       8.0       4.0
     H1 968654560    436523438  968654559 ...  14.384432   0.0       8.0       4.0
     H1 968654560    311523438  968654559 ...  14.465309   0.0       8.0       4.0
     H1 968654560    186523438  968654559 ...  13.045853   0.0       8.0       4.0
     H1 968654560    561523438  968654560 ...  11.636543   0.0       4.0       4.0
     H1 968654560    436523438  968654560 ...  15.344837   0.0       4.0       4.0
     H1 968654560    311523438  968654560 ...  11.367717   0.0       4.0       4.0
    Length = 2052 rows

.. admonition::  Single-table files

    The ``tablename`` keyword can be omitted if there is only a single table
    in the file.

To restrict the columns returned in the new `EventTable`, use the `columns`
keyword argument:

.. code-block:: python
    :name: gwpy-table-io-ligolw-columns
    :caption: Reading specific columns into an `EventTable` from ``LIGO_LW`` XML.

    >>> t = EventTable.read(
    ...     "H1-LDAS_STRAIN-968654552-10.xml.gz",
    ...     tablename="sngl_burst",
    ...     columns=["peak_time", "peak_time_ns", "snr", "peak_frequency"],
    ... )

Many LIGO_LW table objects (as defined in :mod:`igwn_ligolw.lsctables`) include
utility functions to create new columns by combining others,
e.g. to calculate the Q of a sine-Gaussian pulse from the duration and
central frequency.
These 'columns' can be requested directly, providing the
:class:`igwn_ligolw.ligolw.Table` representation of the data has a
:meth:`get_<name>` method for that name:

.. code-block:: python
    :name: gwpy-table-io-ligolw-get-columns
    :caption: Reading ``get_`` columns into an `EventTable` from ``LIGO_LW`` XML.

    >>> t = EventTable.read(
    ...     "H1-LDAS_STRAIN-968654552-10.xml.gz",
    ...     tablename="sngl_burst",
    ...     columns=["snr", "q", "duration", "central_freq"],
    ... )

.. note::

    When reading a processed column in this manner, all required input columns
    for a processed column must be included in the `columns` keyword list.
    To exclude these columns from the returned data, use the
    ``ligolw_columns=`` keyword to specify the columns required to provide
    the output columns:

    .. code-block:: python

        >>> t = EventTable.read(
        ...     "H1-LDAS_STRAIN-968654552-10.xml.gz",
        ...     tablename="sngl_burst",
        ...     columns=["snr", "q"],
        ...     ligolw_columns=["snr", "duration", "central_freq"],
        ... )

By default, the returned `EventTable` uses the dtypes returned by the
:mod:`ligol.lw` library, and functions therein, which often end up as
`numpy.object_` arrays in the table.
To force all columns to have real `numpy` data types, use the
``use_numpy_dtypes=True`` keyword, which will cast (known) custom object
types to a standard `numpy.dtype`, e.g:

.. code-block:: python
    :name: gwpy-table-io-ligolw-use_numpy_dtypes
    :caption: Example of using ``use_numpy_dtypes=True`` when reading an `EventTable` from ``LIGO_LW`` XML.

    >>> t = EventTable.read(
    ...     "H1-LDAS_STRAIN-968654552-10.xml.gz",
    ...     tablename="sngl_burst",
    ...     columns=["peak"],
    ...     ligolw_columns=["peak_time", "peak_time_ns"])
    >>> print(type(t[0]["peak"]))
    <type 'lal.LIGOTimeGPS'>
    >>> t = EventTable.read(
    ...     "H1-LDAS_STRAIN-968654552-10.xml.gz",
    ...     tablename="sngl_burst",
    ...     columns=["peak"],
    ...     ligolw_columns=["peak_time", "peak_time_ns"],
    ...     use_numpy_dtypes=True)
    >>> print(type(t[0]["peak"]))
    <type 'numpy.float64'>


Writing
-------

A table can be written as follows:

.. code-block:: python

    >>> t.write('new-table.xml', format='ligolw', tablename='sngl_burst')

Because ``LIGO_LW`` isn't the only scheme of XML, the ``format`` keyword is
required for all `Table.write()` operations.

If the target file already exists, an :class:`~exceptions.IOError` will be
raised, use ``overwrite=True`` to force a new file to be written.

To write a table to an existing file, use ``append=True``:

.. code-block:: python

    >>> t.write('new-table.xml', format='ligolw', tablename='sngl_burst', append=True)

To replace an existing table of the given type in an existing file, while
preserving other tables, use *both* ``append=True`` and ``overwrite=True``:

.. code-block:: python

    >>> t.write('new-table.xml', format='ligolw', tablename='sngl_burst', append=True, overwrite=True)

.. _gwpy-table-io-ascii-cwb:

============================================
Coherence WaveBurst ASCII (aka `EVENTS.txt`)
============================================

|cWBl|_ is an analysis pipeline is used to detect generic gravitational-wave
bursts, without using a signal model to restrict the analysis, and runs in
both low-latency (online) and offline modes over current GWO data.
The analysis uses the ROOT framework for most data products, but also
produces ASCII data in a custom format commonly written in a file called
``EVENTS.txt``.

Reading
-------

To read a cWB ASCII file:

.. code-block:: python
    :name: gwpy-table-io-cwb-read
    :caption: Reading an `EventTable` from cWB-format ASCII.

    >>> t = EventTable.read('EVENTS.txt', format='ascii.cwb')

See the :func:`astropy.io.ascii.read` documentation for full details on
keyword arguments when reading ``ascii.cwb`` files.

Writing
-------

To write a table using the cWB ASCII format:

.. code-block:: python
    :name: gwpy-table-io-cwb-write
    :caption: Writing an `EventTable` to a cWB-format ASCII file.

    >>> t.write('EVENTS.txt', format='ascii.cwb')

[the output file name is not required to be ``'EVENTS.txt'``, this is
simply the convention used in the cWB analysis.]

.. _gwpy-table-io-root:

====
ROOT
====

**Additional dependencies:** :doc:`uproot <uproot:index>`

Reading
-------

To read a `ROOT <https://root.cern.ch/>`_ tree into a `Table`
(or `EventTable`), specify the relevant tree via the ``treename``
keyword argument:

.. code-block:: python
    :name: gwpy-table-io-root-read
    :caption: Reading an `EventTable` from a ROOT file.

    >>> t = Table.read('my-data.root', treename='triggers')

If ``treename=None`` is given (default), a single tree will be read if only one exists in the file, otherwise a `ValueError` will be raised.

Any other keyword arguments will be passed directly to :meth:`uproot.tree.TTreeMethods.arrays`.

Writing
-------

To write a `Table` as a ROOT tree:

.. code-block:: python
    :name: gwpy-table-io-root-write
    :caption: Writing an `EventTable` to a ROOT file.

    >>> t.write('new-table.root')

As with reading, the ``treename`` keyword argument can be used to specify
the tree, the default is ``treename='tree'``.

By default, an existing file with an existing tree of the given name will be
appended to, to overwrite use the ``mode='recreate'`` keyword argument:

.. code-block:: python
    :name: gwpy-table-io-root-write-mode
    :caption: Writing an `EventTable` over an existing tree in a ROOT file.

    >>> t.write('new-table.root', treename='triggers', mode='recreate')

Any other keyword arguments will be passed directly to :class:`uproot.newtree`.

.. _gwpy-table-io-pycbc_live:

=================
PyCBC Live (HDF5)
=================

PyCBC Live is a low-latency search for gravitational waves from compact
binary coalescences, built from the :doc:`PyCBC <pycbc:index>` analysis package.
This search writes files on the LIGO Data Grid (LIGO.ORG-authenticated users
only) in HDF5 format, containing tables of events; each column in the table
is recorded as a separate HDF5 Dataset.

Identifying
-----------

PyCBC Live HDF5 files are identified automatically if the file is identified
as an HDF5 file (see :func:`astropy.io.misc.hdf5.is_hdf5`) **and** the
file _name_ matches the following regular expression:

.. code-block:: text

    ([A-Z][0-9])+-Live-[0-9.]+-[0-9.]+.(h5|hdf|hdf5)

e.g.

.. code-block:: text

    H1-Live-1234567890-4.h5

If the format cannot be auto-identifed (because the filename doesn't match the
above regular expression), the format can be specified via

.. code-block:: python

    format="hdf5.pycbc_live"

Reading
-------

To read an `EventTable` from a ``pycbc_live`` format HDF5 file:

.. code-block:: python
    :name: gwpy-table-io-pycbc_live-read
    :caption: Reading an `EventTable` from a PyCBC-Live HDF5 file.

    >>> t = EventTable.read("H1-Live-1234567890-4.h5")

To restrict the returned columns, use the ``columns`` keyword argument:

.. code-block:: python
    :name: gwpy-table-io-pycbc_live-read-columns
    :caption: Reading specific columns into an `EventTable` from PyCBC-Live HDF5.

    >>> t = EventTable.read(
    ...     "H1-MY_DATA-1234567890-4.h5",
    ...     format="hdf5.pycbc_live",
    ...     columns=["end_time", "snr", "chisq"],
    ... )

Similarly to the :ref:`gwpy-table-io-ligolw` format, some processed columns
can be specified that are not included in the HDF5 files, but are created
on-the-fly.
Supported processed columns are:

- ``mchirp``
- ``new_snr``

These can be specified without having to specify any of the input columns.

Additionally, PyCBC HDF5 table Groups include extra datasets that aren't
part of the table, e.g. ``'psd'``.
These can be included in the returned `EventTable.meta` `dict` via the
keyword ``extended_metadata=True`` (default), or excluded with
``extended_metadata=False``.

Writing
-------

Writing tables in PyCBC Live HDF5 format is not supported at this time.

.. _gwpy-table-io-gstlal:

========================
GstLAL (``LIGO_LW`` XML)
========================

|GstLAL|_ is a low-latency search for gravitational waves from compact
binary coalescences.
This search writes files on the LIGO Data Grid (LIGO.ORG-authenticated users
only) in ``LIGO_LW`` XML format, containing tables of events.

Reading
-------

To read an `EventTable` from a ``gstlal`` format ``LIGO_LW`` XML file, use the
``format='ligolw.gstlal'`` keyword:

.. code-block:: python
    :name: gwpy-table-io-gstlal-read
    :caption: Reading an `EventTable` from a GstLAL ``LIGO_LW`` XML file.

    >>> t = EventTable.read("H1L1-GstLAL-1234567890-4.xml.gz", format="ligolw.gstlal")

GstLAL ``LIGO_LW`` XML files contain information about triggers from each detector separately
as well as from a combination of detectors.
Accessing these different sets of information can be done using the ``triggers`` keyword.
By default, information about triggers from each detector separately is read in.
This is equivalent to using ``triggers='sngl'``.
To instead read information about triggers from multiple detectors, you can instead use
the ``triggers='coinc'`` keyword:

.. code-block:: python
    :name: gwpy-table-io-gstlal-read-coinc-table
    :caption: Reading coincident triggers into an `EventTable` GstLAL ``LIGO_LW`` XML HDF5.

    >>> t = EventTable.read(
    ...     "H1L1-GstLAL-1234567890-4.xml.gz",
    ...     format="gstlal.gstlal",
    ...     triggers='coinc',
    ... )

To restrict the returned columns, use the ``columns`` keyword argument:

.. code-block:: python
    :name: gwpy-table-io-gstlal-read-columns
    :caption: Reading specific columns into an `EventTable` GstLAL ``LIGO_LW`` XML HDF5.

    >>> t = EventTable.read(
    ...     "H1L1-GstLAL-1234567890-4.xml.gz",
    ...     format="gstlal.gstlal",
    ...     columns=["end_time", "snr", "chisq"],
    ... )

Similarly to the :ref:`gwpy-table-io-ligolw` format, some processed columns
can be specified that are not included in the XML files, but are created
on-the-fly.
In addition to processed columns support by :ref:`gwpy-table-io-ligolw`,
the additional supported processed columns are:

- ``mchirp``
- ``snr_chi``
- ``chi_snr``

These can be specified without having to specify any of the input columns.

Writing
-------

Writing tables in GstLAL ``LIGO_LW`` XML format is not supported at this time.

.. _gwpy-table-io-snax:

===========
SNAX (HDF5)
===========

The SNAX (Signal-based Noise Acquisition and eXtraction) analysis pipeline
is a low-latency search for identifying glitches in h(t) and auxiliary
channel data using glitch waveforms, operating in low-latency (online)
and offline modes.
This search writes files on the LIGO Data Grid (LIGO.ORG-authenticated users
only) in HDF5 format containing regularly-sampled features; each channel in
the table is recorded as a separate HDF5 Dataset.

Reading
-------

To read an `EventTable` from a ``snax`` format HDF5 file,
use the ``format='hdf5.snax'`` keyword:

.. code-block:: python
    :name: gwpy-table-io-snax-read
    :caption: Reading an `EventTable` from a SNAX-format HDF5 file.

    >>> t = EventTable.read("H-SNAX_FEATURES-1255853400-20.h5", format="hdf5.snax")

By default, all channels contained in the file are read in.
To restrict the returned channels, use the ``channels`` keyword argument:

.. code-block:: python
    :name: gwpy-table-io-snax-read-channels
    :caption: Reading specific channels into an `EventTable` from a SNAX-format HDF5 file.

    >>> t = EventTable.read(
    ...     "H-SNAX_FEATURES-1255853400-20.h5",
    ...     format="hdf5.snax",
    ...     channels="H1:CAL-DELTAL_EXTERNAL_DQ",
    ... )

To restrict the returned columns, use the ``columns`` keyword argument:

.. code-block:: python
    :name: gwpy-table-io-snax-read-columns
    :caption: Reading specific columns into an `EventTable` from SNAX-format HDF5.

    >>> t = EventTable.read(
    ...     "H-GSTLAL_IDQ_FEATURES-1255853400-20.h5",
    ...     format="hdf5.snax",
    ...     columns=["channel", "time", "snr"],
    ... )

Writing
-------

Writing tables in SNAX HDF5 format is not supported at this time.

.. _gwpy-table-io-gwf:

===
GWF
===

**Additional dependencies:** :ref:`gwpy-external-framecpp`

The Gravitational-Wave Frame file format supports tabular data via
``FrEvent`` structures, which allow storage of arbitrary event information.

Reading
-------

To read an `EventTable` from a ``GWF``-format file, specify the filename and
the ``name`` of the ``FrEvent`` structures to read:

.. code-block:: python
    :name: gwpy-table-io-gwf-read
    :caption: Reading an `EventTable` from GWF.

    >>> t = EventTable.read('events.gwf', 'my-event-name')

To restrict the returned columns, use the ``columns`` keyword argument:

.. code-block:: python
    :name: gwpy-table-io-gwf-read-columns
    :caption: Reading specific columns into an `EventTable` from GWF.

    >>> t = EventTable.read('events.gwf', 'my-event-name', columns=['time', 'amplitude'])

All ``FrEvent`` structures contain the following columns, any other
columns are use-specific:

===============  ====================================================================================
Column name      Description (from :dcc:`LIGO-T970130`)
===============  ====================================================================================
``time``         Reference time of event, as defined by the search algorithm
``amplitude``    Continuous output amplitude returned by event
``probability``  Likelihood estimate of event, if available (probability = -1 if cannot be estimated)
``timeBefore``   Signal duration before reference time (seconds)
``timeAfter``    Signal duration after reference time (seconds)
``comment``      Descriptor of event
===============  ====================================================================================

Writing
-------

Writing tables in GWF format is not supported at this time.
