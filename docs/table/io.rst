.. include:: ../references.txt
.. currentmodule:: gwpy.table

.. _gwpy-table-io:

####################################################
Reading and writing `Table` and `EventTable` objects
####################################################

.. note::

   This document complements the upstream Astropy documentation on reading/writing :class:`~astropy.table.Table` objects, please refer to http://docs.astropy.org/en/stable/table/io.html.

Astropy provides an excellent unified input/output system for the
`~astropy.table.Table` object, and GWpy extends upon that to include common
gravitational-wave file types, as well as providing event-specific
input/output registrations for event data.
In the most general case you can read a table of data as follows::

    >>> from gwpy.table import Table
    >>> table = Table.read('mydata.txt')

GWpy extends the Astropy functionality with readers for the following file formats:

- :ref:`gwpy-table-io-ligolw`
- :ref:`gwpy-table-io-ascii-cwb`
- :ref:`gwpy-table-io-root`
- :ref:`gwpy-table-io-pycbc_live`

Each of the sub-sections below outlines how to read and write in these file formats, include the custom keyword arguments to pass to :meth:`EventTable.read` and :meth:`EventTable.write`.


.. _gwpy-table-io-ligolw:

===============
``LIGO_LW`` XML
===============

**Additional dependencies:** |glue.ligolw|_

The LIGO Scientific Collaboration uses a custom scheme of XML in which to
store tabular data, called the ``LIGO_LW`` scheme.
Complementing the scheme is a python library - :mod:`glue.ligolw` - which
allows users to read and write all of the different types of tabular data
produced by gravitational-wave searches.

Reading and writing tables in ``LIGO_LW`` XML format is supported with ``format='ligolw.<tablename>'`` where ``<tablename>`` can be any of the supported LSC table names (see below for a full list).

Reading
-------

When reading, the `format` keyword argument must be given, to identify the table in the file, as follows::

    >>> t = EventTable.read('H1-LDAS_STRAIN-968654552-10.xml.gz', format='ligolw.sngl_burst')

The result should be something similar to this::

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

To restrict the columns returned in the new `EventTable`, use the `columns` keyword argument::

    >>> t = EventTable.read('H1-LDAS_STRAIN-968654552-10.xml.gz', format='ligolw.sngl_burst', columns=['peak_time', 'peak_time_ns', 'snr', 'peak_frequency'])

Many LIGO_LW table objects (as defined in :mod:`glue.ligolw.lsctables`) include utility functions to create new columns by combining others, e.g. to calculate the Q of a sine-Gaussian pulse from the duration and central frequency. To have the returned `EventTable` include these processed columns, use the ``get_as_columns`` keyword argument, which will call each discovered :meth:`get_xxx` method into a column called ``xxx``::

   >>> t = EventTable.read('H1-LDAS_STRAIN-968654552-10.xml.gz', format='ligolw.sngl_burst', columns=['snr', 'q', 'duration', 'central_freq'], get_as_columns=True)

.. note::

   When using `get_as_columns=True`, all required input columns for a processed column must be included in the `columns` keyword list

Writing
-------

A table can be written as follows::

    >>> t.write('new-table.xml', format='ligolw.sngl_burst')

If the target file already exists, an :class:`~exceptions.IOError` will be raised, use ``overwrite=True`` to force a new file to be written.

To write a table to an existing file, use ``append=True``::

    >>> t.write('new-table.xml', format='ligolw.sngl_burst', append=True)

To replace an existing table of the given type in an existing file, while preserving other tables, use *both* ``append=True`` and ``overwrite=True``::

    >>> t.write('new-table.xml', format='ligolw.sngl_burst', append=True, overwrite=True)


Supported LIGO_LW tables
------------------------

========================  =================================
Table name                Format name
========================  =================================
`coinc_definer`           ``ligolw.coinc_definer``
`coinc_event`             ``ligolw.coinc_event``
`coinc_event_map`         ``ligolw.coinc_event_map``
`coinc_inspiral`          ``ligolw.coinc_inspiral``
`coinc_ringdown`          ``ligolw.coinc_ringdown``
`dq_list`                 ``ligolw.dq_list``
`experiment`              ``ligolw.experiment``
`experiment_map`          ``ligolw.experiment_map``
`experiment_summary`      ``ligolw.experiment_summary``
`external_trigger`        ``ligolw.external_trigger``
`filter`                  ``ligolw.filter``
`gds_trigger`             ``ligolw.gds_trigger``
`lfn`                     ``ligolw.lfn``
`ligolw_mon`              ``ligolw.ligolw_mon``
`multi_burst`             ``ligolw.multi_burst``
`multi_inspiral`          ``ligolw.multi_inspiral``
`process`                 ``ligolw.process``
`process_params`          ``ligolw.process_params``
`search_summary`          ``ligolw.search_summary``
`search_summvars`         ``ligolw.search_summvars``
`segment`                 ``ligolw.segment``
`segment_definer`         ``ligolw.segment_definer``
`segment_summary`         ``ligolw.segment_summary``
`sim_burst`               ``ligolw.sim_burst``
`sim_inspiral`            ``ligolw.sim_inspiral``
`sim_inst_params`         ``ligolw.sim_inst_params``
`sim_ringdown`            ``ligolw.sim_ringdown``
`sngl_burst`              ``ligolw.sngl_burst``
`sngl_inspiral`           ``ligolw.sngl_inspiral``
`sngl_ringdown`           ``ligolw.sngl_ringdown``
`stochastic`              ``ligolw.stochastic``
`stochsumm`               ``ligolw.stochsumm``
`summ_mime`               ``ligolw.summ_mime``
`summ_value`              ``ligolw.summ_value``
`time_slide`              ``ligolw.time_slide``
`time_slide_segment_map`  ``ligolw.time_slide_segment_map``
`veto_definer`            ``ligolw.veto_definer``
========================  =================================


.. _gwpy-table-io-ascii-cwb:

============================================
Coherence WaveBurst ASCII (aka `EVENTS.txt`)
============================================

The `Coherent WaveBurst <http://dx.doi.org/10.1088/0264-9381/25/11/114029>`_ analysis pipeline is used to detect generic gravitational-wave bursts, without using a signal model to restrict the analysis, and runs in both low-latency (online) and offline modes over current GWO data.
The analysis uses the ROOT framework for most data products, but also produces ASCII data in a custom format commonly written in a file called ``EVENTS.txt``.

Reading
-------

To read a cWB ASCII file::

   >>> t = EventTable.read('EVENTS.txt', format='ascii.cwb')

See the :func:`astropy.io.ascii.read` documentation for full details on keyword arguments when reading ``ascii.cwb`` files.

Writing
-------

To write a table using the cWB ASCII format:

   >>> t.write('EVENTS.txt', format='ascii.cwb')

[the output file name is not required to be ``'EVENTS.txt'``, this is simply the convention used in the cWB analysis.]

.. _gwpy-table-io-root:

====
ROOT
====

**Additional dependencies:** |root_numpy|_

Reading
-------

To read a `ROOT <https://root.cern.ch/>`_ tree into a `Table` (or `EventTable`), specify the relevant tree via the ``treename`` keyword argument::

    >>> t = Table.read('my-data.root', treename='triggers')

If ``treename=None`` is given (default), a single tree will be read if only one exists in the file, otherwise a `ValueError` will be raised.

To specify the branches to read, use the ``branches`` keyword argument::

    >>> t = Table.read('my-data.root', treename='triggers', branches=['time', 'frequency', 'snr'])

Any other keyword arguments will be passed directly to :func:`root_numpy.root2array`.

Writing
-------

To write a `Table` as a ROOT tree::

    >>> t.write('new-table.root')

As with reading, the ``treename`` keyword argument can be used to specify the tree, the default is ``treename='tree'``.

By default, an existing file with an existing tree of the given name will be appended to, to overwrite use the ``mode='recreate'`` keyword argument::

    >>> t.write('new-table.root', treename='triggers', mode='recreate')

Any other keyword arguments will be passed directly to :func:`root_numpy.array2root`.

.. _gwpy-table-io-pycbc_live:

=================
PyCBC Live (HDF5)
=================

**Additional dependencies:** |h5py|_

PyCBC Live is a low-latency search for gravitational waves from compact binary coalescences, built from the `PyCBC <https://github.com/ligo-cbc/pycbc/>`_ analysis package.
This search writes files on the LIGO Data Grid (LIGO.ORG-authenticated users only) in HDF5 format, containing tables of events; each column in the table is recorded as a separate HDF5 Dataset.

Reading
-------

To read an `EventTable` from a ``pycbc_live`` format HDF5 file, use the ``format='hdf5.pycbc_live'`` keyword::

   >>> t = EventTable.read('H1-Live-1234567890-4.hdf', format='hdf5.pycbc_live')

To restrict the returned columns, use the ``columns`` keyword argument::

   >>> t = EventTable.read('H1-Live-1234567890-4.hdf', format='hdf5.pycbc_live', columns=['end_time', 'snr', 'chisq'])

Similarly to the :ref:`gwpy-table-io-ligolw` format, some processed columns can be specified that are not included in the HDF5 files, but are created on-the-fly.
Supported processed columns are:

- ``mchirp``
- ``new_snr``

These can be specified without having to specify any of the input columns.

Writing
-------

Writing tables in PyCBC Live HDF5 format is not supported at this time.

======================
Available file formats
======================

For a full list of available file formats, see the documentation for the `Table.read` method:

.. automethod:: Table.read

The `EventTable.read` method can understand *all* of the above file formats (auto-identify is **not** inherited), and the following::

.. automethod:: EventTable.read
