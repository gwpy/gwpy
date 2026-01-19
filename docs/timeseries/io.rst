:tocdepth: 3

.. currentmodule:: gwpy.timeseries

.. _gwpy-timeseries-io:

####################################
Reading and writing time series data
####################################

The `TimeSeries` object includes :meth:`~TimeSeries.read` and
:meth:`~TimeSeries.write` methods to enable reading from and writing to files
respectively.
For example, to read from an ASCII file containing time and amplitude columns:

.. code-block:: python
    :name: gwpy-timeseries-io-example

    >>> from gwpy.timeseries import TimeSeries
    >>> data = TimeSeries.read('my-data.txt')

:meth:`TimeSeries.read` will attempt to automatically identify the file
format based on the file extension and/or the contents of the file, however,
the ``format`` keyword argument can be used to manually identify the input
file-format.

The :meth:`~TimeSeries.read` and :meth:`~TimeSeries.write` methods take
different arguments and keywords based on the input/output file format,
see :ref:`gwpy-timeseries-io-formats` for details on reading/writing for
each of the built-in formats.

.. _gwpy-timeseries-io-remote:

*******************
Reading remote data
*******************

`TimeSeries.read` supports reading data directly from remote sources,
where the input filename is a ``http://``, ``https://``, or ``ftp://`` URL,
as supported by Astropy's :doc:`astropy:utils/data`.

Additionally, GWpy supports reading data directly from Pelican or OSDF
URLs, where the input filename is a ``pelican://`` or ``osdf://`` URL,
if the additional :doc:`requests-pelican <requests-pelican:index>`
package is installed.

.. code-block:: python
    :caption: Reading data from Pelican

    >>> from gwpy.timeseries import TimeSeries
    >>> data = TimeSeries.read(
    ...     "osdf:///gwdata/O3b/strain.4k/hdf.v1/H1/1268776960/"
    ...     "H-H1_GWOSC_O3b_4KHZ_R1-1269358592-4096.hdf5",
    ...     format="hdf5.gwosc",
    ... )
    >>> print(data)
    TimeSeries([2.69325914e-20, 2.98416387e-20, 2.56382655e-20, ...,
                3.49428204e-20, 3.19402495e-20, 2.73257992e-20],
               unit: dimensionless,
               t0: 1269358592.0 s,
               dt: 0.000244140625 s,
               name: H1:GWOSC-4KHZ_R1_STRAIN,
               channel: None)


.. admonition:: Tip: Installing GWpy with Pelican support
    :class: tip

    To install GWpy with Pelican support, use:

    .. code-block:: console

        pip install gwpy[pelican]

******************************
Automatic discovery of GW data
******************************

For full details on automatic data discovery,
including data from GWOSC or GWDataFind,
see :ref:`gwpy-timeseries-get`.

.. _gwpy-timeseries-io-formats:

*********************
Built-in file formats
*********************

.. _gwpy-timeseries-io-ascii:

=====
ASCII
=====

GWpy supports writing `TimeSeries` (and `~gwpy.frequencyseries.FrequencySeries`) data to ASCII in a two-column ``time`` and ``amplitude`` format.

Reading
-------

To read a `TimeSeries` from ASCII:

.. code-block:: python

    >>> t = TimeSeries.read('data.txt')

See :func:`numpy.loadtxt` for keyword argument options.

Writing
-------

To write a `TimeSeries` to ASCII:

.. code-block:: python

    >>> t.write('data.txt')

See :func:`numpy.savetxt` for keyword argument options.

.. _gwpy-timeseries-io-gwf:

===
GWF
===

**Additional dependencies:**
:ref:`gwpy-external-framecpp`
or :ref:`gwpy-external-framel`
or :ref:`gwpy-external-lalframe`

The raw observatory data are archived in ``.gwf`` files, a custom binary
format that efficiently stores the time streams and all necessary metadata,
for more details about this particular data format,
take a look at the specification document :dcc:`LIGO-T970130`.

GWF library availability
------------------------

GWpy can use any of the three named GWF input/output libraries, and will try
to find them in the order they are listed
(FrameCPP first, then FrameL, then LALFrame).
If you need to read/write GWF files, any of them will work, but re recommend
to try and install the libraries in that order; FrameCPP provides a more
complete Python API than the others.

However, not all libraries may be available on all platforms, the linked pages
for each library include an up-to-date listing of the supported platforms.

Reading
-------

To read data from a GWF file, pass the input file path (or paths) and the name of the data channel to read:

.. code-block:: python

    >>> data = TimeSeries.read('HLV-HW100916-968654552-1.gwf', 'L1:LDAS-STRAIN')

.. note::

    The ``HLV-HW100916-968654552-1.gwf`` file is included with the GWpy source under `/gwpy/testing/data/ <https://gitlab.com/gwpy/gwpy/-/tree/main/gwpy/testing/data>`_.

Reading a `StateVector` uses the same syntax:

.. code-block:: python

    >>> data = StateVector.read('my-state-data.gwf', 'L1:GWO-STATE_VECTOR')

For instance, to read injections flags from a GWOSC GWF file,
you can use the `LOSC-INJMASK` channel:

.. code-block:: python

    >>> injections = StateVector.read('my-state-data.gwf', 'L1:LOSC-INJMASK')

Multiple files can be read by passing a list of files:

.. code-block:: python

    >>> data = TimeSeries.read([file1, file2], 'L1:LDAS-STRAIN')

When reading multiple files, the ``nproc`` keyword argument can be used to distribute the reading over multiple CPUs, which should make it faster:

.. code-block:: python

    >>> data = TimeSeries.read([file1, file2, file3, file4], 'L1:LDAS-STRAIN', nproc=2)

The above command will separate the input list of 4 file paths into two sets of 2 files, combining the results into a single `TimeSeries` before returning.

The ``start`` and ``end`` keyword arguments can be used to downselect data to a specific ``[start, end)`` time segment when reading:

.. code-block:: python

    >>> data = TimeSeries.read('HLV-HW100916-968654552-1.gwf', 'L1:LDAS-STRAIN', start=968654552.5, end=968654553)

Additionally, the following keyword arguments can be used:

.. warning::

    These keyword arguments are only supported when using the
    |LDASTools.frameCPP|_ GWF API.

.. table:: Keyword arguments for `TimeSeries.read`
    :align: left
    :name: timeseries-read-kwargs

    ============  =======  =======  ==========================================
    Keyword       Type     Default  Usage
    ============  =======  =======  ==========================================
    ``scaled``    `bool`   `True`   Apply ADC calibration when reading
    ``type``      `str`    `None`   `dict` of channel types
                                    (``'ADC'``, ``'Proc'``, or ``'Sim'``) for
                                    each channel to be read. This option
                                    optimises the reading operation.
    ============  =======  =======  ==========================================

Reading multiple channels
-------------------------

To read multiple channels from one or more GWF files (rather than opening and closing the files multiple times), use the `TimeSeriesDict` or `StateVectorDict` classes, and pass a list of data channel names:

.. code-block:: python

    >>> data = TimeSeriesDict.read('HLV-HW100916-968654552-1.gwf', ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN'])

.. note::

    A mix of `TimeSeries` and `StateVector` objects can be read by using only `TimeSeriesDict` class, and casting the returned data to a `StateVector` using :meth:`~TimeSeries.view`.

Writing
-------

To write data held in any of the :mod:`gwpy.timeseries` classes to a GWF file, simply use:

.. code-block:: python

    >>> data.write('output.gwf')

**If the output file already exists it will be overwritten,** use ``overwrite=False`` to
prevent this (an `OSError` will be raised).

.. note::

    When writing a timeseries to a GWF, the :attr:`TimeSeries.name`
    property is used for the `name` variable of the GWF data structures
    (`FrProcData` and `FrVect`).
    So, if you want to write a file and then read it back in, you must ensure
    that the :attr:`~TimeSeries.name` property is correctly assigned, e.g:

    .. code-block:: python

        >>> channel = "L1:CHANNEL_NAME"
        >>> output_file = "output.gwf"
        >>> data = TimeSeries([1, 2, 3])
        >>> data.name = channel
        >>> data.write(output_file)
        >>> data = TimeSeries.read(output_file, channel)


.. _gwpy-timeseries-io-hdf5:

====
HDF5
====

GWpy allows storing data in HDF5 format files, using a custom specification for storage of metadata.

.. warning::

    To read GWOSC data from HDF5, please see
    :ref:`gwpy-timeseries-io-hdf5-gwosc`.

Reading
-------

To read `TimeSeries` or `StateVector` data held in HDF5 files pass the filename (or filenames) or the source, and the path of the data inside the HDF5 file:

.. code-block:: python

    >>> data = TimeSeries.read('HLV-HW100916-968654552-1.hdf', 'L1:LDAS-STRAIN')

As with GWF, the ``start`` and ``end`` keyword arguments can be used to downselect data to a specific ``[start, end)`` time segment when reading:

.. code-block:: python

    >>> data = TimeSeries.read('HLV-HW100916-968654552-1.hdf', 'L1:LDAS-STRAIN', start=968654552.5, end=968654553)

Analogously to GWF, you can read multiple `TimeSeries` from an HDF5 file via :meth:`TimeSeriesDict.read`:

.. code-block:: python

    >>> data = TimeSeriesDict.read('HLV-HW100916-968654552-1.hdf')

By default, all matching datasets in the file will be read, to restrict the output, specify the names of the datasets you want:

.. code-block:: python

    >>> data = TimeSeriesDict.read('HLV-HW100916-968654552-1.hdf', ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN'])


Writing
-------

Data held in a `TimeSeries`, `TimeSeriesDict, `StateVector`, or `StateVectorDict` can be written to an HDF5 file via:

.. code-block:: python

    >>> data.write('output.hdf')

The output argument (``'output.hdf'``) can be a file path, an open `h5py.File` object, or a `h5py.Group` object, to append data to an existing file.

If the target file already exists, an :class:`~exceptions.IOError` will be raised, use ``overwrite=True`` to force a new file to be written.

To write a `TimeSeries` to an existing file, use ``append=True``:

.. code-block:: python

    >>> data.write('output.hdf', append=True)

To replace an existing dataset in an existing file, while preserving other data, use *both* ``append=True`` and ``overwrite=True``:

.. code-block:: python

    >>> data.write('output.hdf', append=True, overwrite=True)

.. _gwpy-timeseries-io-hdf5-gwosc:

============
HDF5 (GWOSC)
============

|GWOSC|_ write data in HDF5 using a custom schema that is incompatible
with `format='hdf5'`.

Reading
-------

GWpy can read data from GWOSC HDF5 files using the `format='hdf5.gwosc'`
keyword:

.. code-block:: python

    >>> data = TimeSeries.read(
    ...     "H-H1_GWOSC_16KHZ_R1-1187056280-4096.hdf5",
    ...     format="hdf5.gwosc",
    ... )

By default, :meth:`TimeSeries.read` will return the contents of the
``/strain/Strain`` dataset, while :meth:`StateVector.read` will return those
of ``/quality/simple``.

It's possible to change which datasets are read with the `path`, `value_dataset`
and `bits_dataset` keywords.
For instance to read injections flags:

.. code-block:: python

    >>> injections = StateVector.read(
    ...     "H-H1_GWOSC_16KHZ_R1-1187056280-4096.hdf5",
    ...     format="hdf5.gwosc",
    ...     path="quality/injections",
    ...     value_dataset="Injmask",
    ...     bits_dataset="InjDescriptions",
    ... )

As with regular HDF5, the ``start`` and ``end`` keyword arguments can be used
to downselect data to a specific ``[start, end)`` time segment when reading.


.. _gwpy-timeseries-io-wav:

===
WAV
===

Any `TimeSeries` can be written to / read from a WAV file using :meth:`TimeSeries.read`:

.. warning::

    No metadata are stored in the WAV file except the sampling rate, so any units or GPS timing information are lost when converting to/from WAV.

Reading
-------

To read a `TimeSeries` from WAV:

.. code-block:: python

    >>> t = TimeSeries.read('data.wav')

See :func:`scipy.io.wavfile.read` for any keyword argument options.

Writing
-------

To write a `TimeSeries` to WAV:

.. code-block:: python

    >>> t.write('data.wav')

See :func:`scipy.io.wavfile.write` for keyword argument options.
