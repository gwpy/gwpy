Customizing time series plots
=============================

The time series plots label the axes X and Y.  To adjust the limits use the ``--xmin``, ``xmax``,
``ymin``, ``ymax``.  The default is to set the axes limits to show all the data.  A good
reason to adjust the y-axis limits is to compare different plots.  The x-axis is generally controlled
by the amount of data transferred but the data transfer is only controllable to integer second resolution.
The x-axis limits may be set in fractional seconds of arbitrary resolution.  The x limits are
specified in GPS seconds, the same as start time.

To produce a log-log or semi-log plot use the ``logx`` and/or ``logy`` arguments.

The following image is a semi-log plot of the pressure in HVEA during a pump down procedure.
It is created with the following command line

.. code-block:: sh

    gwpy-ldvw.py timeseries ``FILL THIS IN JOE!``


.. image:: /../../cli_examples/cli-03.png
    :align: center
    :alt: HVEA pressure during pump down

The next example is a sub-second plot of a major carry glitch.
It is created with the following command line:

.. code-block:: sh

    gwpy-ldvw.py timeseries ``FILL THIS IN JOE!``


.. image:: /../../cli_examples/cli-04.png
    :align: center
    :alt: sample spectrum
