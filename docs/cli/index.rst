*******************************
Command line plotting with GWpy
*******************************

The ``gwpy-plot`` command-line script provides a terminal-based user interface
to querying data and generating images.
Functionality for this tool is primarily inspired by LigoDV-web
(LDVW, https://ldvw.ligo.caltech.edu), a web tool for viewing LIGO data,
available to members of the joint LIGO-Virgo collaboration.
LDVW, written in Java, uses the ``gwpy-plot`` command-line script provided by
GWpy to generate the plots based on web-form input from the user.

The basic usage for `gwpy-plot` is as follows:

.. code-block:: bash

    gwpy-plot <ACTION> --chan <channel-name> --start <gps-start-time> [OPTIONS]

Where ``<ACTION>`` is the type of plot to make.

You can run ``gwpy-plot --help`` to list the plot types ('actions') you can use:

.. command-output:: gwpy-plot --help

To see the arguments and options for a given action, you can run, for example,
``gwpy-plot timeseries --help``:

.. command-output:: gwpy-plot timeseries --help

The options for each action vary but many are common.

=====================
Specifying input data
=====================

.. currentmodule:: gwpy.timeseries

The following table summarises the allowed number of inputs for each action

+-------------------+--------------------+-----------------+
| Action            | Number of channels | Number of times |
+===================+====================+=================+
| ``timeseries``    | >=1                | >=1             |
+-------------------+--------------------+-----------------+
| ``spectrum``      | >=1                | >=1             |
+-------------------+--------------------+-----------------+
| ``coherence``     | >=2                | >=1             |
+-------------------+--------------------+-----------------+
| ``spectrogram``   | 1                  | 1               |
+-------------------+--------------------+-----------------+
| ``coherencegram`` | 2                  | 1               |
+-------------------+--------------------+-----------------+
| ``qtransform``    | 1                  | 1               |
+-------------------+--------------------+-----------------+

By default all data are retrieved using :meth:`TimeSeriesDict.get`, which uses |nds2|_ for data access, but the ``--framcache``
option allows you to pass your own data via a LAL-format cache file.

=================
Interactive mode
=================

The ``--interactive`` argument uses :mod:`~matplotlib.pyplot` to display
the image and allow interactive manipulations, (zoom, pan, etc).

========
Examples
========

.. examples.rst index is dynamically generated
.. include:: examples/examples.rst

.. add a hidden toctree to stop sphinx complaining
.. that the examples RST files aren't included in a toctree
.. when they clearly are
.. toctree::
   :hidden:
   :glob:

   examples/*
