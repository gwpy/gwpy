##################
GWpy Documentation
##################

GWpy is a collaboration-driven package containing a set of common tools for characterising and analysing data from current gravitational wave detectors and studying the astrophysics associated with gravitational wave emission.

.. image:: examples/hoft.png
   :width: 40%

.. image:: examples/specgram.png
   :width: 59%

******************
User documentation
******************

.. toctree::
   :maxdepth: 2

   overview
   install
   getting_started
   detector/index
   data/index
   segments/index
   plotter/index

***************
Worked examples
***************

GWpy is package with a set of worked examples, providing step-by-step tutorials on how to use the GWpy package to study data from gravitational-wave detectors.

The examples are provided in the ``examples/`` directory of the git repository, as ``.py`` files, runnable from your console, e.g.:

.. code:: bash

   python gw_ex_plot_timeseries.py

**List of examples:**

.. toctree::
   :maxdepth: 1

   examples/gw_ex_plot_timeseries
   examples/gw_ex_compare_spectra
   examples/gw_ex_plot_spectrogram
