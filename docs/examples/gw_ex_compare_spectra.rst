=========================================================
GWpy.Ex: comparing a Channel's spectrum between two times
=========================================================

.. currentmodule:: gwpy.spectrum.core
.. |ex.py| replace:: ../../examples/gw_ex_compare_spectra.py

Problem
-------

I'm interested in comparing the amplitude spectrum of a channel between a
known 'good' time - where the spectrum is what we expect it to be - and a
known 'bad' time - where some excess noise appeared and the spectrum
changed appreciably.

Solution
--------

First, we define what the 'good' and 'bad' times are:

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 15,23-24

and how long we want to search each one

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 25

Then we can fetch the :class:`~gwpy.timeseries.core.TimeSeries` data for both times for the in-loop photodiode signal for the intensity stabilisation servo of the pre-stabilised laser:

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 28-31

We can now calculate the amplitude spectral density for each time,

.. literalinclude:: ../../examples/gw_ex_compare_spectra.py
   :lines: 34-37

and make a comparison plot:

.. plot:: ../examples/gw_ex_compare_spectra.py

The extra peak at 620 Hz is worrying, so we can zoom in around that frequency range to see what's going on:

.. code:: python

   plot.logx = False
   plot.xlim = [600, 640]
   plot.refresh()

.. plot::

   execfile('../examples/gw_ex_compare_spectra.py')
   plot.logx = False
   plot.xlim = [600, 640]
   plot.ylim = [1e-6, 5e-4]
   plot.refresh()

That needs investigating, better call it in!
