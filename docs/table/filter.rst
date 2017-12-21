.. currentmodule:: gwpy.table

.. _gwpy-table-filter:

################
Filtering tables
################

In order to perform detailed analysis of tabular data, it is useful to
extract portions of a table based on some criteria, this is called filtering.
The `EventTable` object comes with a :meth:`~EventTable.filter` method
that provides an intuitive interface to down-selecting rows in a table.

To demonstrate, first we create a catalogue of gravitational-wave detections using data available from `LOSC <https://losc.ligo.org/>`_::

    >>> from gwpy.table import EventTable
    >>> table = EventTable.read(
    ...     """name      gps           m1      m2         snr   distance network
    ...        GW150914  1126259462.00 36.2    23.7       23.7  420      HL
    ...        LVT151012 1128678900.44 23      13         9.7   440      HL
    ...        GW151226  1135136350.65 14.2    7.5        13    880      HL
    ...        GW170104  1167559936.60 31.2    19.4       13    1000     HL
    ...        GW170814  1186741861.53 30.5    25.3       18    540      HLV""",
    ...     format='ascii')

==============
Simple filters
==============

We can then filter the table based on ``snr``  to get the really loud events::

    >>> print(table.filter('snr > 15'))
      name        gps       m1   m2  snr
    -------- ------------- ---- ---- ----
    GW150914  1126259462.0 36.2 23.7 23.7
    GW170814 1186741861.53 30.5 25.3 18.0

================
Filter functions
================

We can also filter the table to find those events from O1 by defining a
custom filter function that compares to the start and end GPS times for O1
(taken from the `LOSC Data Usage Notes <https://losc.ligo.org/data/#yellow_box>`_)::

    >>> from gwpy.time import to_gps
    >>> o1start = to_gps("Sep 2015")
    >>> o1end = to_gps("Feb 2016")
    >>> def in_o1(column, interval):
    ...     return (column >= interval[0]) & (column < interval[1])
    >>> print(table.filter(('gps', in_o1, (o1start, o1end))))
       name        gps       m1   m2  snr  network
    --------- ------------- ---- ---- ---- -------
     GW150914  1126259462.0 36.2 23.7 23.7      HL
    LVT151012 1128678900.44 23.0 13.0  9.7      HL
     GW151226 1135136350.65 14.2  7.5 13.0      HL

The custom filter function could have been as complicated as we liked, as long
as the two (and only two) input arguments were the column array for the
relevant column, and the collection of other arguments to work with.

Similarly, we could filter the catalogue to find only those events that include
data from the Virgo observatory::

    >>> import numpy
    >>> print(table.filter(('network', numpy.char.endswith, 'V')))
      name        gps       m1   m2  snr  network
    -------- ------------- ---- ---- ---- -------
    GW170814 1186741861.53 30.5 25.3 18.0     HLV

======================
Using multiple filters
======================

Filters can be trivially chained (either in `str` form, or functional form)::

    >>> print(table.filter('snr > 15', 'distance > 5000'))
      name        gps       m1   m2  snr  distance network
    -------- ------------- ---- ---- ---- -------- -------
    GW170814 1186741861.53 30.5 25.3 18.0      540     HLV

=======
Gotchas
=======

The parser used to intrepet simple filters doesn't recognised strings containing alpha-numeric characters as single words, meaning things like LIGO data channel names will get parsed incorrectly if not quoted.
So, if in doubt, always pass a string in quotes; the quotes will get removed internally by the parser anyway. E.g., use ``channel = "X1:TEST"`` and not ``channel = X1:TEST``.

================
Built-in filters
================

The GWpy package defines a small number of filter functions that implement
standard filtering operations used in gravitational-wave data analysis:

.. automodsumm:: gwpy.table.filters
   :functions-only:
