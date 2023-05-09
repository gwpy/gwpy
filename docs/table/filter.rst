.. currentmodule:: pydischarge.table

.. _pydischarge-table-filter:

################
Filtering tables
################

In order to perform detailed analysis of tabular data, it is useful to
extract portions of a table based on some criteria, this is called filtering.
The `EventTable` object comes with a :meth:`~EventTable.filter` method
that provides an intuitive interface to down-selecting rows in a table.

To demonstrate, we can download |GWTC-2| from |GWOSC|:

.. code-block:: python

   >>> from pydischarge.table import EventTable
   >>> events = EventTable.fetch_open_data("GWTC-2")
   >>> print(events)
          name        chi_eff_upper ...     GPS      final_mass_source_upper
                                    ...                      solMass
   ------------------ ------------- ... ------------ -----------------------
   GW190408_181802-v1          0.14 ... 1238782700.3                     3.9
          GW190412-v3          0.08 ... 1239082262.2                     3.9
   GW190413_052954-v1          0.29 ... 1239168612.5                    12.5
                  ...           ... ...          ...                     ...
   GW190924_021846-v1           0.3 ... 1253326744.8                     5.2
   GW190929_012149-v1          0.34 ... 1253755327.5                    33.6
   GW190930_133541-v1          0.31 ... 1253885759.2                     9.2
   Length = 39 rows

==============
Simple filters
==============

The simplest `EventTable` filter is a `str` statement that provides
a mathematical operation for a column and a threshold.
With the above GWTC-2 events, we can use the filter
``"network_matched_filter_snr > 15"`` to pick out those events with
high signal power:

.. code-block:: python
   :name: pydischarge-table-filter-statement-example
   :caption: Filtering an `EventTable` using a `str` definition

   >>> print(events.filter("network_matched_filter_snr > 15"))
          name        chi_eff_upper ...     GPS      final_mass_source_upper
                                    ...                      solMass
   ------------------ ------------- ... ------------ -----------------------
          GW190412-v3          0.08 ... 1239082262.2                     3.9
   GW190521_074359-v1           0.1 ... 1242459857.5                     6.5
   GW190630_185205-v1          0.12 ... 1245955943.2                     4.4
          GW190814-v2          0.06 ... 1249852257.0                     1.1
   GW190828_063405-v1          0.15 ... 1251009263.8                     7.2

================
Filter functions
================

More complicated filtering can be achieved by defining a `function` that
takes in two arguments - the first being the column slice of the input table,
the second can be whatever you want - and returns a boolean array.
The :meth:`EventTable.filter` method is then called passing in a filter
3-`tuple` with these elements

1. the column name (`str`) or a tuple of names
2. the function to call
3. the other argument(s) for the function (normally a single value, or a
   `tuple` of arguments)

If a single column name is given as the first tuple element, the function will
receive a single `~astropy.table.Column` as the input.
If a `tuple` of names is given, the input will be a slice of the original table
containing only the named columns.

Using the same ``events`` table we can define a function to include only
those events in the first six months of 2019:

.. code-block:: python
   :name: pydischarge-table-filter-function-example
   :caption: Filtering an `EventTable` using a filter function

   >>> from pydischarge.time import to_gps
   >>> start = to_gps("Jan 2019")
   >>> end = to_gps("Jul 2019")
   >>> def q12_2019(column, interval):
   ...     """Returns `True` if ``interval[0] <= column < interval[1]``
   ...     """
   ...     return (column >= interval[0]) & (column < interval[1])
   >>> print(events.filter(('GPS', q12_2019, (start, end))))
          name        chi_eff_upper ...     GPS      final_mass_source_upper
                                    ...                      solMass
   ------------------ ------------- ... ------------ -----------------------
   GW190408_181802-v1          0.14 ... 1238782700.3                     3.9
          GW190412-v3          0.08 ... 1239082262.2                     3.9
   GW190413_052954-v1          0.29 ... 1239168612.5                    12.5
                  ...           ... ...          ...                     ...
   GW190706_222641-v1          0.26 ... 1246487219.3                    18.3
   GW190707_093326-v1           0.1 ... 1246527224.2                     1.9
   GW190708_232457-v1           0.1 ... 1246663515.4                     2.5
   Length = 24 rows

The custom filter function could have been as complicated as we liked, as long
as the two (and only two) input arguments were the column array for the
relevant column, and the collection of other arguments to work with.
For example could filter the table to return only those events with
high mass ratio:

.. code-block:: python
   :name: pydischarge-table-filter-function-example-2
   :caption: Filtering an `EventTable` using multiple columns

   >>> def high_mass_ratio(table, threshold):
   ...     """Returns `True` if ``mass_1_source / mass_2_source >= threshold``
   ...     """
   ...     return (table['mass_1_source'] / table['mass_2_source']) >= threshold
   >>> print(events.filter((('mass_1_source', 'mass_2_source'), high_mass_ratio, 3.0)))
          name        chi_eff_upper ...     GPS      final_mass_source_upper
                                    ...                      solMass
   ------------------ ------------- ... ------------ -----------------------
          GW190412-v3          0.08 ... 1239082262.2                     3.9
   GW190426_152155-v1          0.32 ... 1240327333.3                    None
          GW190814-v2          0.06 ... 1249852257.0                     1.1
   GW190929_012149-v1          0.34 ... 1253755327.5                    33.6

======================
Using multiple filters
======================

Filters can be chained (either in `str` form, or functional form):

.. code-block:: python
   :name: pydischarge-table-filter-chaining
   :caption: Chaining multiple filters with :meth:`EventTable.filter`

   >>> print(events.filter("network_matched_filter_snr > 15", "luminosity_distance > 1000"))
          name        chi_eff_upper ...     GPS      final_mass_source_upper
                                    ...                      solMass
   ------------------ ------------- ... ------------ -----------------------
   GW190521_074359-v1           0.1 ... 1242459857.5                     6.5
   GW190828_063405-v1          0.15 ... 1251009263.8                     7.2

=======
Gotchas
=======

The parser used to interpret simple filters doesn't recognise strings
containing alpha-numeric characters as single words, meaning things like
LIGO data channel names will get parsed incorrectly if not quoted.
So, if in doubt, always pass a string in quotes; the quotes will get removed
internally by the parser anyway. E.g., use ``channel = "X1:TEST"`` and not
``channel = X1:TEST``.

================
Built-in filters
================

The pyDischarge package defines a small number of filter functions that implement
standard filtering operations used in gravitational-wave data analysis:

.. automodsumm:: pydischarge.table.filters
   :functions-only:
