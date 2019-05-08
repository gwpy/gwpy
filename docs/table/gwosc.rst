.. sectionauthor:: Duncan Macleod <duncan.macleod@ligo.org>
.. currentmodule:: gwpy.table

.. _gwpy-table-gwosc:

#############################
Querying for catalogue events
#############################


.. _gwpy-table-gwosc-query:

==============
Simple queries
==============

The :class:`EventTable` class comes with a :meth:`~EventTable.fetch_open_data`
method that queries the Gravitational-Wave Open Science Center (GWOSC) database
for events and their parameters associated with one of the catalogue event
releases.

.. note::

   For more details on the GWOSC catalogues, see https://www.gw-openscience.org/catalog/.

The simplest query just requires the catalogue name, and will return all parameters
for all events in the catalogue::

   >>> from gwpy.table import EventTable
   >>> t = EventTable.fetch_open_data("GWTC-1-confident")
   >>> print(t)
     name              E_rad                L_peak    ...      tc       utctime
            8.98755e+16 m2 solMass / s2 1e+56 erg / s ...
   -------- --------------------------- ------------- ... ------------ ----------
   GW150914                         3.1           3.6 ... 1126259462.4 09:50:45.4
   GW151012                         1.5           3.2 ... 1128678900.4 09:54:43.4
   GW151226                         1.0           3.4 ... 1135136350.6 03:38:53.6
   GW170104                         2.2           3.3 ... 1167559936.6 10:11:58.6
   GW170608                         0.9           3.5 ... 1180922494.5 02:01:16.5
   GW170729                         4.8           4.2 ... 1185389807.3 18:56:29.3
   GW170809                         2.7           3.5 ... 1186302519.8 08:28:21.8
   GW170814                         2.7           3.7 ... 1186741861.5 10:30:43.5
   GW170817                        0.04           0.1 ... 1187008882.4 12:41:04.4
   GW170818                         2.7           3.4 ... 1187058327.1 02:25:09.1
   GW170823                         3.3           3.6 ... 1187529256.5 13:13:58.5

The full list of available columns can be queried as follows::

   >>> print(t.info)
   <EventTable length=11>
      name     dtype              unit
   ---------- ------- ---------------------------
         name    str8
        E_rad float64 8.98755e+16 m2 solMass / s2
       L_peak float64               1e+56 erg / s
      a_final float64
      chi_eff float64
     distance float64                         Mpc
      far_cwb   str32                      1 / yr
   far_gstlal float64                      1 / yr
    far_pycbc   str32                      1 / yr
        mass1 float64                     solMass
        mass2 float64                     solMass
       mchirp float64                     solMass
       mfinal float64                     solMass
     redshift float64
     sky_size float64                        deg2
      snr_cwb   str32
   snr_gstlal float64
    snr_pycbc   str32
           tc float64
      utctime   str10

.. _gwpy-table-gwosc-filter:

================
Filtered queries
================

The columns returned can be selected using the ``column`` keyword,
and mathematical selection filters can be applied on-the-fly
using the ``selection`` keyword::

   >>> t = EventTable.fetch_open_data(
   ...     "GWTC-1-confident",
   ...     selection="mass1 < 4",
   ...     columns=["name", "mass1", "mass2", "distance"],
   ... )
   >>> print(t)
     name    mass1   mass2  distance
            solMass solMass   Mpc
   -------- ------- ------- --------
   GW170817    1.46    1.27     40.0

For more information on filtering, see :ref:`gwpy-table-filter`.

For some examples of visualising the catalogues, see
:ref:`gwpy-example-table-scatter` and :ref:`gwpy-example-table-histogram`.

.. _gwpy-table-gwosc-index:

======================
Indexing by event name
======================

The `EventTable` returned by :meth:`~EventTable.fetch_open_data` includes an
index based on the ``'name'`` column, so you can directly access rows in the
table based on their name::

   >>> t = EventTable.fetch_open_data(
   ...     "GWTC-1-confident",
   ...     columns=["name", "utctime", "mass1", "mass2", "distance"],
   ... )
   >>> print(t.loc["GW170817"])
     name    utctime    mass1   mass2  distance
                       solMass solMass   Mpc
   -------- ---------- ------- ------- --------
   GW170817 12:41:04.4    1.46    1.27     40.0
