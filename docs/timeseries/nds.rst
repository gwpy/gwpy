.. currentmodule:: gwpy.timeseries.core

#################################
Fetching a `TimeSeries` using NDS
#################################

The `Network Data Server <https://www.lsc-group.phys.uwm.edu/daswg/projects/nds-client.html>`_ protocol, developed by the LIGO Scientific Collaboration, offers methods to remotely fetch data stored on disk, allowing users to download data to their laptops, or other machines.

=================
Available servers
=================

The LIGO Data Grid operates the following NDS servers, with authenticated access for collaboration members:

===========================  =================================================================
``nds.ligo.caltech.edu``     General access to the majority of available data
``nds.ligo-la.caltech.edu``  Lower-latency access to data from the LIGO Livingston Observatory
``nds.ligo-wa.caltech.edu``  Lower-latency access to data from the LIGO Hanford Observatory
===========================  =================================================================

=====================
Fetching `TimeSeries`
=====================


.. warning::

   Fetching data via NDS requires the |nds2-client|_ package (including SWIG bindings for Python) to be installed on your system.

.. |nds2-client| replace:: ``nds2-client``
.. _nds2-client: https://www.lsc-group.phys.uwm.edu/daswg/projects/nds-client.html

Data for a given :class:`~gwpy.detector.channel.Channel` can be fetched from NDS using the :meth:`TimeSeries.fetch` method::

    >>> from gwpy.timeseries import TimeSeries
    >>> data = TimeSeries.fetch('L1:PSL-ODC_CHANNEL_OUT_DQ', 1067042880, 1067042912)

where the arguments are a channel name, and GPS start and end times.
Optionally, the NDS server to use can be specified using the ``host`` and ``port`` keyword arguments; if these are not given, GWpy will cycle through a logically-ordered list of hosts, starting the from host server for the relevant observatory.

=======================
Kerberos authentication
=======================

Access to the authenticated NDS2 protocol services (all apart from ``nds40``) require a valid and active kerberos ticket.
On all Unix-based operating systems, the ``kinit`` command-line tool provided by the KRB5 package is the standard method of generating or renewing such a ticket.
If this utility is available on your system, the :meth:`TimeSeries.fetch` method will call it if connecting to the NDS server fails due to an authentication error, as in the following example::

    >>> from gwpy.timeseries import TimeSeries
    >>> data = TimeSeries.fetch('L1:PSL-ODC_CHANNEL_OUT_DQ', 1067042880, 1067042912)
    error detail: SASL(-1): generic failure: GSSAPI Error: Unspecified GSS failure.  Minor code may provide more information (Credentials cache file '/tmp/krb5cc_5308' not found)

    Error authenticating against nds.ligo-la.caltech.edu
    Please provide username for the LIGO.ORG kerberos realm: albert.einstein
    Password for albert.einstein@LIGO.ORG:
    Kerberos ticket generated for albert.einstein@LIGO.ORG

where the user has been prompted for both their LIGO.ORG username and password, in order to generate a ticket.

Kerberos keytabs
-----------------

Users can generate a kerberos keytab file to enable password-less tickets using the `ktutil` tool:

Run `ktutil`::

   $ ktutil

At the ``ktutil:`` prompt enter::

   addent -password -p albert.einstein@LIGO.ORG -k 1 -e des3-cbc-sha1

and enter your password when prompted.
At the next prompt enter ``wkt`` followed by the name of the kerberos keytab to write to, and then quit::

   wkt albert.einstein.keytab
   quit

Users should store the location of this file in the ``KRB5_KTNAME`` environment variable, enabling :meth:`TimeSeries.fetch` to locate and utilise the keytab for seamless authentication.
