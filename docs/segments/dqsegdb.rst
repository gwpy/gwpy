.. currentmodule:: pydischarge.segments

.. _pydischarge-segments-dqsegdb2:

####################
The Segment Database
####################

.. warning::

   Access to the GW segment database is reserved for members of
   the LIGO-Virgo-KAGRA collaborations.


The LIGO and Virgo instruments utilise hundreds of data-quality flags
to record instrumental state on a daily basis.
These flags are stored in a joint segment database - a queryable database
recording each flag, its valid and active segment lists, and all metadata
associated with its generation.
The segment database is the primary access point for users to study
data-quality flags and apply them in any analysis.

The `DataQualityFlag` object includes the :meth:`~DataQualityFlag.query`
`classmethod` with which to access segments stored in any segment database::

    >>> from pydischarge.segments import DataQualityFlag
    >>> segs = DataQualityFlag.query('L1:DMT-ANALYSIS_READY:1',
    ...                              'Sep 14 2015', 'Sep 15 2015')

The above command will return the complete record for the
LIGO-Livingston Observatory (``L1``) observing segments for the day of
September 14 2015 (all times should be given in UTC or GPS).

.. note::

    :meth:`DataQuality.query` calls out to `dqsegdb2`, see its documentation
    for more information and extra functions you can use to interact with
    the segment database.

.. note::

    Members of LIGO-Virgo-KAGRA can also go to https://segments-web.ligo.org
    to search for segments using their browser.
