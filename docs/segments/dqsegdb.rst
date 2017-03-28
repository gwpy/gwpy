.. currentmodule:: gwpy.segments
.. include:: ../references.txt

.. _gwpy-segments-dqsegdb:

####################
The Segment Database
####################

**Additional dependencies**: |glue|_, |dqsegdb|_

The LIGO and Virgo instruments utilise hundreds of data-quality flags to record instrumental state on a daily basis.
These flags are stored in a joint segment database - a queryable database recording each flag, its valid and active segment lists, and all metadata associated with its generation.
The segment database is the primary access point for users to study data-quality flags and apply them in any analysis.

The `DataQualityFlag` object includes the :meth:`~DataQualityFlag.query` classmethod with which to access segments stored in any segment database::

    >>> from gwpy.segments import DataQualityFlag
    >>> segs = DataQualityFlag.query('L1:DMT-ANALYSIS_READY:1', 'Sep 14 2015', 'Sep 15 2015')

The above command will return the complete record for the LIGO-Livingston Observatory (``L1``) observing segments for the day of September 14 2015 (all times should be given in UTC or GPS).

.. note::

    Members of the LIGO Scientific Collaboration or the Virgo Collaboration
    can also go to https://segments-web.ligo.org to search for segments
    using their browser.
