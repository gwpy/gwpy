###################################
Creating reduced datasets with GWpy
###################################

The command-line tool `gwpy-rds` provides a convenient way to create reduced
data sets using GWpy.

`gwpy-rds` accesses data for the requested channels for a single time
interval, and writes these data to a single new file.
This can be then fed to batch workflows, or other applications, to
speed up data access without needing to transfer the original full dataset.

=====
Usage
=====

.. argparse::
   :ref: gwpy.tool.rds.create_parser
   :prog: gwpy-rds

========
Examples
========

.. code-block:: bash
    :name: gwpy-rds
    :caption: How to use `gwpy-rds`.

    gwpy-rds --start "2015-09-14 09:50:41" --end "2015-09-14 09:50:49" --channel H1:GDS-CALIB_STRAIN --output-file gw150914.h5
