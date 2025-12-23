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
    :ref: gwpy.tools.rds.create_parser
    :prog: gwpy-rds
