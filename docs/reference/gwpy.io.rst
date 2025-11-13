:tocdepth: 2

#############################
Input/Output (:mod:`gwpy.io`)
#############################

.. automodule:: gwpy.io

.. seealso::

    - **TimeSeries I/O**: :doc:`/timeseries/io`
    - **FrequencySeries I/O**: :doc:`/spectrum/io`
    - **Segment I/O**: :doc:`/segments/io`
    - **Table I/O**: :doc:`/table/io`

.. automodapi:: gwpy.io
    :include-all-objects:
    :no-heading:
    :no-main-docstr:

----

.. toctree::
    :caption: General I/O utilities
    :maxdepth: 1

    gwpy.io.cache
    gwpy.io.utils

.. toctree::
    :caption: Data Discovery
    :maxdepth: 1

    gwpy.io.datafind

.. toctree::
    :caption: Data Format Handlers
    :maxdepth: 1

    gwpy.io.gwf
    gwpy.io.hdf5
    gwpy.io.ligolw
    gwpy.io.root

.. toctree::
    :caption: Authentication / authorisation utilities
    :maxdepth: 1

    gwpy.io.kerberos
    gwpy.io.scitokens

.. toctree::
    :caption: Remote Data Access
    :maxdepth: 1

    gwpy.io.nds2
    gwpy.io.pelican
    gwpy.io.remote


.. toctree::
    :caption: I/O registry
    :maxdepth: 1

    gwpy.io.registry
