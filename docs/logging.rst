.. sectionauthor:: Duncan Macleod <macleoddm@cardiff.ac.uk>
.. _gwpy-logging:

#################
Logging with GWpy
#################

Many GWpy modules and classes use the standard Python logging module to provide
informative messages about their operation.
This allows users to control the
verbosity of output and to capture log messages for later analysis.

.. _gwpy-logging-init:

Configuring Logging
===================

Just GWpy
---------

To configure logging just for GWpy modules, you can use the following code:

.. code-block:: python
    :caption: Configuring logging for GWpy modules

    import gwpy
    gwpy.init_logging()

This will default to the `INFO` logging level, but can be overridden by
passing a different value to the ``init_logger`` function, or by setting the
``GWPY_LOG_LEVEL`` environment variable to the name or number of a valid
logging level (see :ref:`gwpy-logging-env`).

.. _gwpy-logging-env:

From the environment
--------------------

You can also configure logging for GWpy modules by setting the following
environment variables:

.. code-block:: shell
    :caption: Configure logging for GWpy modules from the environment

    export GWPY_INIT_LOGGING=1
    export GWPY_LOG_DATEFMT="%Y-%m-%d %H:%M:%S"
    export GWPY_LOG_FORMAT="%(asctime)s %(name)s %(levelname)s: %(message)s"
    export GWPY_LOG_LEVEL=INFO

For more details on the available environment variables, see :ref:`gwpy-env-variables`.

.. _gwpy-logging-global:

Global logging
--------------

To configure logging for all modules, including GWpy, you can use the
following code:

.. code-block:: python
    :caption: Configuring global logging

    import logging
    logging.basicConfig(level=logging.INFO)

For more details on configuring logging in Python, see
:doc:`python:howto/logging`.

.. _gwpy-logging-coloredlogs:

Coloredlogs
===========

If you have the `coloredlogs` package installed, `gwpy.init_logging` will
configure coloured log messages, which can make it easier to distinguish
between different levels of output.

To configure coloured logging for all modules, you can use the
following code:

.. code-block:: python
    :caption: Configuring global coloured logging

    import coloredlogs
    coloredlogs.install(level='INFO')

Or set the following environment variables:

.. code-block:: shell
    :caption: Setting environment variables for coloured logging

    export COLOREDLOGS_AUTO_INSTALL=1
    export COLOREDLOGS_LOG_LEVEL=INFO

For more details on configuring `coloredlogs`, see :doc:`coloredlogs:api`.
