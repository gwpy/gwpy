.. _pydischarge-external-lalsuite:

########
LALSuite
########

.. image:: https://img.shields.io/conda/vn/conda-forge/lalsuite.svg
   :alt: lalsuite conda-forge version
   :target: https://anaconda.org/conda-forge/lalsuite
.. image:: https://img.shields.io/conda/pn/conda-forge/lalsuite.svg
   :alt: lalsuite conda-forge platforms
   :target: https://anaconda.org/conda-forge/lalsuite

LALSuite is a large collection of gravitational-wave data analysis libraries,
mainly written in C but with high-level bindings for Python.

The main LALSuite documentation can be found at

https://lscsoft.docs.ligo.org/lalsuite/

LALSuite is made up of a number of component libraries, some of which are
utilised by pyDischarge to provide various extra capabilities.

The full LALsuite can be installed alongside pyDischarge using
`Conda <https://conda.io>`__
(or `Mamba <https://mamba.readthedocs.io/en/stable/>`__)
or `pip`:

.. md-tab-set::
    :name: pydischarge-install-lalsuite

    .. md-tab-item:: Install LALSuite with conda/mamba

        .. code-block:: shell

            conda install -c conda-forge lalsuite

    .. md-tab-item:: Install LALSuite with pip

        .. code-block:: shell

            python -m pip install lalsuite

.. _pydischarge-external-lal:

===
LAL
===

.. image:: https://img.shields.io/conda/vn/conda-forge/lal.svg
   :alt: lal conda-forge version
   :target: https://anaconda.org/conda-forge/lal
.. image:: https://img.shields.io/conda/pn/conda-forge/lal.svg
   :alt: lal conda-forge platforms
   :target: https://anaconda.org/conda-forge/lal

LAL is the base-level library in LALSuite that provides much of the core
functionality.
It's documentation is at

https://lscsoft.docs.ligo.org/lalsuite/lal/

pyDischarge provides utilities to transform to and from LAL objects
(like the :lal:`XLALREAL8TimeSeries`) from/to their pyDischarge equivalent
(e.g. :class:`pydischarge.timeseries.TimeSeries`).

LAL can be installed using `Conda <https://conda.io>`__
(or `Mamba <https://mamba.readthedocs.io/en/stable/>`__)
on Unix systems (not Windows):

.. code-block:: shell
    :name: conda-install-lal
    :caption: Install python-lal with conda

    conda install -c conda-forge python-lal

.. _pydischarge-external-lalframe:

========
LALFrame
========

.. image:: https://img.shields.io/conda/vn/conda-forge/lalframe.svg
   :alt: lalframe conda-forge version
   :target: https://anaconda.org/conda-forge/lalframe
.. image:: https://img.shields.io/conda/pn/conda-forge/lalframe.svg
   :alt: lalframe conda-forge platforms
   :target: https://anaconda.org/conda-forge/lalframe

LALFrame provides a GWF I/O library compatible with LAL series types.
It's documentation is at

https://lscsoft.docs.ligo.org/lalsuite/lalframe/

pyDischarge utilises the Python bindings for LALFrame on Unix platforms
(Linux and macOS) to provide GWF input/output capabilities.

This optional dependency can be installed using `Conda <https://conda.io>`__
(or `Mamba <https://mamba.readthedocs.io/en/stable/>`__)
on Unix systems (not Windows):

.. code-block:: shell
    :name: conda-install-python-lalframe
    :caption: Install python-lalframe with conda

    conda install -c conda-forge python-lalframe
