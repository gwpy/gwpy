.. _gwpy-external-framecpp:

########
FrameCPP
########

.. image:: https://img.shields.io/conda/vn/conda-forge/python-ldas-tools-framecpp.svg
    :alt: python-ldas-tools-framecpp conda-forge version
    :target: https://anaconda.org/conda-forge/python-ldas-tools-framecpp
.. image:: https://img.shields.io/conda/pn/conda-forge/python-ldas-tools-framecpp.svg
    :alt: python-ldas-tools-framecpp conda-forge platforms
    :target: https://anaconda.org/conda-forge/python-ldas-tools-framecpp

FrameCPP is a library for reading and writing data in GWF format defined
in :dcc:`LIGO-T970130`.
The main library is written in C++ and documented at

https://computing.docs.ligo.org/ldastools/LDAS_Tools/ldas-tools-framecpp/

GWpy utilises the Python bindings for FrameCPP on Unix platforms
(Linux and macOS) to provide GWF input/output capabilities.

This optional dependency can be installed using `Conda <https://conda.io>`__
(or `Mamba <https://mamba.readthedocs.io/en/stable/>`__):

.. code-block:: shell
    :name: conda-install-python-ldas-tools-framecpp
    :caption: Install python-ldas-tools-framecpp with conda

    conda install -c conda-forge python-ldas-tools-framecpp
