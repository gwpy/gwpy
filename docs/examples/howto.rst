#########################################
Gwpy-devel: how to write a worked example
#########################################

This page outlines the steps that should be followed by developers to provide a worked example for any module or function in `GWpy`.

#. Write your example using the `IPython notebook <http://ipython.org/notebook.html>`_ environment. Each example must include:

    #. A description of the problem, under a level-2 heading 'Problem'.
    #. A worked solution, also under a level-2 heading 'Solution'. Please include all necessary text to describe each step of the solution, any functions you are using, and the arguments required. The solution should not require imports of any modules not listed in the package dependencies; if the examples desperately needs it, you should ask for the package to be included in the dependencies.

#. Test your example code runs from start to finish in the notebook environment.

#. Save and Checkpoint the notebook in an evaluated state (i.e. with all cells evaluated, and plots generated) in the ``examples/`` directory of the git repository. The notebook should be named ``gw_ex_myexample.ipynb`` where ``myexample`` is a unique filename tag that describes the purpose of the examples, e.g. ``gw_ex_plot_timeseries.ipynb``. If the notebook includes plots, please run the final save with the ``--pylab inline`` options to ipython, to include plots where they sit in the notebook.

#. Export the notebook to python format in the ``examples/`` directory using nbconvert:

.. code:: bash
   ipython nbconvert --to python example.ipynb

#. Test that the sphinx-rendering of the example will work using the Makefile in the ``docs/`` directory. View the output in a web browser to make sure:

.. code:: bash
   cd docs
   make examples html
