====================
Testing and coverage
====================

The following command runs all tests and creates a html report in a folder named `htmlcov` 
(the settings are governed by the `.coveragerc` file):

.. code-block:: shell

   python -m pytest --cov-report=html --cov-config=.coveragerc --cov=sigmaepsilon.mesh

Alternatively, you can use Poetry to test the package:

.. code-block:: shell

   poetry run pytest --cov-report=html --cov-config=.coveragerc --cov=sigmaepsilon.mesh
