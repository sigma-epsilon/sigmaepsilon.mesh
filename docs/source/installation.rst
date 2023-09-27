Installation
============

sigmaepsilon.mesh can be installed from PyPI using `pip` on Python >= 3.8:

.. code-block:: sh

   $ pip install sigmaepsilon.mesh

or chechkout with the following command using GitHub CLI

.. code-block:: sh

   $ gh repo clone sigma-epsilon/sigmaepsilon.meshpip install sigmaepsilon.mesh

and install from source by typing

.. code-block:: sh

    $ pip install .

If you want to run the tests, you can install the package along with the necessary optional dependencies like this

.. code-block:: sh

    $ pip install ".[test]"

Installation for development
----------------------------

If you are a developer and want to install the library in development mode, the suggested way is by using this command:

.. code-block:: sh

    $ pip install "-e .[test, dev]"

Check your installation
-----------------------

.. code-block:: sh

    $ python
    Python 3.8.10 (tags/v3.8.10:3d8993a, May  3 2021, 11:48:03) [MSC v.1928 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    $ import sigmaepsilon.mesh
    $ sigmaepsilon.mesh.__version__
    '1.1.0'