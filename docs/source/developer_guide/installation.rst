===========================
Installation for developers
===========================

For developers, the installation process is a little bit more complicated. We use `uv` to
manage our dependencies and the project as well. To install the project, you can follow the
steps below:

1. Clone the repository using GitHub Desktop or the command line. In the latter case, we recommend using a secure SSH connection over HTTPS.

2. Install `uv` globally:

.. code-block:: shell

   pip install uv

3. Install the library with the necessary optional dependencies by issuing the following command:

.. code-block:: shell

   uv sync --all-groups


This process will install the library with all dependencies. Note that with `uv`, the library itself is always installed in editable mode by default.
However, if you are working across several solutions in the SigmaEpsilon namespace, and you want them installed in editable mode, the suggested
way is to add them as path sources. For instance, if you want to install `sigmaepsilon.core` in editable mode and it is located at the same level as
`sigmaepsilon.mesh`, you issue the following command:

.. code-block:: shell

   uv add --editable ../sigmaepsilon.core


If the library that you want to install in editable mode is located somewhere else, adjust the path accordingly.

.. note::

   On macOS, the test suite aborts unless Numba can use its OpenMP threading layer. Follow the steps in the
   :ref:`macOS installation notes <macos_installation>` before running the tests.
