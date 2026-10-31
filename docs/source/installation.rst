============
Installation
============

.. note::

   If you are a developer, please refer to the :ref:`installation for developers <development_guide>` section.

You can install the library from ``PyPI`` via ``pip`` using the following command:

.. code-block:: shell

   pip install sigmaepsilon.mesh

To test your installation, run the following command in your Python environment:

.. code-block:: shell

   $ python
   Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import sigmaepsilon.mesh
   >>> sigmaepsilon.mesh.__version__
   'x.y.z'

Where ``x.y.z`` is the version number of ``sigmaepsilon.mesh``.

.. _macos_installation:

macOS
-----

Some functions of the library run nested parallel loops with `Numba <https://numba.pydata.org/>`_. On macOS, Numba installed with ``pip`` or ``uv`` has no TBB threading layer, and its OpenMP threading layer cannot find the OpenMP runtime by default. Numba then falls back to its ``workqueue`` threading layer, which does not support nested parallelism, and the Python process aborts with an error like this:

.. code-block:: text

   Numba workqueue threading layer is terminating: Concurrent access has been detected.

To avoid this, install the OpenMP runtime with `Homebrew <https://brew.sh/>`_ and make it visible to Numba:

.. code-block:: shell

   brew install libomp
   export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib"

Homebrew installs ``libomp`` without linking it into the default library path, which is why ``DYLD_LIBRARY_PATH`` is needed. Add the ``export`` line to your shell profile (e.g. ``~/.zshrc``) to make it permanent. If you use Jupyter, the variable must be set in the environment that starts the kernel.

To check which threading layer Numba uses, run a parallel function first:

.. code-block:: python

   >>> import numba
   >>> from numba import njit, prange
   >>> @njit(parallel=True)
   ... def f(n):
   ...     s = 0.0
   ...     for i in prange(n):
   ...         s += i
   ...     return s
   >>> f(10)
   45.0
   >>> numba.threading_layer()
   'omp'

The result should be ``'omp'``, not ``'workqueue'``. See `issue #159 <https://github.com/sigma-epsilon/sigmaepsilon.mesh/issues/159>`_ for the details.
