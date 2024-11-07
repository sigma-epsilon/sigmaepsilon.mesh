===========================
Installation for developers
===========================

For developers, the installation process is a little bit more complicated. We use `Poetry` to
manage our dependencies and the project as well. To install the project, you can follow the
steps below:

1. Clone the repository using GitHub Desktop or the command line. In the latter case, we recommend using a secure SSH connection over HTTPS.

2. Install `Poetry` globally:

.. code-block:: shell
   
   pip install poetry

3. Upgrade `pip`
   
.. code-block:: shell
   
   poetry run pip install --upgrade pip

4. Install the library with the necessary optional depencencies by issuing the following command:

.. code-block:: shell
   
   poetry install --with dev,test,docs
   

This process will install the library with all dependencies. Note that with `Poetry`, libraries are always installed in editable mode by default. 
However, If you are working across several solutions in the SigmaEpsilon namespace, and you want them installed in editable mode, the suggested 
way is to use `Pip`. For instance, if you want to install `sigmaepsilon.core` in editable mode and it is located at the same level as 
`sigmaepsilon.mesh`, you issue the following command:

.. code-block:: shell

   poetry run pip install -e ..\sigmaepsilon.core


If the library that you want to install in editable mode is located somewhere else, adjust the path accordingly.