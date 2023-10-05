Installation for development
============================

If you are a developer and want to install the library in development mode, the suggested way is by using this command:

.. code-block:: sh

    $ pip install "-e .[test, dev]"

Testing
=======

To test the library, we use `pytest`. Run the following command to run all tests:

.. code-block:: sh

    $ pytest tests
    