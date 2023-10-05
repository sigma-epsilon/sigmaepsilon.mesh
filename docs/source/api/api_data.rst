===============
Data structures
===============

Every mesh is stored in a :class:`~sigmaepsilon.mesh.data.PolyData` instance, which is a subclass of
:class:`~linkeddeepdict.LinkedDeepDict`, therefore essentially being a nest of dictionaries.
Every container in the nest can hold onto points and cells, data attached to
either the points or the cells, or other similar containers. To store data, every container 
contains a data object for points, or cells, or for both.  These data objects wrap themselves
around instances of `awkward.Record`, utilizing their effective memory layout, handling of jagged
data and general numba and gpu support.

.. automodule:: sigmaepsilon.mesh.data
    :members:
