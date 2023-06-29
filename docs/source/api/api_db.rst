====================
Data Model of a Mesh
====================

Every mesh is stored in a :class:`~polymesh.PolyData` instance, which is a subclass of
:class:`~linkeddeepdict.LinkedDeepDict`, therefore essentially being a nest of dictionaries.
Every container in the nest can hold onto points and cells, data attached to
either the points or the cells, or other similar containers. To store data, every container 
contains a data object for points, or cells, or for both.  These data objects wrap themselves
around instances of :class:`awkward.Record`, utilizing their effective memory layout, handling of jagged
data and general numba and gpu support.

Data Classes
============

.. autoclass:: polymesh.pointdata.PointData
    :members:

.. autoclass:: polymesh.celldata.CellData
    :members:

Mesh Classes
============

.. autoclass:: polymesh.linedata.LineData
    :members: 

.. autoclass:: polymesh.PolyData
    :members: 

.. autoclass:: polymesh.TriMesh
    :members: 

.. autoclass:: polymesh.TetMesh
    :members:

.. autoclass:: polymesh.Grid
    :members: 