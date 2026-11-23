# Design proposal for `sigmaepsilon.mesh`: data model, classes and configuration

**Status:** draft, under discussion. Nothing in this document has been implemented yet.

**Scope:** how point and cell data, block configuration and computed values are stored in a `PolyData` tree, how users attach their own data, how packages such as `sigmaepsilon.solid.fem` extend the model, how meshes are plotted, saved and loaded, and which class plays which role (containers, data holders, cell types, schemas, configuration). Section 3.15 sets the class names used throughout, and section 3.16 shows the whole design in one example.

The findings in section 2 were checked by running code against `main` at `ad942db`.

---

## 1. Context and goals

A mesh is a tree of `PolyData` blocks (a `DeepDict`). In the common case, the root holds the point data and each leaf block holds one block of cells, whose topology refers to those points. Inner blocks may hold their own point data, so one mesh object can host several meshes in the same space. Point data and cell data both hold data about their points and cells. Today all of that data is stored with Awkward.

Goals:

- Store data efficiently.
- Store metadata for each block.
- Keep internal data (e.g. topology) next to data provided by the user.
- Let users attach any data to points or cells without name clashes with internal data.
- Store scalars and plain arrays, with the same rules for data declared by packages and data attached by users. Vector and tensor data that transform with frames are a future extension (section 3.14).
- Keep the data of a block regular: ragged arrays appear only in results that span blocks (e.g. the `topology()` of a mixed subtree), without Awkward union or option types.
- Keep the model extensible: `sigmaepsilon.solid.fem` builds on it, and more packages may do so in the future.
- Save meshes and load them back, preferably to a single file.
- Make the classes easy to read: one role per class, and names that say what each class holds.

---

## 2. Audit of the current implementation

### 2.1 Internal and user field names can clash

- The only protection is a naming convention, and it isn't consistent:
  - `PointData` uses double underscores (`__x`, `__id`, `__activity`), and its docstring warns users about them.
  - `CellData` uses single underscores (`_topology`, `_id`, `_frames`, `_t`).
  - `solid.fem` adds a third convention (`_K_`, `_density_`, `_shapes_`), and some of its internal fields have no prefix at all (`loads`, `mass`, `fixity`, `forces`, `reactions`).
- `cd.db["_id"] = zeros` silently overwrites the global cell ids.
- Reading and writing by key don't agree. `AkWrapper.__getitem__` translates public names through `_attr_map_`, but `__setitem__` doesn't. After `cd.db["nodes"] = my_array`, reading `cd.db["nodes"]` returns the topology, and the user's array can't be reached by that key.
- `CellData(fields={"t": ..., "id": ...})` silently remaps the user's `t` and `id` onto the internal fields.
- Internal names leak into `fields`, `to_dataframe` and parquet output.

### 2.2 Subclasses can't reliably extend the attribute map

`ABCMeta_AkWrapper` merges `_attr_map_` so that the **base class wins**. In `solid.fem`, `CellData` declares `"activity": "_activity_"`, but the merged map resolves to the mesh's `_activity`.

### 2.3 Attribute access falls through to Awkward

- `PolyCell` falls through to `CellData`, which falls through to the Awkward array.
- `pd.temperature = arr` doesn't store data: it quietly becomes a plain Python attribute, so it never reaches exports or plots.
- Which attributes exist depends partly on Awkward's own API, and error messages are hard to read.

### 2.4 Awkward is mostly used as a dict of regular columns

- Every field the code relies on is a regular array: coordinates, topology, frames, nodal distribution factors. Each access goes through `.to_numpy()`.
- Writing into a returned array in place works today because the layout is regular. It silently stops working once a column becomes optional or ragged.
- Per-cell frames are always stored expanded to (nE, 3, 3), even when every cell shares one frame (2160 bytes for 30 cells with one frame in the test).
- All fields of a block are zipped into one record. Combining blocks with different fields produces union and optional types on export (`var * int64`, `?float64`).

### 2.5 Block metadata and configuration live in ad hoc places

- `mesh['A']['material'] = 'steel'` works, but it sits in the same dict as the child blocks.
- Because the tree auto-creates missing keys, a typo like `mesh['A']['materal']` silently creates an empty child block.
- `config` (plotting settings, a `DeepDict` per block) is a third place where per-block state is kept. It is meant to cascade: `_get_config_(key)` returns the config of the nearest block, going up the tree, that has `key`. In practice it doesn't:
  - Class defaults are written into every instance by `_init_config_` (`show_edges` on every `PolyData`; colour, line width and tubes on every `LineData`). So every block already has the default key `("pv", "default")`, and a value set on a parent never reaches its children. Checked: after `mesh.config[key]["color"] = "red"`, a child block still resolves to `{'show_edges': True}`.
  - The lookup replaces whole dicts instead of merging single options, so a child that sets one option loses all options of its parents.
  - `pvplot` calls `config.pop("color")` on the dict returned by `_get_config_`, which is the stored config itself. Plotting a block with scalars permanently removes its colour setting (checked).
  - The keys aren't consistent: the code uses `("pv", "default")` and `("k3d", "default")`, the docstring of `PolyData.config` uses `('pyvista', 'plot')`.
  - Defaults live in instance state, so they are copied rather than inherited: `solid.fem`'s `LineMesh._init_config_` repeats `LineData`'s line by line, and a changed class default doesn't reach existing objects.
  - The config isn't saved with the mesh.

### 2.6 Ids depend on where a block sits in the tree

Global point and cell ids come from the root's index managers. They are rewritten when a block joins a tree and deleted when it leaves (`_reindex`, `__after_join_parent__`, `__after_leave_parent__`). A block's stored data changes whenever it moves, and copying or merging needs re-indexing.

### 2.7 Nothing round-trips through a file

- `pickle` returns a mesh with `pd=None` and `cd=None`. `DeepDict.__reduce__` passes only the dict items, and point and cell data are instance attributes.
- `PolyData.to_parquet` writes two flat files with all blocks concatenated. The tree, frames, cell types and metadata are lost, and there is no `PolyData.from_parquet`.
- `to_ak` goes through Python lists: 2.9 s for 216k H8 cells.
- The remote branches `154-save-to-one-file` and `feature/save_to_one_file` (last commit 2025-07-26) contain no save or load implementation. Their changes are build, plotting and type-hint updates.

### 2.8 Bugs found along the way

- `CellData.set_nodal_distribution_factors` always raises `KeyError: '_ndf'`. It looks up `_attr_map_` with the internal key, and its fallback indexes with `.id` instead of `_id`.
- `AkWrapper.to_dict` and `to_list` call `fields.extend(args)`, which modifies the caller's list (and fails for tuples).
- `IndexManager.generate`, behind the root's `pim` and `cim`, hands out the same id twice. When the recycled ids cover the request, it still runs `self.next += n - nQ`, which moves the counter backwards. After `generate(10)` and `recycle(range(10))`, `generate(4)` and `generate(10)` return `[0, 1, 2, 3]` and `[4, 5, 6, 7, 8, 9, 4, 5, 6, 7]` (checked). Fix: advance the counter only by the ids generated beyond the queue.
- `IndexManager.recycle` accepts ids that aren't in use: ids never generated, negative ids, and ids already recycled or passed twice in one call. Each one is generated twice later. Fix: raise `ValueError` for them.
- `PointCloud.closest` and `furthest` with several targets return `PointCloud(..., inds=id)`, the rows of the slice, instead of `self.inds[id]`, the rows of the original cloud. `pc[[5, 10, 20]].closest(c[[20, 10]]).inds` is `[2, 1]` instead of `[20, 10]` (checked).
- `PointCloud.index_of_furthest`, and so `furthest`, fails on a plain array of several targets with a `ValueError` from `matmul`: unlike `index_of_closest`, it doesn't infer the frame from the shape of the input (checked). Its docstring also says "closest".

### 2.9 Methods on a subtree don't line up

`PolyData` methods such as `topology()`, `volumes()` and `cell_ids()` walk the cell blocks of a subtree, but each one does its own walk with its own rules. Checked on a block holding two H8 cells and seven L2 cells:

- `cell_ids()` and `volumes()` return 9 values, but `areas()` returns 7, because it drops the 3D blocks. Its docstring says to match the values with `cell_ids()`, which no longer works.
- `topology()` returns a ragged `TopologyArray` (8 and 2 nodes per cell), but `cells_coords()` passes it to a Numba kernel that expects a regular array and fails with a `TypingError`.
- `volume()` returns 9.0: 2.0 for the solids and 7.0 for the beams, whose volumes use a cross-section area of 1.0 when none is given.
- There is no `id` on `PolyData`; the cell ids of a subtree come from the method `cell_ids()`, while blocks with data use `celldata.id`.

---

## 3. Proposal

### 3.1 Core fields and user fields instead of naming conventions

Each `PointData` and `CellData` has two kinds of fields:

- **Core fields** are declared by a package in the class's `Schema` (section 3.5): `x`, `topology`, `activity` by `mesh`, `K` by `fem`, and so on. They are reached through generated properties (`pd.x`, `cd.topology`). They never appear among user fields. Frames (`pd.frame`, `cd.frames`) are not fields either: they are core state of their block, given at construction and read-only afterwards (section 3.3).
- **User fields** live in `pd.data` / `cd.data`, a `MutableMapping` with no naming rules. `pd["temperature"]` stays as a shortcut for `pd.data["temperature"]`.

This is how PyVista (`point_data`), meshio and xarray (`coords` vs `data_vars`) handle it. User fields can't clash with core fields and need no prefixes. The `__getattr__` fall-through to Awkward is removed.

```python
coords, topo = grid(size=(1, 1, 1), shape=(2, 2, 2), eshape="H8")
frame = CartesianFrame(dim=3)
pd = PointData(x=coords, frame=frame)
cd = H8(topology=topo, frames=frame)
nP, nE = len(coords), len(topo)

cd.data["id"] = np.arange(nE) + 100      # a user field called "id"
cd.id                                    # still the global cell ids (section 3.7), untouched
list(cd.data)                            # ['id']: only user fields are listed

my_nodes = np.zeros((nE, 8))
cd.data["nodes"] = my_nodes
cd.data["nodes"]                         # my_nodes: reading and writing use the same key
cd.topology                              # the topology, reached only through its property

temperature = np.full(nP, 20.0)
pd.temperature = temperature             # AttributeError instead of a silent Python attribute
pd["temperature"] = temperature          # stored, exported and plotted
```

Core field names are plain and unqualified: there are no package namespaces. The ecosystem is hierarchical (`mesh` → `fem` → ...), so a core field name is unique along the class hierarchy. A package that reuses a name declared by a class it extends gets an error at class creation (section 3.5). User fields are a separate mapping, so users can still use any name, including the names of core fields.

Core fields and user fields differ only in where their spec comes from and how they are reached. They follow the same schema rules: the same kinds, locations and options, the same validation on every write, and the same storage (section 3.2).

### 3.2 Fields

A field is a typed object, not a bare column: values plus the spec that describes them.

```text
Field
├─ values   ndarray                 item axes + component axes (layout rule below)
└─ spec     FieldSpec               kind, location, dtype, component shape, options
```

The two parts live in different places:

| | Values | Spec |
|---|---|---|
| Core field (`cd.topology`, `cd.K`) | the private storage of the data holder (`PointData` / `CellData` instance), reached through `cd.core` and the generated properties | the `Schema` of the class: one spec for every instance, fixed at class creation (section 3.5) |
| User field (`cd.data["damage"]`) | the private storage of the data holder, reached through `cd.data` | the data holder too, next to the values: each block has its own user-field specs, fixed by the first write (below) |
| In a saved file | one dataset per field and block (section 3.12) | an attribute of that dataset; the file also stores the merged `Schema` of every class with its owners |

A uniform value is stored in the data holder like any other value, as a single value instead of an array. The inherited value of a field is not part of the field: it is an attribute in the configuration of a block above (section 3.6). There is no `Field` object at runtime: reading a field returns its values as an `ndarray`, and `spec(name)` returns its spec.

**Kinds:**

| kind | Values per item | Returned as |
|---|---|---|
| `"scalar"` | one number (temperature, thickness, density) | `ndarray` |
| `"array"` | a regular block of numbers (ids, flags, colours, topology, material matrices) | `ndarray` |

- Neither kind transforms with a frame: for the mesh, values are plain numbers. If a package stores the components of a physical vector or tensor in an `"array"` (e.g. the material stiffness `C` of `fem`), the frame they refer to is a convention of that package, and the mesh never changes them. Frames are fixed after construction (section 3.3), so the mesh can't break that convention.
- `kind` is kept instead of being inferred from the shape, because it is where vector and tensor kinds will be added (section 3.14).

**Locations.** Where a value sits follows from the host of the field, and only cell data has a choice:

| Host | `location` | One value per | Item axes |
|---|---|---|---|
| point data | none | point of the point cloud | points |
| cell data | `"cell"` (default) | cell | cells |
| cell data | `"node"` | node of each cell, in the library's node order of the cell type | cells, nodes |
| cell data | `"master"` | sample point of each cell, given in master coordinates, in the given order | cells, sample points |

- **Point data** has no locations: a field of a point cloud always has one value per point, and its spec has no `location`. A value per point is shared by every cell that uses the point. Data that belongs to the cells belongs to the cell data, even when it sits at the nodes: the node values of neighbouring cells may differ (e.g. stresses).
- **`"node"`** is a preset: values at the nodes of the cell type, in the order of its `master_coordinates()`. A `"master"` field whose points happen to be the nodes stays a `"master"` field, and its values are never reordered.
- **`"master"`** needs the points: `at` in the spec holds the master coordinates of the sample points, an `(nS, dim)` array in the reference domain of the block's cell type. Several fields of one block can use different points, e.g. a full and a reduced quadrature rule for selective integration. The user provides the points; presets such as named quadrature rules are a possible future extension.
- **Master coordinates** follow the conventions of the cell type: [-1, 1]ᵈ for quadrilaterals and hexahedra, and the reference domain the cell type defines for triangles, tetrahedra and wedges. There is no general standard for these, so they are documented per cell type. They are treated as stable and are not versioned (section 3.13).
- **Utilities** map points between physical and master coordinates. Given points in any frame, they find the containing cells and the master coordinates within them, building on what `PolyCell` has today (`locate`, `glob_to_loc`, `loc_to_glob`).

**One spec type for every field.** Core fields and user fields are described by the same `FieldSpec`:

```python
class FieldSpec(BaseModel):                  # pydantic: validated, JSON-serializable
    kind: Literal["scalar", "array"]
    location: Literal["cell", "node", "master"] | None = None   # None for point data; "cell" by default for cell data
    at: list[list[float]] | None = None      # master coordinates (nS, dim), only for "master";
                                                     # fixed by the first write when omitted
    dtype: str | None = None                 # fixed by the first write when omitted
    shape: tuple[int, ...] | None = None     # component axes; fixed by the first write when omitted
    default: Any = None
    persist: bool = True                     # False: kept in the session, never saved (section 3.11)
```

**The same rules for core fields and user fields:**

1. A field has exactly one spec, fixed once the field exists.
2. Every write is validated against that spec: the kind, the item axes against the location (number of points, cells, nodes per cell, or sample points of `at` per cell), the dtype (safe casts only) and the component shape.
3. Both follow the layout rule below, can be uniform, and are saved with their spec (section 3.12).
4. A spec changes only by removing the field and declaring it again: through a migration for core fields (section 3.13), with `del` for user fields.

The differences are about where a field comes from and how it is reached, not about what it can hold:

| | Core fields | User fields |
|---|---|---|
| Spec comes from | the `Schema` of the class, at class creation (section 3.5) | the first write: given with `spec=`, or inferred from the value |
| Reached through | generated properties, `cd.core` | `cd.data`, `pd["..."]` |
| Names | unique along the class hierarchy | any name, in a separate mapping |
| Versioning | the schema version of the owning package, with migrations | none: saved as they are |

An inferred spec is the narrowest one that fits the first value: kind `"scalar"` for one value per item and `"array"` otherwise, the location `"cell"` on cell data unless `location=` says otherwise, the points given with `at=` for `"master"`, and the dtype and component shape of the value. A user who wants a different spec, e.g. a wider dtype than the first value has, passes it with `spec=`.

**Layout rule for values.** The axes of `values` form two consecutive groups:

- **Item axes** come first and say which item a value belongs to. Their number follows from the location: a point data field or a `"cell"` field has one item axis (the points or cells), a `"node"` or `"master"` field has two (the cell, then the node or sample point within it).
- **Component axes** come last and hold the value of one item: none for a scalar, the `shape` of the spec for an array.

Values stored in a block are always regular, because a block holds cells of one type (section 3.8). Raggedness appears only in results that span blocks (section 3.9), and there only item axes may be ragged; component axes are always regular. For example, the state variables of a material model, four numbers per sample point, read from a subtree with a block of H8 cells (8 sample points) and a block of TET4 cells (1 sample point) have the Awkward type `n * var * 4`:

```text
n * var * 4
└──┬──┘   └── component axes
   └── item axes
```

`n` is the number of cells, `var` the sample points of each cell (8 or 1, so this axis is ragged), and `4` the state variables at a point. A type like `n * 4 * var`, where the shape of the value itself varies, is not allowed. Because of this rule, a ragged result is always offsets over one regular array, here of shape `(total number of sample points, 4)`, which Numba kernels can use directly. Within each block there is no ragged axis at all: on the H8 block the field is a plain `(nE, 8, 4)` array.

**Uniform fields:** a field may hold a single value that stands in for every item (a thickness for a whole part). It is expanded only when needed. Uniformity is a form of the stored value, like the forms of frames (section 3.3), not part of the spec: any field can hold a single value, and the next write may give a full array again. There is no flag for it: the shape of the value decides, compared with the axes of the spec:

- **Only the component axes** (`()` for a scalar, `shape` for an array): the value is uniform.
- **All item axes plus the component axes:** one value per item.
- **Anything else** raises. A `(8,)` value for a scalar `"node"` field of H8 cells is neither, so it isn't silently repeated for every cell.

The first write of a user field can be ambiguous, because there is no spec to compare with yet: on a block of 3 cells, a `(3,)` value is either one scalar per cell or a single 3-vector for every cell. Inference takes the narrowest spec, one scalar per cell. A caller who means a uniform vector passes `spec=` on the first write; after that the spec decides.

```python
coords, topo = grid(size=(1, 1, 1), shape=(2, 2, 2), eshape="H8")
frame = CartesianFrame(dim=3)
pd = PointData(x=coords, frame=frame)
cd = H8(topology=topo, frames=frame)
nP, nE = len(coords), len(topo)

# specs inferred from the first write
pd.data["flags"] = np.zeros((nP, 2), dtype=bool)    # array, one value per point, bool, shape (2,)
cd.data["temperature"] = np.full(nE, 20.0)          # scalar, cell, float64
cd.data.set("nodal_t", np.zeros((nE, 8)), location="node")   # scalar, node: H8 node order
pd.data.set("t", np.zeros(nP), location="node")     # ValueError: point data has no locations

# later writes are validated against the spec, as for core fields
cd.data["temperature"] = np.zeros(nE + 1)           # ValueError: one value per cell expected
pd.data["flags"] = np.zeros((nP, 3), dtype=bool)    # ValueError: shape (3,), the spec says (2,)

# uniform: one value stands in for every cell
cd.data["t_ref"] = 20.0                             # inferred spec: scalar, cell, float64
cd.data.get("t_ref").shape                          # (nE,): expanded on demand, not in storage
cd.data.spec("t_ref")                               # FieldSpec(kind='scalar', location='cell', ...)

# values at points given in master coordinates, in the order given
g = 1 / np.sqrt(3)
gauss8 = np.array(np.meshgrid([-g, g], [-g, g], [-g, g], indexing="ij")).reshape(3, -1).T   # (8, 3)
cd.data.set("state", np.zeros((nE, 8, 4)), location="master", at=gauss8)
cd.data.set("state_reduced", np.zeros((nE, 1, 4)), location="master", at=np.zeros((1, 3)))
cd.data["state"] = np.zeros((nE, 4, 4))             # ValueError: 8 sample points per H8 expected

# a spec given explicitly: the same FieldSpec a package uses in its Schema
spec = FieldSpec(kind="array", location="master", at=gauss8.tolist(), shape=(4,), dtype="float64")
cd.data.set("state_f64", np.zeros((nE, 8, 4), dtype="float32"), spec=spec)   # stored as float64

# from physical coordinates to master coordinates
x = np.array([[0.25, 0.25, 0.25], [0.75, 0.5, 0.5]])   # points in the frame of the point cloud
cells, xm = cd.to_master(x, frame=frame)             # containing cells (2,) and master coordinates (2, 3)

v = Vector(np.zeros((nE, 3)), frame=frame)
cd.data["v"] = v                                    # TypeError: no vector or tensor fields yet (section 3.14)
```

### 3.3 Frames

Every block keeps its frames, as today: the frame of the point cloud, and the frames of a cell block (e.g. beam orientations, the local directions of shells). No field depends on them for now, because no field is stored in a frame (section 3.2).

- Every cell block has frames. A block created without them gets a single set of axes equal to the frame of its point cloud, so every block is handled the same way.
- The frames are stored in one of three forms, and none of them is expanded to (nE, 3, 3) (section 2.4):
  - **shared:** a single set of axes for all cells,
  - **per cell:** stacked axes of shape (nE, dim, dim),
  - **indexed:** k distinct frames plus an integer index per cell, e.g. 10⁵ beams with 3 orientations.
- Frames are core state of their block and are stored with it, like the topology. There is no registry shared between blocks.

```python
# all cells share one frame: one set of axes, nothing expanded
coords, topo = grid(size=(1, 1, 1), shape=(2, 2, 2), eshape="H8")
cd = H8(topology=topo)                                   # the frame of the point cloud

# indexed: 3 orientations for 10^5 beams
nB = 100_000
topo_beams = np.stack([np.arange(nB), np.arange(1, nB + 1)], axis=1)   # (nB, 2)
axes3 = np.stack([np.eye(3), np.eye(3)[[1, 2, 0]], np.eye(3)[[2, 0, 1]]])   # (3, 3, 3)
orientation = np.arange(nB) % 3                      # (nB,): integers in {0, 1, 2}
cd = L2(topology=topo_beams, frames=ReferenceFrame(axes3), frame_index=orientation)
```

**Frames are fixed after construction.**

- Frames are given at construction (e.g. beam orientations), or the cell type computes them.
- After that, `cd.frames` is read-only. It returns a new frame object over a read-only view of the stored axes, so orienting it never changes the block, and `cd.frames = F` raises. The same holds for the frame of the point cloud. No change to `math` is needed for this.
- Nothing in the mesh changes frames afterwards. `PolyData.rotate` is removed (section 3.14): an operation that changes frames has to say what happens to the data that refers to them, and that is the open question of vector and tensor fields.

```python
coords, topo = grid(size=(1, 1, 1), shape=(2, 2, 2), eshape="H8")
frame = CartesianFrame(dim=3)
cd = H8(topology=topo, frames=frame)

F = CartesianFrame(dim=3)
cd.frames = F                                        # AttributeError: frames are read-only
cd.frames.orient("Space", [0, 0, np.pi / 2], "XYZ")  # orients a temporary object only
```

A point cloud has a single frame. Per-node frames, e.g. for skewed supports, are not covered (section 3.14).

### 3.4 Access API

User fields and core fields share one access API. Both always return `ndarray`, and both validate every write against the spec of the field:

```python
class Fields(Mapping):                        # common base of pd.data / cd.data and pd.core / cd.core
    def __getitem__(self, name: str) -> ndarray: ...

    def spec(self, name: str) -> FieldSpec: ...

    def get(self, name: str, default: Any = None, *,
            resolve: bool = True,            # fall back to uniform / cascaded values
            copy: bool = False) -> ndarray | Any: ...

    def set(self, name: str, value: ArrayLike) -> None: ...   # validated against self.spec(name);
                                                              # a value with only component axes is uniform

class UserFields(Fields, MutableMapping):    # pd.data, cd.data
    def set(self, name: str, value: ArrayLike, *,
            spec: FieldSpec | None = None,   # first write only; inferred when omitted
            location: Literal["cell", "node", "master"] | None = None,   # cell data only
            at: ArrayLike | None = None,                                 # with location="master"
            ) -> None: ...                   # location, at: shortcuts for parts of an inferred spec

    def __delitem__(self, name: str) -> None: ...   # removes the field and its spec

class CoreFields(Fields):                    # pd.core, cd.core
    ...                                      # specs come from the Schema; undeclared names raise
```

- **User fields.** `cd.data` is a full `MutableMapping`: `[]`, `in`, `del`, `keys()`, `items()`. The first write of a name creates the field and its spec; every later write is validated against it.
- **Core fields.** Each core field has a generated property (`cd.topology`, `cd.density`). Reading it is the same as `cd.core.get(name)`, assigning to it the same as `cd.core.set(name, value)`. `cd.core` is a view over the declared fields only: setting a name the `Schema` doesn't declare raises, and core fields can't be deleted.
- `get`'s second positional argument stays `default`, so it behaves like `dict.get`.
- `sigmaepsilon.math` objects (`Vector`, `Tensor2`, ...) are rejected by both with a `TypeError` until vector and tensor fields exist (section 3.14).

### 3.5 A nested `Schema` per data class

A data class declares only its own fields in a nested `Schema`. The metaclass of the data classes merges it with the schemas of the base classes and rejects reused names:

```python
class CellData(metaclass=DataMeta):
    class Schema:
        topology = FieldSpec(kind="array", location="cell", dtype="int64")

class FemCellData(CellData):
    class Schema:                               # only the new fields; no base class needed
        C       = FieldSpec(kind="array", location="cell")   # material stiffness, (nE, 6, 6)
        K       = FieldSpec(kind="array", location="cell", persist=False)   # kept by fem, never saved
        density = FieldSpec(kind="scalar", location="cell")

class H8Solid(FemCellData, H8):                 # fem's fields on the H8 cell type (section 3.15)
    label = "H8_Solid"

coords, topo = grid(size=(1, 1, 1), shape=(2, 2, 2), eshape="H8")
frame = CartesianFrame(dim=3)
C = np.tile(np.eye(6), (len(topo), 1, 1))
cd = H8Solid(topology=topo, frames=frame, C=C, density=7850.0)
cd.topology                     # declared by CellData (mesh)
cd.density                      # declared by FemCellData (fem), uniform: one value for every cell
cd.K                            # declared by FemCellData (fem); None until fem's methods fill it (section 3.11)

list(FemCellData.Schema.fields())
# ['topology', 'C', 'K', 'density']
FemCellData.Schema.owner("K")   # FemCellData, package sigmaepsilon.solid.fem
FemCellData.Schema.to_json()    # as stored in a saved file

class Broken(FemCellData):
    class Schema:
        topology = FieldSpec(kind="array", location="cell")   # TypeError: 'topology' is already declared by CellData
```

What `DataMeta` does when a data class is created:

```python
class DataMeta(ABCMeta):
    def __new__(mcs, name, bases, ns):
        declared = ns.pop("Schema", None)
        cls = super().__new__(mcs, name, bases, ns)
        cls.Schema = merge_schemas(cls, bases, declared)  # raises on reused names
        generate_properties(cls)                          # cd.topology, cd.K, ...
        return cls
```

- **Merge.** `cls.Schema` is a new schema class built from the merged schemas of the bases and the declared specs. A subclass without its own `Schema` inherits the merged one unchanged, and inherited fields can't be dropped by accident. This replaces `_attr_map_` and `ABCMeta_AkWrapper`, and removes the merge-order problem of section 2.2.
- **No reuse.** If a declared name is already a field of a base schema, class creation raises a `TypeError` that names the class that declared it first. Changing the spec of an inherited field (dtype, default) counts as reuse too. In the case of section 2.2, `fem` uses the `activity` of `mesh` or names its own flag differently.
- **No clash with the class.** A field name can't shadow an attribute or method of the data class or its bases (e.g. a field called `copy`, or `frames`), because the generated property would hide it.
- **Multiple inheritance.** A field reached through several bases is fine if it comes from the same declaring class (a diamond). Two different declarations of the same name raise. Sibling packages that both extend `CellData` can declare the same name independently; the error only comes when a class combines them.
- Each spec records its **owner**: the class that declared it and the package of that class. Owners are what versioning, migrations and opaque loading are keyed by (section 3.13). They are recorded internally and never appear in field names.
- **Schema version.** Each package that declares fields defines one module-level constant in its root module, e.g. `__schema_version__ = 3` in `sigmaepsilon/solid/fem/__init__.py`. It is a constant of the code, not a runtime setting: the version describes what the code writes and reads. The metaclass walks up the declaring class's `__module__` (`sigmaepsilon.solid.fem.cells.data` → `sigmaepsilon.solid.fem.cells` → `sigmaepsilon.solid.fem`) and takes the first module that defines it. That module is the owning package, and a data class with new fields but no such module above it raises. So the owner is well defined for namespace packages too, and one package can't declare two versions.
- Frames are not part of the `Schema`. `pd.frame` and `cd.frames` are ordinary read-only properties of `PointData` and `CellData`, the only classes that have frames, and the frames are stored with the block in one of the forms of section 3.3. A declarative mechanism for frames would only pay off if packages declared frames of their own, which is left for section 3.14.
- `FieldSpec` as a pydantic model gives JSON serialization for free; pydantic is already in the stack through `sigmaepsilon.math`.
- A saved file stores the merged schema with the owner and schema version of every field (section 3.13).
- User fields have the same `FieldSpec`s, given or inferred at their first write (section 3.2). They form a dynamic schema per block, validated and saved the same way.

### 3.6 Block configuration: attributes, style and metadata

Each block gets `block.config`, one pydantic model that holds everything set on the block apart from its point and cell data. It replaces today's `config` (section 2.5); the change is breaking and ships with a new major version, without a compatibility shim. It has three parts, which differ in who reads them and in how they cascade:

| Part | Holds | Read by | Declared | Cascades |
|---|---|---|---|---|
| `config.attrs` | model properties: `material`, `name`, `description`, a `quadrature` setting of `fem` | the library and packages | typed fields; undeclared keys allowed | per field, as declared |
| `config.style` | presentation: colour, opacity, tubes, overrides per backend | the plotting layer | typed fields; unknown names raise | always, per property |
| `config.meta` | annotations from other tools: the AxisVM model name, a domain id in that model, a Blender object name, settings of an add-on | other tools only; `mesh` never reads it | free-form JSON, namespaced by tool | never |

Only `PolyData` instances are allowed as dict items of the block itself, so typos like `mesh["A"]["materal"] = "steel"` raise an error instead of creating blocks.

Field data cascades too, but by its own rules:

| Kind | Cascades? | Why |
|---|---|---|
| Point/cell field arrays | never | The parent's lengths have nothing to do with the child's. |
| Uniform fields (`t`, `A`, `density`) and frames (`frame`, `frames`) | yes, as a fallback | A single value or frame stands in for a whole array. |

Reading a cell field like `t` follows the chain: a per-cell array, then a uniform value in the block's cell data, then the attribute of the same name in `config.attrs`, resolved like any cascading attribute (the block, its parent, and so on up to the root), then the default of the field. The inherited value lives in the configuration because the blocks above usually have no cells, but every block has a configuration: `mesh["Structure"].config.attrs.t = 0.012` gives a thickness to every 2D block below that doesn't set its own.

**The models.** All parts are pydantic models with `validate_assignment=True`, like the `Config` of `sigmaepsilon.solid.fourier`: values are validated when they are set, and the whole configuration round-trips through JSON. A block class names its configuration model in `config_class`. Subclasses of blocks and packages extend it by subclassing the models, so inheritance replaces a separate spec mechanism for merging:

```python
class Attrs(BaseModel, validate_assignment=True, extra="allow"):   # undeclared keys allowed, never cascade
    material: Annotated[str | None, Cascade()] = None
    name: str | None = None
    description: str | None = None               # free text about the block, never cascades
    t: Annotated[float | None, Cascade()] = None   # fallback of the field t of 2D cells (section 3.9)
    A: Annotated[float | None, Cascade()] = None   # fallback of the field A of 1D cells (section 3.9)

class Style(BaseModel, validate_assignment=True, extra="forbid"):  # unknown names raise
    # the cells themselves: the line of a 1d cell, the surface of a 2d or 3d cell
    color: str | None = None                     # None: the renderer chooses
    opacity: float = 1.0
    visible: bool = True
    cmap: str | None = None
    # the edges of the mesh drawn over 2d and 3d cells
    show_edges: bool = True
    edge_color: str | None = None
    edge_width: float | None = None
    # 1d cells
    line_width: float | None = None
    tube: bool = False                           # lines drawn as tubes
    # nodes
    point_size: float | None = None
    backend_options: dict[str, dict[str, Any]] = {}   # raw keyword arguments per backend

class BlockConfig(BaseModel, validate_assignment=True):
    attrs: Attrs = Field(default_factory=Attrs)
    style: Style = Field(default_factory=Style)           # neutral, used by every backend
    backend_styles: dict[str, Style] = {}                 # overrides per backend, e.g. {"matplotlib": Style(line_width=1.0)}
    named_styles: dict[str, Style] = {}                   # alternative styles, e.g. for print
    meta: dict[str, dict[str, Any]] = {}                  # namespaced by tool, never read by mesh

class PolyData:
    config_class = BlockConfig

# a package adds new style properties, never changed defaults (those are presets, section 3.10)
class BeamStyle(Style):
    show_section: bool = False                   # draw the cross-section along the axis

class BeamConfig(BlockConfig):
    style: BeamStyle = Field(default_factory=BeamStyle)
    backend_styles: dict[str, BeamStyle] = {}
    named_styles: dict[str, BeamStyle] = {}

class BeamMesh(LineData):                        # in fem
    config_class = BeamConfig
```

**Local values.** A block's configuration holds only what was set on that block. Pydantic records explicitly set fields in `model_fields_set`, so "set to the default value" and "not set" stay distinct (also for `None`), and only set values are saved (section 3.12). Class defaults are never copied into instances as set values, so a changed default reaches existing objects.

**Attributes.**

- A field marked `Cascade()` is looked up on the block itself, then on its parent, and so on up to the root, then in its default. Any other field, and any undeclared key, is taken from the block itself only (a `name` or a `description` shouldn't be inherited).
- Packages add attributes by subclassing `Attrs`, e.g. `fem` declaring `quadrature`. A name already declared by a base model raises, as for schemas (section 3.5).
- **Fallbacks of fields.** A cascading attribute with the name of a scalar cell field is that field's inherited value, as described above: `t` and `A` of `mesh`, or a `density` that `fem` declares in its `Attrs` subclass. The value is validated against the spec of the field when it is read (dtype, safe casts only). Undeclared keys never cascade, so a user field has no inherited value; a user who needs one declares the attribute in a subclass.
- `block.resolved_attrs()` returns a new `Attrs` with the resolved values.

**Style.** Plotting is the only place where the library uses per-block configuration today. `config.style` holds backend-neutral properties that say how a block looks. The backend is not part of the style: it is chosen per plot call (section 3.10), because it is a decision about the output (an interactive view or a print figure), not a property of the model. How each property becomes a keyword argument of that backend is the job of the plotting layer.

The same number doesn't mean the same thing in every backend: a line width that suits PyVista is far too thick in matplotlib, which measures it in points. So a block can override properties for one backend in `config.backend_styles[backend]`, a partial `Style` like the default one. These overrides are what lets a user fine-tune a plot level by level, e.g. thinner lines in matplotlib for a whole frame, and thicker ones for its columns.

Every style property cascades. Resolution is per property, for the backend B chosen by the plot call; what is set closest to the block wins, and cascading only provides defaults for the properties the block leaves open. A property is looked up:

1. on the block itself: in `backend_styles[B]`, then in `style`;
2. on its parent, then its grandparent, and so on up to the root, checking both in the same order at each level (local values only);
3. in the preset of B for the block's cells (section 3.10);
4. in the default of the base `Style`.

So a setting made on a part reaches every block below it that leaves the property open, nested `LineData` blocks included, and the nearest block that sets it wins.

- Within one block, the value for the backend beats the neutral one. Across blocks, the nearer block wins, so a neutral value set on a child beats a value for the backend set on its parent. Otherwise a subtree couldn't be overridden with a neutral setting.
- **Presets come last.** The defaults that differ between kinds of cells and between backends (lines black and drawn as tubes in PyVista, a line width in points in matplotlib) are presets of the plotting layer, not class defaults of style models. They come after the ancestors because anything placed before them would stop a setting on a parent from reaching the blocks below, which is the bug of section 2.5 again. The price: a colour set on a parent now also reaches the line blocks below it. To give lines a different colour, set it on the line blocks.
- **One style for every kind of cells.** A block above the leaves has no dimension of its own: `mesh["A"]` may hold both H8 solids and L2 beams, and its style has to reach both. So there is one `Style` for every block, not one per dimension. Each property is named after the drawn element it applies to, so that it means the same thing on every block: `line_width` is the width of 1d cells and `edge_width` the width of the mesh edges drawn over 2d and 3d cells. Setting `line_width` on a mixed part changes its beams and leaves the edges of its solids alone. A renderer uses the properties of the elements it draws and ignores the rest (section 3.10).
- Style models are subclassed only to add properties (e.g. a package's style for beams), never to change defaults. A parent's style may be such a subclass; a child whose style model doesn't have the property ignores it.
- **Named styles** hold alternative settings, e.g. for print. `config.named_styles["print"]` is looked up along the chain first; a property no block sets there is resolved as in the default style, overrides per backend and presets included.
- **Backend options** are an escape hatch for keyword arguments with no neutral property: `config.style.backend_options["pyvista"] = {"specular": 0.5}`. They cascade per option like properties, are saved as they are without validation, and are passed to that backend only, after the translated properties. They are clearly not portable. Overrides in `backend_styles`, in contrast, are neutral properties, validated like any other.
- `backend_styles` is saved with the mesh (section 3.12). It holds the user's tuning and only applies when that backend is chosen, so it doesn't tie a saved mesh to a backend; a key that names no registered backend is kept and ignored.
- `block.resolved_style(backend=None, name="default")` returns a **new** `Style`, so the plotting layer may change it freely; the stored style never changes while plotting (section 2.5). `backend=None` means the backend `plot` would choose for the block.

**Metadata.** `config.meta` is for data the mesh library carries but never interprets: where a block came from and what other tools need to know about it.

- Keys are namespaced by tool, `meta["axisvm"]["domain_id"]`, so two tools can't overwrite each other's entries.
- Values are any JSON. Nothing is validated, nothing cascades, and it is saved verbatim.
- A setting that `mesh` or a package acts on doesn't belong here: it is an attribute (with a declared field) or a style property. Otherwise the same setting ends up in two places.

```python
coords, topo = grid(size=(1, 1, 1), shape=(2, 2, 2), eshape="H8")
frame = CartesianFrame(dim=3)
mesh = PolyData(pointdata=PointData(x=coords, frame=frame))
mesh["A"] = PolyData()
mesh["A"]["Solids"] = PolyData(celldata=H8(topology=topo, frames=frame))
mesh["A"]["Beams"] = LineData(celldata=L2(topology=topo[:, :2], frames=frame))   # one edge per H8, as L2 cells

# attributes
mesh["A"].config.attrs.material = "steel"
mesh["A"].config.attrs.name = "portal"
solids = mesh["A"]["Solids"]
solids.config.attrs.model_fields_set            # set(): nothing set on this block
solids.resolved_attrs().material                # 'steel': inherited from mesh["A"]
solids.resolved_attrs().name                    # None: "name" doesn't cascade
mesh["A"].config.attrs.description = "Portal frame of the main hall, columns fixed at the base."
solids.config.attrs.material = None
solids.resolved_attrs().material                # None: set explicitly, so not inherited

# style
mesh.config.style.opacity = 0.5
mesh["A"].config.style.color = "red"
lines = mesh["A"]["Beams"]
mesh["A"]["Solids"].resolved_style().color      # 'red': inherited from mesh["A"]
lines.resolved_style().color                    # 'red': inherited too, it comes before the preset
lines.resolved_style().opacity                  # 0.5: inherited from mesh
lines.config.style.color = "blue"
lines.resolved_style().color                    # 'blue': set on the block itself

# overrides per backend
mesh["A"].config.backend_styles["matplotlib"] = Style(line_width=1.0)
lines.resolved_style("matplotlib").line_width   # 1.0: inherited from mesh["A"], for matplotlib
lines.resolved_style("pyvista").line_width      # the PyVista preset for lines: nothing set for PyVista
lines.config.style.line_width = 3.0
lines.resolved_style("matplotlib").line_width   # 3.0: a neutral value on the block beats its parent's override

mesh["A"].config.style.edge_width = 0.5         # the mesh edges of the solids; the lines keep their width

mesh["A"].config.named_styles["print"] = Style(opacity=1.0)
mesh["A"]["Solids"].resolved_style(name="print").opacity   # 1.0; the rest comes from the default style

mesh.config.style.colour = "red"                # ValidationError: unknown style property
mesh.config.style.opacity = "half"              # ValidationError: not a float

# metadata from another tool
mesh["A"].config.meta["axisvm"] = {"model": "bridge.axs", "domain_id": 12}
mesh["A"]["Solids"].config.meta                 # {}: metadata never cascades

mesh["A"]["materal"] = "steel"                  # TypeError: only PolyData blocks are dict items
```

### 3.7 Ids

Global ids are computed from per-block offsets when needed and cached when the mesh is locked (`lock()`). A block's data no longer changes because it moved in the tree.

```python
coords, topo = grid(size=(1, 1, 1), shape=(2, 2, 2), eshape="H8")
frame = CartesianFrame(dim=3)
mesh = PolyData(pointdata=PointData(x=coords, frame=frame))
mesh["A"] = PolyData()
mesh["B"] = PolyData()
mesh["A"]["Part1"] = PolyData(celldata=H8(topology=topo, frames=frame))

part = mesh["A"].pop("Part1")
mesh["B"]["Part1"] = part       # no re-indexing: the stored arrays are untouched
part.celldata.id                # global ids: the block's offset + local index

mesh.lock()
part.celldata.id                # computed once, cached until the mesh is unlocked
```

### 3.8 Cell types and ragged results

The data of a block is always regular; only results that span blocks can be ragged:

- Each field is its own column, not part of one `ak.zip` record, so the union/optional types on export disappear.
- Every stored field is a NumPy array (zero-copy), saved as one dataset.
- Ragged arrays appear only as results of the subtree methods of section 3.9, e.g. `topology()` over blocks of H8 and L2 cells. They are `sigmaepsilon.math.linalg.sparse.JaggedArray` (Awkward-backed), so the ecosystem has one ragged type: offsets over regular trailing axes, with no optional or union types. Numba kernels receive `(offsets, flat_values)`.

**Decided:** one cell type per cell block. A `CellData` holds cells of a single type, so its `topology` is a regular `(nE, nNE)` array that kernels and FEM elements can use directly. A mesh with several cell types is a tree with one block per type. Questions about cells across blocks are answered by the subtree methods of section 3.9, without merging the blocks.

Merging the tree into one mixed-type cell block (a `celltype` column, ragged topology, a block table for the configuration of the blocks, `from_single_block` to rebuild the tree) is **not part of this proposal**. No engineering use needs it yet, and it would bring mixed-type `CellData`, grouped same-type views and validity masks for fields that some blocks lack. The block tree stays the source of truth, so a lossless single-block round trip can be added later without breaking code or files.

Interop formats that want one flat grid (VTK unstructured grids, PyVista, meshio) get a **one-way export**, as `to_vtk` and `to_pv` do today. The export is lossy by design: it writes the cell types, the topology in global point ids and the fields that every block has, and it doesn't come back as a tree. The saved format of section 3.12 is the lossless one.

```python
coords, topo = grid(size=(1, 1, 1), shape=(2, 2, 2), eshape="H8")
coords_tet, topo_tet = H8_to_TET4(coords, topo)
frame = CartesianFrame(dim=3)
mesh = PolyData()
mesh["Hexa"] = PolyData(pointdata=PointData(x=coords, frame=frame), celldata=H8(topology=topo, frames=frame))
mesh["Tetra"] = PolyData(pointdata=PointData(x=coords_tet, frame=frame), celldata=TET4(topology=topo_tet, frames=frame))

mesh["Hexa"].celldata.topology.shape     # (8, 8): regular, one cell type per block
mesh.celltypes()                         # per cell: "H8", ..., "TET4", ... (section 3.9)
mesh.topology()                          # JaggedArray of global point ids (section 3.9)
grid_pv = mesh.to_pv()                   # one-way: a flat unstructured grid for PyVista
```

### 3.9 Methods on a subtree

Every block answers the questions about cells, whether it holds cells itself or not. `mesh["Structure"]` in section 3.16 holds no cells, but `mesh["Structure"].id` returns the ids of all the cells below it, `mesh["Structure"].topology()` their topology and `mesh["Structure"].volume()` their total volume. A leaf block is the case of a subtree with one cell block, so code written against a block works at any depth.

**Decided:** the cell methods of `PolyData` cascade over the subtree, i.e. over `self.cellblocks(inclusive=True)`. The rules:

- **One order.** The blocks are visited in tree order (depth first, insertion order), the same order for every method. Per-cell results line up row by row: `id[i]`, `topology()[i]`, `celltypes()[i]` and `volumes()[i]` describe the same cell. A per-cell method returns one row per cell and never skips blocks. Where a value doesn't apply to a cell type (the `areas()` of an H8), the row holds `NaN`, and the reductions ignore it.
- **Cell types are a column.** `celltypes()` returns one label per cell, so a result that spans blocks of different types can still be split by type.
- **Missing fields are the caller's choice.** `cell_field(name, missing="raise", fill_value=None)` governs blocks of the subtree that lack the field:
  - `missing="raise"` (default): raises, naming the first block without the field. A narrower subtree, or the filter arguments of `cellblocks()`, select the blocks that have it.
  - `missing="fill"`: the cells of those blocks get `fill_value`, or the default of the field's spec when it is `None` (`NaN` for floats). The value must cast safely to the field's dtype, so an integer field needs an explicit `fill_value`. The shape of a filled row follows the spec, e.g. one value per node for a `"node"` field.

  Either way the blocks that have the field must agree on its spec (kind, location, dtype, component shape); otherwise the call raises. There are no validity masks: which cells were filled follows from the blocks, through `cellblocks()`.
- **Topology is ragged in general.** H8 cells have 8 nodes and L2 cells 2, so `mesh["Structure"].topology()` can't be a 2D array. `PolyData.topology()` always returns a `JaggedArray`, even when every cell has the same number of nodes, so that the return type doesn't depend on the data; `to_numpy()` converts a regular one. Kernels that need regular arrays work per block on `celldata.topology`, which is always regular (section 3.8).
- **Global point ids.** A stored topology holds indices into the point data of its block's source (section 3.7: nothing is re-indexed when a block moves). `PolyData.topology()` adds each source's point offset, so the result is in global point ids, consistent across a subtree that spans several point clouds (the `Hexa` / `Tetra` example of section 3.8). It indexes `self.root.coords()`; `detach()` gives a standalone sub-mesh with its own numbering.
- **Downward, not upward.** Methods cascade down the tree, while attributes and style resolve up it (section 3.6). `pointdata` and `celldata` don't cascade: they are the block's own data holders and `None` on a block without data.
- **Filters and caching.** Every cascading method takes the filter arguments of `cellblocks()`. Results are computed on demand and cached while the mesh is locked, as ids are (section 3.7).

| Method | Result over a subtree of `nE` cells | Per block |
|---|---|---|
| `id` | `(nE,)` global cell ids | `celldata.id` |
| `topology()` | `JaggedArray` of global point ids | `celldata.topology` plus the point offset of the source |
| `celltypes()` | `(nE,)` labels | `celldata.label`, repeated for each cell |
| `lengths()`, `areas()` | `(nE,)`, `NaN` where not defined | the cell type's method |
| `volumes()` | `(nE,)`, defined for every cell; raises if an input is missing (below) | the cell type's method |
| `centers()`, `cells_coords()` | `(nE, 3)`, ragged `(nE, var, 3)` | the cell type's method |
| `cell_field(name, missing="raise", fill_value=None)` | `ndarray` / `JaggedArray`; raises or fills where a block lacks the field | `celldata.core` or `celldata.data` |
| `number_of_cells()`, `length()`, `area()`, `volume()` | scalar | sum of the per-cell results |

**Volumes per cell kind.** Every cell has a volume, but what it needs depends on the dimension of its cell type:

| Cell kind | Volume | Needs |
|---|---|---|
| 3D (`H8`, `TET4`, ...) | integral over the cell | the coordinates |
| 2D (`T3`, `Q4`, ...) | area × thickness | the coordinates and `t` |
| 1D (`L2`, `L3`, ...) | length × cross-section area | the coordinates and `A` |

- `t` and `A` are core fields declared by `mesh` on the bases `CellData2d` and `CellData1d` (section 3.16): scalars per cell, `float64`, with no default. They can be uniform, and are read through the chain of section 3.6: a per-cell array, a uniform value in the block's cell data, then the cascading attribute `t` or `A` in the configuration of the block or a block above it. So one thickness set on a parent block serves every 2D block below it.
- If the chain ends without a value (the fields have no default), `volumes()` raises a `ValueError` naming the block and the field. There is no fallback: today a missing thickness or area silently counts as 1.0 (section 2.9).
- Over a subtree, `volumes()` and `volume()` raise at the first block that lacks its input. A caller that only wants the solids narrows the subtree.
- `lengths()` and `areas()` need only the coordinates. `areas()` is the surface area of 2D cells, `NaN` for 1D and 3D cells; the cross-section area of a 1D cell is the field `A`, not a result of `areas()`.
- The field is called `A`, not `area`, because the name of a field can't shadow a method of the class (section 3.5), and cell blocks have `area()`.

Each method is the per-block method plus one shared walk, so a cell type only implements the block version. A package that adds a per-block method (e.g. `fem`'s element masses) registers it as cascading with a decorator instead of writing its own walk:

```python
class PolyData(...):
    @cascade(reduce="concat")
    def volumes(self) -> ndarray: ...      # body: self.celldata.volumes()

    @cascade(reduce="sum")
    def volume(self) -> float: ...         # body: self.celldata.volumes()
```

`cell_ids()` stays as a warning alias of `id`.

### 3.10 Extensibility

- **Schemas merged by the metaclass** (sections 3.1 and 3.5) are the main extension mechanism. A package adds fields by subclassing a data class and declaring only the new ones.
- **Storage and behaviour are separated inside the class, not by layers of wrappers** (section 3.15). A cell type is a subclass of `CellData`: the values live in `core` and `data`, views over private storage; the cell type in the nested `Geometry`; the declared fields in the nested `Schema`; and the behaviour in methods. A package extends a cell type by subclassing it and declaring its fields, so `fem`'s elements are `CellData` subclasses too.
- **Class registry.** Cell types and container classes are registered by label when the class is created, which loading needs anyway: cell types as today (`"H8"`), and subclasses of `PolyData` the same way (`"PolyData"`, `"LineData"`). Each class records a `base_label`, the label of its nearest base class that has one (e.g. `H8_Solid` → `H8`, `FemMesh` → `PolyData`), so a file still loads as a plain mesh when the extending package isn't installed. Labels are unqualified, like field names. Cell types and containers have separate namespaces, and registering a label twice in one of them raises. `label` is only this class attribute: `config.attrs` has no `label`, and a block is named by `attrs.name` (section 3.6).
- **Entry point.** A Python entry point (group `sigmaepsilon.mesh.classes`) lets a saved file open `fem`'s elements and containers without importing `fem` by hand. Its value is a module; loading it imports the module, which creates and so registers the classes. `mesh` reads the group lazily, only when a file holds a label that isn't registered, so importing `mesh` or loading a plain mesh never imports other packages.
- The tree events `DeepDict` already provides (`__after_join_parent__` etc.) remain the extension hooks.

What this could look like in `solid.fem`:

```python
class FemCellData(CellData):                   # fem's fields, shared by all its elements
    class Schema:
        C = FieldSpec(kind="array", location="cell", shape=(6, 6))

class H8Solid(FemCellData, H8):                # Geometry from H8; registered as "H8_Solid"
    label = "H8_Solid"                         # base_label "H8", from the inheritance

    def stiffness(self) -> ndarray:
        return element_stiffness(self.pointdata.x, self.topology, self.C)

class FemMesh(PolyData):                       # model-level operations; registered as "FemMesh"
    label = "FemMesh"                          # base_label "PolyData", from the inheritance

    def assemble(self) -> spmatrix: ...
```

```toml
# solid.fem's pyproject.toml: lets a saved file open H8_Solid cells and FemMesh
# containers without importing fem; the module defines and so registers them
[project.entry-points."sigmaepsilon.mesh.classes"]
fem = "sigmaepsilon.solid.fem.classes"
```

**Plotting backends and renderers.** Several backends can draw a mesh, but not every backend can draw every cell: 1d and 2d cells (topological dimension, whatever the dimension of the space they are in) can be drawn with matplotlib or PyVista, 3d cells only with PyVista. One plot call produces one figure, so it uses exactly one backend for the whole subtree.

**Decided:** two backends, PyVista and matplotlib; k3d and Plotly are dropped. PyVista (on top of VTK) is the only general-purpose renderer and the default for every mesh: it draws every cell type, in 2d or 3d space, interactively in notebooks too. matplotlib stays for static, print-quality figures of 1d and 2d cells, used when asked for or when PyVista isn't installed. It draws cells in 3d space on 3d axes (e.g. a `LineData` of a space frame), but without real depth sorting, which is why it isn't the default. Each backend has its own job, and every renderer exists at most twice. Removed:

- k3d: `k3dplot`, `to_k3d`, `plotting/k3dplot.py`, `io/to_k3d.py`, the `__hask3d__` flag, the `("k3d", "default")` config key, and the `k3d` entries of the `test` and `docs` dependency groups. `docs/source/notebooks/plotting.ipynb` switches to PyVista.
- Plotly: `plotting/plotly/` (`plot_lines_plotly`, `scatter_lines_plotly`, `scatter_points_plotly`, `triplot_plotly`), `LineData.__plot_plotly__`, the `__hasplotly__` flag, `tests/plotting/test_plotly.py`, the `plotly` entries of the `test` and `docs` groups, and `sphinx-plotly-directive` in the docs. `LineData.plot` defaults to Plotly today; without a backend named, it gets PyVista.

If either is needed again, it comes back as one more backend with its own renderers, without changes to the rest of the design.

Plotting has three parts:

- A **backend** (`"pyvista"`, `"matplotlib"`) is registered with a name and its availability (the `__has*__` flags of `config.py`). It is chosen once per plot call.
- A **renderer** draws one block inside a backend. Renderers are registered per backend and cell type, and the most specific one wins: the cell type label, then its `base_label` chain, then the topological dimension of the cells. A renderer translates the resolved style (section 3.6) into the keyword arguments of its backend, uses the properties of the elements it draws (`line_width` and `tube` for 1d cells, `show_edges`, `edge_color` and `edge_width` for 2d and 3d cells) and ignores the rest. A backend can draw a block if it has a renderer for it.
- A **preset** is a partial `Style` with the defaults of one backend for one kind of cells, keyed like renderers: by cell type label or by topological dimension. Each property is looked up in the most specific preset that sets it: the cell type label, then its `base_label` chain, then the topological dimension. The library ships presets for both backends (in PyVista, lines black, drawn as tubes, with a width in pixels; in matplotlib, a width in points); packages register their own next to their renderers. Presets are session settings: a user changes them in place, `presets("matplotlib", dim=1).line_width = 0.8`, for every mesh plotted afterwards. They are validated like any style and never saved with a mesh. They come after the whole chain of blocks (section 3.6, step 3), so they only fill in what no block sets.

`mesh.plot(backend=None, style="default", skip_unsupported=False, **kwargs)` chooses the backend:

1. A backend named explicitly is used as is.
2. With `backend=None`, PyVista is used. If PyVista isn't installed, matplotlib is used with a warning if it can draw every cell block of the subtree; otherwise (a mesh with 3d cells) plotting raises an `ImportError` naming the extra to install.

- The backend is never stored on a block, so `mesh["Beams"].plot()` and `mesh.plot()` always use the same backend, and a saved mesh carries no backend preference. What a user tunes per block for a backend lives in `config.backend_styles` (section 3.6) and only applies when that backend is chosen.
- If the chosen backend can't draw some blocks, plotting raises an error naming them; `skip_unsupported=True` draws the rest.
- `mesh.pvplot(...)` stays as a shortcut for `mesh.plot(backend="pyvista", ...)`.
- Keyword arguments of a plot call go to the backend for that call only, for every block of the subtree, after the translated style and the backend options. They are never stored. Settings for single blocks of the subtree belong in their configuration.
- Packages register their own renderers and presets, e.g. `fem` drawing beams with their cross-sections in PyVista, through an entry point like the one for classes (`sigmaepsilon.mesh.renderers`). Where no such renderer exists, the generic renderer of the base cell type is used.

```python
@register_renderer("pyvista", celltype="L2")
class PyVistaLines(Renderer):
    def render(self, block, style: Style, figure, **kwargs):
        params = {"color": style.color, "line_width": style.line_width,
                  "render_lines_as_tubes": style.tube, "opacity": style.opacity}
        params.update(style.backend_options.get("pyvista", {}), **kwargs)
        figure.add_mesh(block.to_pv(), **params)

register_preset("pyvista", dim=1, style=Style(color="k", line_width=10, tube=True))
register_preset("matplotlib", dim=1, style=Style(color="k", line_width=1.5))

coords, topo = grid(size=(1, 1, 1), shape=(2, 2, 2), eshape="H8")
frame = CartesianFrame(dim=3)
mesh = PolyData(pointdata=PointData(x=coords, frame=frame))
mesh["Solids"] = PolyData(celldata=H8(topology=topo, frames=frame))
mesh["Frame"] = LineData()                                  # nested line blocks; groups hold no cells
mesh["Frame"]["Columns"] = LineData()
mesh["Frame"]["Columns"]["Main"] = LineData(celldata=L2(topology=topo[:, :2], frames=frame))
mesh["Frame"]["Columns"]["Braces"] = LineData(celldata=L2(topology=topo[:, 2:4], frames=frame))
mesh["Frame"]["Beams"] = LineData(celldata=L2(topology=topo[:, 4:6], frames=frame))

mesh.plot()                                  # pyvista: the default
mesh["Frame"].plot()                         # pyvista too: the backend is never stored
mesh.pvplot()                                # the same as mesh.plot(backend="pyvista")
mesh.plot(backend="matplotlib")              # error: mesh["Solids"] has 3d cells

# fine-tuning level by level
f = mesh["Frame"]
f.config.style.color = "navy"                                       # every backend, the whole frame
f.config.backend_styles["matplotlib"] = Style(line_width=0.8)       # matplotlib, the whole frame
f["Columns"].config.backend_styles["matplotlib"] = Style(line_width=2.0)
f["Columns"]["Braces"].config.style.color = "gray"                  # every backend, braces only

f.plot(backend="matplotlib")    # main columns and braces 2.0, beams 0.8; braces gray, the rest navy
f.plot()                        # pyvista: the colours apply, widths come from the PyVista preset

presets("matplotlib", dim=1).line_width = 1.0    # session-wide default for lines nobody set

mesh.config.named_styles["print"] = Style(opacity=0.8)
mesh.plot(style="print", opacity=1.0)        # a named style, opacity overridden for this call
```

### 3.11 Computed values

**Decided:** `mesh` doesn't track how values were computed. There are no derived fields, no dependency lists, no provenance and no staleness checks, and reading a field never starts a computation.

- **Computation is explicit.** A package computes through methods of its classes, e.g. an analysis or `stiffness_matrix()` in `fem`, and the user calls them. Which intermediate results a package keeps, and where, is an internal detail of that package, not part of the data model.
- **Values that can go stale aren't saved.** A package that keeps an intermediate result in a field (element matrices, caches) declares it in its `Schema` like any other field, with `persist=False`. Such a field lives only in the session and is never written to a file, so a saved file never holds a stale intermediate result.
- **Saved values are the user's responsibility.** Inputs and the results of an analysis are saved as they are. After changing the materials or the geometry of a block, the user runs the analysis again; otherwise the results saved are those of the old model.
- **Why:** dependency tracking can't be made reliable at a reasonable cost. Writes into returned arrays bypass any write counter, a hand-written list of inputs can miss some (frames and attributes aren't fields at all), and a fix to the computing code leaves the inputs unchanged. Automatic recomputation would also hide long, resource-heavy computations behind a plain read.

```python
class FemCellData(CellData):
    class Schema:
        C = FieldSpec(kind="array", location="cell", shape=(6, 6))    # material stiffness, an input
        K = FieldSpec(kind="array", location="cell", persist=False)   # element stiffness, kept by fem

class H8Solid(FemCellData, H8):
    label = "H8_Solid"

coords, topo = grid(size=(1, 1, 1), shape=(2, 2, 2), eshape="H8")
frame = CartesianFrame(dim=3)
cd = H8Solid(topology=topo, frames=frame, C=np.tile(np.eye(6), (len(topo), 1, 1)))
mesh = FemMesh(pointdata=PointData(x=coords, frame=frame))
mesh["A"] = PolyData(celldata=cd)

cd.K                            # None: nothing has computed it yet
mesh.stiffness_matrix()         # explicit; fem may keep the element matrices in cd.K
mesh.save("model.h5")           # C is saved, K isn't
```

Not saving intermediate results also keeps files small. An H27 element stiffness matrix is 81×81 doubles, about 52 kB per cell, so 10⁵ cells would add about 5 GB to the file.

### 3.12 Persistence

A format-neutral `to_dict()` / `from_dict()` produces a description of the mesh plus its arrays:

- the tree of block paths, with each block's container class (its `label` and `base_label` chain, section 3.10) and its `config` as JSON, local values only (attributes, style with overrides per backend, named styles and backend options, metadata; presets are session settings and aren't saved),
- per block: point and cell fields, core and user alike, with their specs, and the cell type label,
- per block: its frames (class name, frame name, axes, and the index of indexed frames). Fields hold no frame references, so loading only has to rebuild the frames of each block,
- the merged schema of every data class, with the owner of each field and the schema version of each owning package (section 3.13),
- the format marker (`"sigmaepsilon.mesh"`), the format version and package versions.

Backends are built on top of it:

| Backend | Single file | Dependencies | Notes |
|---|---|---|---|
| `.npz` + JSON description | yes | none | One member per field, read when accessed (`np.load` returns a lazy `NpzFile`), but a member is always read whole: no slicing, no memory mapping, and any change rewrites the archive. |
| HDF5 (`h5py`) | yes | optional extra `hdf5` | Groups map onto the tree, metadata goes into attributes. Datasets chunked along the item axis, so a range of cells or points can be read alone; gzip compression per chunk; in-place updates (`upgrade`, appending fields). |
| Zarr (zip store) | yes | optional | The same layout, cloud-friendly. |
| Parquet | no | already used | Tabular, so the tree only fits as one file per block. Keep it as a data export. |
| VTK `.vtm` / XDMF | multi-file | optional | Interop export only: frames and metadata don't survive. |

**The file name chooses the format; there is no default backend.** `save` and `load` pick the backend from the suffix (`.npz`, `.h5` / `.hdf5`, ...). A name without a suffix, or with a suffix no backend is registered for, raises a `ValueError` listing the known suffixes; nothing is guessed and no suffix is appended. Every format is a full round trip of `to_dict()`, so none of them is a lesser fallback, and the documentation only describes what each is good at. A backend whose dependency is missing raises an `ImportError` naming the extra to install (`pip install sigmaepsilon.mesh[hdf5]`); its availability is a flag in `config.py`, like the other optional dependencies.

A backend may register several suffixes for one format (HDF5: `.h5` and `.hdf5`), compared case-insensitively; a suffix belongs to exactly one backend. Because these suffixes are generic (any HDF5 or NumPy archive has them), every file records a format marker, `format = "sigmaepsilon.mesh"`, next to `format_version`, and `load` checks it before reading anything else: a file without it raises a `ValueError` saying it isn't a `sigmaepsilon.mesh` file, rather than failing later on a missing key. Interop formats are read through their own import functions, never through `load`.

Both single-file backends store one entry per field, so the description, the tree and the specs can be read without reading the arrays. A partial-load API (a subtree, a selection of fields, a range of cells) is not part of this proposal; the layout keeps it possible.

Sketch of the HDF5 layout:

```
/                                attrs: format ("sigmaepsilon.mesh"), format_version, packages,
                                        schemas (JSON, with owners)
/blocks/<block path>/            attrs: container ("FemMesh"), container_bases (["PolyData"]),
                                        config (JSON, local values only)
    point/core/x                 attrs: spec (JSON)
    point/frame/axes             attrs: class, name
    point/data/<user field>      attrs: spec (JSON)
    cell/                        attrs: celltype ("H8_Solid"), celltype_bases (["H8"])
    cell/frames/axes             attrs: class, name, form ("shared" | "per_cell" | "indexed")
    cell/frames/index            only for indexed frames
    cell/data/<user field>       attrs: spec (JSON)
    cell/core/topology           regular: one dataset
    cell/core/C                  attrs: spec (owner: fem); K is persist=False and never saved
/opaque/...                      fields whose owning package is missing or newer
```

```python
class FemCellData(CellData):              # in solid.fem
    class Schema:
        C = FieldSpec(kind="array", location="cell", shape=(6, 6))   # material stiffness

class H8Solid(FemCellData, H8):
    label = "H8_Solid"

coords, topo = grid(size=(1, 1, 1), shape=(2, 2, 2), eshape="H8")
frame = CartesianFrame(dim=3)
cd = H8Solid(topology=topo, frames=frame, C=np.tile(np.eye(6), (len(topo), 1, 1)))
cd.data["temperature"] = np.full(len(topo), 20.0)
mesh = FemMesh(pointdata=PointData(x=coords, frame=frame))
mesh["A"] = PolyData()
mesh["A"]["Part1"] = PolyData(celldata=cd)

mesh.save("model.h5")                     # the backend follows the suffix
mesh.save("model.npz")                    # dependency-free
mesh.save("model")                        # ValueError: no suffix, the format must be named
m = PolyData.load("model.h5")

type(m)                                   # FemMesh: the saved class, although PolyData.load was called
cd_loaded = m["A"]["Part1"].celldata
np.allclose(cd_loaded.C, cd.C)            # True: core fields
cd_loaded.data.spec("temperature")        # the inferred spec of the user field, saved with it
type(cd_loaded)                           # H8Solid; both found through solid.fem's entry point

FemMesh.load("model.h5")                  # FemMesh
FemMesh.load("plain.h5")                  # TypeError: the file holds a PolyData, not a FemMesh
PolyData.load("model.h5", resolve="base") # PolyData with H8 cells, fem not imported

d = mesh.to_dict()                        # the format-neutral description and arrays
PolyData.from_dict(d)                     # FemMesh, by the same rules as load()
```

**Classes on load.** A file records the class of every block and every cell block, so a mesh comes back as the classes it was saved with, the way `pickle` restores the class of an object, not the class it was loaded through. Without that, a mesh saved from a package's container would lose that container's methods on load (`fem`'s assembly), and a `LineData` block the style defaults its class declares (section 3.6), because only local values are saved. `cls.load(path)` and `cls.from_dict(d)`:

- return an instance of the saved class of the root if that class is `cls` or a subclass of it. So `PolyData.load` on a file saved from a `FemMesh` returns a `FemMesh` when `fem` is installed; code written against `PolyData` keeps working on it.
- raise a `TypeError` if the saved class isn't a subclass of `cls`: `FemMesh.load` on a plain mesh file, or `LineData.load` on a file saved from a `FemMesh`. Returning something other than what was asked for would be worse.
- resolve every block below the root the same way, block by block, since subclasses such as `LineData` sit at the leaves. Cell blocks are resolved by their cell type label.
- resolve a label that isn't registered through the entry point (section 3.10). If it is still unknown, the `base_label` chain is walked, with a warning (section 3.13). If the class reached for the root no longer satisfies `cls` (`FemMesh.load` without `fem` installed), loading raises.
- with `resolve="base"`, use only the classes of `mesh`: every label is mapped to the nearest class along its chain that `mesh` registers, and no entry point is read. This is for viewers and converters that only need the geometry. Fields of other packages are kept opaque, as with a missing package.

Loading a file can import other installed packages through the entry point. They are installed, so they are trusted as much as any import, but it is a side effect of `load` that `resolve="base"` turns off.

`pickle` and `deepcopy` are fixed separately with `__getstate__` / `__setstate__`.

### 3.13 Versioning and migrations

**Three things are versioned separately**, because they change for different reasons and different packages own them:

| What | Owner | Changes when |
|---|---|---|
| `format_version` | `mesh` io | The file layout changes (tree, frames). |
| Schema version per package (`mesh`, `fem`, ...) | the package declaring the fields (their owner), as its `__schema_version__` (section 3.5) | A stored field is renamed, changes type, or changes meaning. |
| Package versions | – | Informational only, recorded for diagnostics. |

Cell types are not versioned. Their conventions (node ordering, master coordinates) are treated as stable; versioning them is a possible future extension if one ever has to change.

The schema version is independent of the package version. Adding an optional field needs no migration. A rename, retype or change of meaning bumps the version and requires one.

**Migrations:**

```python
@migration(package="sigmaepsilon.solid.fem", from_version=2, to_version=3)
def rename_density(block: StoredBlock) -> StoredBlock:
    block.fields.rename("rho", "density")      # only fields owned by fem are visible here
    return block
```

- **Each package ships its own migrations**, for the fields it owns, together with its sample files. `mesh` provides only the machinery: the `@migration` decorator, the stored representation migrations work on, and the runner that chains them on load. So a package that changes its schema ships the migration in the same release, without waiting for a `mesh` release, and packages outside the namespace take part the same way. `mesh` holds no migrations of other packages.
- Migrations work on the **stored representation**: a lazy, dict-like view of the arrays, specs, metadata and frames. They never touch live classes, because classes keep changing and a migration that imports today's `FemCellData` breaks later (the rule used by Alembic and Django migrations).
- They run as a chain (v1→v2→v3), in memory on load. An explicit `sigmaepsilon.mesh.io.upgrade(path)` rewrites the file in place.
- Migrations only go forward. Writing an older format isn't supported.
- A migration sees only the fields its package owns, and a rename must not take a name already used along the hierarchy (the same rule as in section 3.5).
- Each package's migrations are tested against sample files: one small file per past schema version, kept in `tests/data/`.
- User fields never get migrations. They are stored as they are, with their specs.

**Load policy for each owning package:**

| Situation | Default | Options |
|---|---|---|
| Versions match | load | – |
| File is older and a migration path exists | migrate in memory | `migrate=False` → error |
| File is older and no path exists | error, naming the missing step | – |
| File is newer than the code | keep those fields **opaque** and warn | `on_newer="error"` |
| Owning package not installed | keep opaque and warn | `on_missing="error"` |

Opaque fields are preserved unchanged and written back on save, the way protobuf keeps unknown fields. Opening and saving a file with an older or smaller install never loses data. Cell types and containers from a missing package fall back along their `base_label` chain, and the labels in the file are written back unchanged, so the file opens with the package's classes again where it is installed. The attributes a package declares in its `Attrs` subclass load as undeclared keys of the base model (section 3.6).

```python
m = PolyData.load("fem_v2.h5")                    # fem fields migrated v2 → v3 in memory
PolyData.load("fem_v2.h5", migrate=False)         # error: fem schema v2, code expects v3
sigmaepsilon.mesh.io.upgrade("fem_v2.h5")         # rewrites the file as v3

# without solid.fem installed
m = PolyData.load("model.h5")                     # warning: fields owned by fem kept opaque
type(m)                                           # PolyData: FemMesh falls back to its base_label
type(m["A"]["Part1"].celldata)                    # H8: H8_Solid falls back to its base_label
m.save("model.h5")                                # fem fields written back unchanged
```

### 3.14 Deferred: vector and tensor fields, rigid motion

Fields that transform with frames (vectors, tensors, and package quantities such as element matrices), and operations that change frames, are left for a later extension. Too many questions about the transformation mechanisms are open to settle them now:

- **Storage frame.** Are components stored in the frame of the host (the point cloud frame, the cell frames) and transformed on write, or does a field keep a frame of its own?
- **Changing frames.** When an operation changes frames, are the stored components carried (the quantity moves with the body) or re-expressed (the quantity stays the same)? Does the operation decide, the spec of the field (a gravity load shouldn't turn with the part), or both?
- **Rigid motion.** What does rotating a mesh mean for point fields, whose frame (the point cloud frame) doesn't turn, compared to cell fields? Does the point cloud frame turn instead of the coordinates?
- **Reading.** Are `sigmaepsilon.math` objects returned as snapshots or as views, and how are ragged results across blocks returned, since bulk tensors in `math` need a regular `ndarray`?
- **Package kinds** with their own transformation rules, e.g. element matrices in `fem` that transform with a block-diagonal direction-cosine matrix.
- **Symmetric tensors:** an optional compact (Voigt) storage form.
- **Per-node frames**, e.g. for skewed supports, in the same shared, per-item or indexed forms as cell frames.
- **Frames declared by packages**, e.g. material axes in `fem`. Today only `PointData` and `CellData` have frames, as plain read-only properties (section 3.5). If packages can declare frames of their own, a declarative mechanism in the `Schema` comes with them.

Until then:

- Only the kinds `"scalar"` and `"array"` exist, and `sigmaepsilon.math` objects are rejected as field values (section 3.4).
- `PolyData.rotate` (and `cd._rotate_`) is deprecated and then removed. It rotates the coordinates and the cell frames, but leaves open what happens to every piece of data that refers to them.
- Components of a vector or tensor can be stored as an `"array"`. They are plain numbers in a frame chosen by convention, and the mesh never transforms them.

What this proposal keeps so the extension can be added later without breaking files or code: the `kind` of every field is part of its spec and saved with it, the `"master"` location places values anywhere inside a cell, and frames are block state in their three forms, read-only after construction.

### 3.15 Classes and their roles

Today it is hard to see from the classes what holds the data, what holds the configuration, what declares the fields, and what defines the cell type:

- `PolyData` is declared as `DeepDict[Hashable, PDP | Any], Generic[PointDataLike, PolyCellLike]`:
  - `PDP | Any` collapses to `Any`, so the value type says nothing.
  - The generic parameters are never filled in (`TriMesh`, `LineData` and user code don't write `PolyData[...]`), so they are documentation only.
- `PolyData.celldata` doesn't return a `CellData`. A cell block has three layers:
  - `mesh.cd` is a `PolyCell` (`H8`, ...), which holds the cell type and the behaviour.
  - `mesh.cd.db` is the `CellData`.
  - `mesh.cd.db.db` is the Awkward array.
  - `PolyCell.__getattr__` falls through to `db`, which blurs the boundary between the layers.
- `PolyCell` reads as one polygonal cell, but it is a block of many cells.
- Points have no such layer: `PointData` holds the data and has the behaviour. Cells and points are built differently.
- `CellDataProtocol` requires `container` and `pointdata`, so even the data holder is expected to know the tree.
- The type variables are defined three times (`typing/data.py`, `polycell.py`, `celldata.py`), under different names (`PolyDataLike`, `MeshDataLike`) and with different variance.

**Decided:** one class per role, and `PolyCell` is merged into `CellData`. A cell type such as `H8` is a subclass of `CellData`. `PolyData` holds a `pointdata` and a `celldata`, and both are data holders. Concerns are separated by nested members, not by layers of wrappers:

| Concern | Lives in |
|---|---|
| Container / tree | `PolyData` |
| Block configuration: attributes, style, metadata | `PolyData.config` (section 3.6) |
| Cell type: the reference element (shape functions, node order, quadrature, VTK id) | the nested `Geometry` of a cell type (`H8.Geometry`) |
| Declared fields | the nested `Schema` of a data class, merged by `DataMeta` (section 3.5) |
| Stored values | `core` and `data`, views over the private storage of a data holder (sections 3.1, 3.4) |
| Behaviour (Jacobians, `pull`, volumes, element matrices) | methods of `PointData`, `CellData` and the cell types |
| Link to the tree | `container`, the only back-reference; `CellData.pointdata` is derived through `container.source()` |

`PointData` and `CellData` are built the same way. Where they differ, it follows from the model: points have no type (no `Geometry`, no `label`), a point cloud has one shared frame while cell frames come in three forms (section 3.3), and `pull` goes in opposite directions (cell data aggregated at the points, point data distributed to the cells).

**Names:**

| Today | New |
|---|---|
| `PolyCell` | merged into `CellData`; `H8`, `L2`, ... subclass `CellData3d`, `CellData1d`, ... (bases of `CellData` per dimension, which declare the inputs of `volumes()`, section 3.9) |
| `PolyCell.db`, `PolyCell.data_class`, `PolyCell.__getattr__` | removed |
| `AkWrapper.db` (the Awkward array) | private storage behind `core` and `data` |
| `PolyData.pd`, `PolyData.cd` | `PolyData.pointdata`, `PolyData.celldata` |
| `PolyData(pd, cd)`, `PolyData(pd=..., cd=...)` | `PolyData(pointdata=..., celldata=...)` |
| `PointData(coords=...)`, `CellData(topo=...)` | `PointData(x=...)`, `CellData(topology=...)`: constructor keywords are the schema names |
| `PolyData._point_class_` | `PolyData.pointdata_class` |
| `PolyData._point_array_class_` | `PointData.pointcloud_class`; `PolyData.points()` delegates to `pointdata.pointcloud()` |
| `PolyCellGeometry1d/2d/3d` | `CellGeometry1d/2d/3d` |
| `ABCMeta_AkWrapper`, `ABCMeta_PolyCell` | `DataMeta` (section 3.5), which also registers cell types |
| `PolyCellProtocol`, `GeometryProtocol` | `CellDataProtocol`, `CellGeometryProtocol` |
| `PointDataLike`, `CellDataLike`, `PolyCellLike`, `PolyDataLike`, `MeshDataLike` | `PointDataT`, `CellDataT`, `PolyDataT`: defined once in `typing/`, invariant |

The declaration of `PolyData` becomes:

```python
class PolyData(DeepDict[Hashable, "PolyData"], Generic[PointDataT, CellDataT]):
    ...
```

The value type is `PolyData`, because only blocks are allowed as dict items (section 3.6). The generic parameters say what `pointdata` and `celldata` return, e.g. `LineData(PolyData[PointData, L2])`. There are no generics in the other direction: `container` is typed with `PolyDataProtocol`. Those circular parameters are what made the old signatures hard to read.

**Consequences:**

- **One cell type per block** (section 3.8): every cell block is an instance of a concrete cell type with its `Geometry`. The base `CellData` has no `Geometry` and is only a base class.
- **Packages** declare their fields on a `CellData` subclass and combine it with the cell types: `class H8Solid(FemCellData, H8)`. `CellData` is reached through both bases, which the rule of section 3.5 allows (a diamond). The label is set on the class, and `base_label` follows from the inheritance (section 3.10).
- **Containers are registered too.** A package that needs model-level operations subclasses `PolyData` (`fem`'s `FemMesh`), sets a `label`, and gets its `base_label` from the inheritance, as cell types do. Files record the container class of every block, and loading restores it (section 3.12).
- **What is lost:** choosing a data class for a cell type separately (`data_class`). Nothing uses it beyond `fem`'s single shared data class, which becomes a common base class instead.
- **Migration:** `mesh.celldata` returns the same object as today, an `H8` instance, which is now a `CellData`. So most user code keeps working. `PolyCell` stays as an alias, and `.db` (returning the instance itself), `.pd`, `.cd`, `pd=`, `cd=`, `coords=` and `topo=` stay as aliases. All of them warn and are removed in the next major version.

### 3.16 The full picture

The definitions and their use in one place: a mesh with a block of H8 cells and a block of L2 beams that share one point cloud. The definitions are sketches: bodies are left out, and `FieldSpec` and `Style` are abbreviated (sections 3.2 and 3.6 have them in full).

```python
# ============================================================================
# DEFINITIONS
# ============================================================================
from typing import Any, ClassVar, Generic, Hashable, Iterator, Literal, TypeVar
from typing import Annotated, Self

import numpy as np
from numpy import ndarray
from pydantic import BaseModel, Field

from sigmaepsilon.deepdict import DeepDict
from sigmaepsilon.math.linalg.sparse import JaggedArray
from sigmaepsilon.mesh.space import CartesianFrame, PointCloud
from sigmaepsilon.mesh.geometry import CellGeometry1d, CellGeometry3d
from sigmaepsilon.mesh.typing import PolyDataProtocol
from sigmaepsilon.mesh.grid import grid


# ---------------------------------------------------------------- schema
class FieldSpec(BaseModel):                     # abbreviated, see section 3.2
    kind: Literal["scalar", "array"]
    location: Literal["cell", "node", "master"] | None = None   # None for point data
    dtype: str | None = None
    shape: tuple[int, ...] | None = None        # component axes


class DataMeta(type):                           # merges nested Schemas, generates properties
    def __new__(mcs, name, bases, ns): ...      # raises on reused names; registers cell types


# ---------------------------------------------------------------- stored values
class Fields:                                   # view over the private storage of a holder
    def __getitem__(self, name: str) -> ndarray: ...
    def __iter__(self) -> Iterator[str]: ...
    def get(self, name: str, default: Any = None, *, resolve: bool = True) -> Any: ...
    def set(self, name: str, value: Any) -> None: ...
    def spec(self, name: str) -> FieldSpec: ...


class CoreFields(Fields): ...                   # specs from the Schema; undeclared names raise


class UserFields(Fields):                       # specs inferred on first write (or given)
    def set(self, name, value, *, spec=None, location=None, at=None): ...
    def __setitem__(self, name: str, value: Any) -> None: ...
    def __delitem__(self, name: str) -> None: ...


# ---------------------------------------------------------------- data holders
class PointData(metaclass=DataMeta):
    pointcloud_class: ClassVar[type[PointCloud]] = PointCloud

    class Schema:
        x = FieldSpec(kind="array", dtype="float64")             # coordinates, (nP, dim)
        activity = FieldSpec(kind="scalar", dtype="bool")

    core: CoreFields                            # x, activity
    data: UserFields
    container: PolyDataProtocol | None          # set when the block joins a tree
    x: ndarray                                  # generated from the Schema
    activity: ndarray                           # generated from the Schema
    frame: CartesianFrame                       # read-only

    def __init__(self, *, x: ndarray, frame: CartesianFrame | None = None, **core): ...
    @property
    def id(self) -> ndarray: ...                # block offset + local index, not stored
    def pointcloud(self) -> PointCloud: ...
    def pull(self, name: str) -> ndarray: ...   # aggregate a cell field at the points


class CellData(metaclass=DataMeta):             # was PolyCell + CellData
    label: ClassVar[str | None] = None
    Geometry: ClassVar[type]                    # the cell type: reference element

    class Schema:
        topology = FieldSpec(kind="array", location="cell", dtype="int64")
        activity = FieldSpec(kind="scalar", location="cell", dtype="bool")

    core: CoreFields                            # topology, activity
    data: UserFields
    container: PolyDataProtocol | None
    topology: ndarray                           # generated from the Schema
    activity: ndarray                           # generated from the Schema
    frames: CartesianFrame                      # read-only; shared, per cell or indexed

    def __init__(self, *, topology: ndarray, frames: CartesianFrame | None = None, **core): ...
    @property
    def id(self) -> ndarray: ...                # block offset + local index, not stored
    @property
    def pointdata(self) -> PointData: ...       # self.container.source().pointdata
    def pull(self, name: str) -> ndarray: ...   # distribute a point field to the cells


# ---------------------------------------------------------------- cell kinds: inputs of volumes()
class CellData1d(CellData):
    class Schema:
        A = FieldSpec(kind="scalar", location="cell", dtype="float64")   # cross-section area


class CellData2d(CellData):
    class Schema:
        t = FieldSpec(kind="scalar", location="cell", dtype="float64")   # thickness


class CellData3d(CellData): ...                 # the coordinates are enough


# ---------------------------------------------------------------- cell types
class H8(CellData3d):
    label = "H8"

    class Geometry(CellGeometry3d):
        number_of_nodes = 8
        vtk_cell_id = 12
        quadrature = {"full": ..., "geometry": "full"}


class L2(CellData1d):
    label = "L2"

    class Geometry(CellGeometry1d):
        number_of_nodes = 2
        vtk_cell_id = 3
        quadrature = {"full": ..., "geometry": "full"}


# ---------------------------------------------------------------- block config
class Cascade: ...                              # marks an attribute that cascades


class Attrs(BaseModel, validate_assignment=True, extra="allow"):
    material: Annotated[str | None, Cascade()] = None
    t: Annotated[float | None, Cascade()] = None    # fallback of the cell field t
    A: Annotated[float | None, Cascade()] = None    # fallback of the cell field A
    name: str | None = None
    description: str | None = None              # free text, never cascades


class Style(BaseModel, validate_assignment=True, extra="forbid"):   # abbreviated, see section 3.6
    color: str | None = None
    opacity: float = 1.0
    show_edges: bool = True                     # mesh edges of 2d and 3d cells
    edge_width: float | None = None
    line_width: float | None = None             # 1d cells
    tube: bool = False


class BlockConfig(BaseModel, validate_assignment=True):
    attrs: Attrs = Field(default_factory=Attrs)
    style: Style = Field(default_factory=Style)
    backend_styles: dict[str, Style] = {}       # overrides per backend, section 3.6
    named_styles: dict[str, Style] = {}
    meta: dict[str, dict[str, Any]] = {}        # namespaced by tool, never read by mesh


# defaults per backend and kind of cells are presets of the plotting layer, section 3.10
def register_preset(backend: str, *, celltype: str | None = None,
                    dim: int | None = None, style: Style) -> None: ...
def presets(backend: str, *, celltype: str | None = None, dim: int | None = None) -> Style: ...


# ---------------------------------------------------------------- containers
PointDataT = TypeVar("PointDataT", bound=PointData)
CellDataT = TypeVar("CellDataT", bound=CellData)


class PolyData(DeepDict[Hashable, "PolyData"], Generic[PointDataT, CellDataT]):
    label: ClassVar[str] = "PolyData"           # registered on class creation, section 3.10
    pointdata_class: ClassVar[type[PointData]] = PointData
    config_class: ClassVar[type[BlockConfig]] = BlockConfig

    pointdata: PointDataT | None
    celldata: CellDataT | None
    config: BlockConfig

    def __init__(self, *, pointdata: PointDataT | None = None,
                 celldata: CellDataT | None = None): ...
    def __setitem__(self, key: Hashable, value: "PolyData") -> None: ...  # only PolyData
    def source(self) -> "PolyData | None": ...  # nearest block up the tree with point data
    def cellblocks(self) -> Iterator["PolyData"]: ...
    # cascading over cellblocks(inclusive=True), section 3.9
    @property
    def id(self) -> ndarray: ...
    def topology(self) -> JaggedArray: ...       # global point ids
    def celltypes(self) -> ndarray: ...
    def lengths(self) -> ndarray: ...
    def length(self) -> float: ...
    def volumes(self) -> ndarray: ...
    def volume(self) -> float: ...
    def number_of_cells(self) -> int: ...
    def resolved_attrs(self) -> Attrs: ...
    def resolved_style(self, backend: str | None = None, name: str = "default") -> Style: ...
    def plot(self, *, backend: str | None = None, **kwargs) -> Any: ...   # None: PyVista
    def save(self, path: str, **kwargs) -> None: ...
    @classmethod
    def load(cls, path: str, *, resolve: Literal["saved", "base"] = "saved") -> Self: ...
    # returns the saved class if it subclasses cls, section 3.12


class LineData(PolyData[PointData, L2]):
    label = "LineData"                           # base_label "PolyData"


# ============================================================================
# USAGE
# ============================================================================

# ---------------------------------------------------------------- input
# Two H8 cells on a 2 x 1 x 1 box, and L2 beams along the bottom edges,
# all sharing the same 12 points.
coords, topo = grid(size=(2, 1, 1), shape=(2, 1, 1), eshape="H8")   # (12, 3), (2, 8)
bottom = [(0, 1), (1, 2), (2, 3), (3, 0)]
edges = np.vstack([topo[:, [i, j]] for i, j in bottom])
edges = np.unique(np.sort(edges, axis=1), axis=0)                    # (7, 2)
frame = CartesianFrame(dim=3)                    # a reference frame (linalg)

# ---------------------------------------------------------------- data holders
pd = PointData(x=coords, frame=frame)
solids = H8(topology=topo)                       # frames: the point cloud's frame
beams = L2(topology=edges, A=0.01)               # frames: computed line axes; A uniform

# ---------------------------------------------------------------- container tree
mesh = PolyData(pointdata=pd)
mesh["Structure"] = PolyData()                   # config only, no data
mesh["Structure"]["Solids"] = PolyData(celldata=solids)
mesh["Structure"]["Beams"] = LineData(celldata=beams)
mesh["Structure"]["materal"] = "steel"           # TypeError: only PolyData blocks are items

# ---------------------------------------------------------------- links
solids.container is mesh["Structure"]["Solids"]  # True
solids.pointdata is pd                           # True: via container.source()
pd.container is mesh                             # True

# ---------------------------------------------------------------- cell type
H8.label, H8.Geometry.number_of_nodes            # ('H8', 8)
beams.Geometry.vtk_cell_id                       # 3

# ---------------------------------------------------------------- schema
list(PointData.Schema.fields())                  # ['x', 'activity']
list(H8.Schema.fields())                         # ['topology', 'activity']
list(L2.Schema.fields())                         # ['topology', 'activity', 'A']

# ---------------------------------------------------------------- core fields
pd.x.shape                                       # (12, 3)
solids.topology.shape                            # (2, 8)
solids.activity = np.array([True, False])        # validated against the Schema
solids.topology = np.zeros((2, 4), dtype=int)    # ValueError: 8 nodes per H8 expected
solids.core.set("density", 7850.0)               # KeyError: not declared by H8

# ---------------------------------------------------------------- frames
pd.frame                                         # one frame
beams.frames.shape                               # (7, 3, 3): per cell
solids.frames = CartesianFrame(dim=3)            # AttributeError: read-only

# ---------------------------------------------------------------- ids
pd.id                                            # [0 .. 11]
solids.id                                        # [0, 1]
beams.id                                         # [2 .. 8]

# ---------------------------------------------------------------- user fields
nP, nE = len(pd.x), len(solids.topology)
pd.data["temperature"] = np.linspace(20.0, 80.0, nP)
solids.data["damage"] = np.zeros(nE)
solids.data.set("nodal_stress", np.zeros((nE, 8, 6)), location="node")
solids.data["t_ref"] = 20.0                      # one value: uniform
solids.data.get("t_ref").shape                   # (2,): expanded on demand
beams.A.shape                                    # (7,): a uniform core field, expanded on demand
solids.data["topology"] = np.ones(nE)            # fine: separate from the core field
solids.data["damage"] = np.zeros(nE + 1)         # ValueError: one value per cell expected
del solids.data["topology"]

# ---------------------------------------------------------------- pull
solids.pull("temperature").shape                 # (2, 8, 1)
pd.pull("damage").shape                          # (12, 1)

# ---------------------------------------------------------------- traversal
for block in mesh.cellblocks():
    print(block.address, block.celldata.label, len(block.celldata.topology))
# ['Structure', 'Solids'] H8 2
# ['Structure', 'Beams'] L2 7

# ---------------------------------------------------------------- subtree
mesh["Structure"].celldata                       # None: no cells of its own
mesh["Structure"].id                             # [0 .. 8]: Solids, then Beams
mesh["Structure"].topology()                     # JaggedArray: 2 rows of 8, 7 rows of 2
mesh["Structure"].celltypes()                    # ['H8', 'H8', 'L2', ..., 'L2']
mesh["Structure"].volumes().shape                # (9,): one row per id
mesh["Structure"].number_of_cells()              # 9
mesh["Structure"]["Solids"].topology().to_numpy().shape   # (2, 8): a leaf is a subtree too

# ---------------------------------------------------------------- subtree: checks
s = mesh["Structure"]
s["Solids"].volume()                             # 2.0: two unit cubes
s["Solids"].volume() == solids.volumes().sum()   # True: a leaf is its cell block
s.volume() == sum(b.volume() for b in s.values())   # True: a parent sums its children
mesh.volume() == s.volume()                      # True: the root adds no cells of its own
s["Beams"].volume()                              # 0.07: 7 beams of length 1.0, A = 0.01
s.volume()                                       # 2.07
s.volumes()[:2]                                  # [1.0, 1.0]: the H8 rows come first, in tree order
s.lengths()                                      # [nan, nan, 1.0, ..., 1.0]: NaN for the H8 rows
s.length()                                       # 7.0: the reduction ignores the NaN rows
s.length() == s["Beams"].length()                # True: only the beams have lengths
s.number_of_cells() == sum(b.number_of_cells() for b in s.values())   # True: 2 + 7
(s.id[s.celltypes() == "L2"] == beams.id).all()  # True: rows of the subtree line up with the blocks

mesh["Structure"]["Bare"] = LineData(celldata=L2(topology=edges))   # no A
s.length()                                       # 14.0: lengths need the coordinates only
s["Solids"].volume()                             # 2.0: unaffected
s.volume()                                       # ValueError: block ['Structure', 'Bare']: L2 needs 'A'
s.config.attrs.A = 0.02                          # a block without cells still has a configuration
s["Bare"].volume()                               # 0.14: A inherited from mesh["Structure"]
s["Beams"].volume()                              # 0.07: its own uniform A wins
s.volume()                                       # 2.21
del mesh["Structure"]["Bare"]

# ---------------------------------------------------------------- config
mesh["Structure"].config.attrs.material = "steel"
mesh["Structure"].config.attrs.name = "portal"
mesh["Structure"]["Beams"].resolved_attrs().material   # 'steel': cascades
mesh["Structure"]["Beams"].resolved_attrs().name       # None: doesn't cascade

mesh.config.style.opacity = 0.6
mesh["Structure"].config.style.color = "red"
mesh["Structure"]["Solids"].resolved_style().color     # 'red': inherited
mesh["Structure"]["Beams"].resolved_style().color      # 'red': inherited, before the line preset
mesh["Structure"]["Beams"].resolved_style().opacity    # 0.6: inherited from the root
mesh["Structure"].config.backend_styles["matplotlib"] = Style(line_width=0.8)
mesh["Structure"]["Beams"].resolved_style("matplotlib").line_width   # 0.8: for matplotlib only

mesh["Structure"].config.meta["axisvm"] = {"domain_id": 12}
mesh.plot()

# ---------------------------------------------------------------- persistence
mesh.save("portal.h5")
m = PolyData.load("portal.h5")
type(m["Structure"]["Beams"].celldata)           # L2
m["Structure"]["Solids"].celldata.data.spec("damage")
m["Structure"].config.attrs.material             # 'steel'

# ---------------------------------------------------------------- extension
class H8Solid(H8):                               # e.g. in solid.fem
    class Schema:                                # only the new fields
        density = FieldSpec(kind="scalar", location="cell")
        C = FieldSpec(kind="array", location="cell", shape=(6, 6))

list(H8Solid.Schema.fields())                    # ['topology', 'activity', 'density', 'C']
fe = H8Solid(topology=topo, density=7850.0, C=np.tile(np.eye(6), (2, 1, 1)))
fe.Geometry is H8.Geometry                       # True

class Broken(H8):
    class Schema:
        activity = FieldSpec(kind="scalar")      # TypeError: already declared by CellData
```

---

## 4. Decisions taken

| Topic | Decision |
|---|---|
| Field type | `kind` naming what the values are, not a numeric rank. For now only `"scalar"` and `"array"`, neither of which transforms with a frame. |
| Locations | Point data: one value per point, no location to choose. Cell data: `"cell"` (default), `"node"` (the nodes in the library's node order), or `"master"` (sample points given by the user in master coordinates of the block's cell type, in the given order). No quadrature-rule presets for now. Utilities map physical coordinates to master coordinates. |
| Core and user fields | The same schema rules: one `FieldSpec` type, the same kinds and locations, one spec per field fixed once it exists, every write validated against it. They differ only in where the spec comes from (the `Schema`, or the first write), how they are reached, naming and versioning. |
| Access | `cd.data` for user fields; generated properties and `cd.core` for core fields. Both share one API and always return `ndarray`. `get`/`set` with options next to `[]`. |
| Schema | Nested `Schema` class per data class, serializable, merged by the `DataMeta` metaclass. |
| Field names | No package namespaces. A core field name is unique along the class hierarchy; reusing one raises at class creation. Each field records its owning package for versioning. |
| Block configuration | `block.config`, a pydantic model with `validate_assignment=True`, replaces today's `config` in a new major version, without a shim. It holds `attrs` (model properties, read by the library, cascading per declared field), `style` (presentation, always cascading) and `meta` (annotations of other tools, namespaced by tool, never read by `mesh`, never cascading). Local values only, through `model_fields_set`; blocks and packages extend the models by subclassing. |
| Style | One `Style` for every kind of cells, with properties named after the drawn element (`line_width` for 1d cells, `edge_width` for mesh edges), so each means the same on every block. Backend-neutral properties, with optional overrides per backend on any block (`backend_styles`). Per property, for the chosen backend: the block's own override, then its neutral value, then the same on each ancestor up to the root, then the backend's preset for the block's cells, then the default of the base `Style`. The nearest block wins, so settings reach nested blocks. Named styles, and backend options as an opaque escape hatch. |
| Plotting | One backend per plot call, never stored on a block: the one named, else PyVista (matplotlib with a warning when PyVista isn't installed and it can draw every block). Renderers per backend and cell type translate the style; presets per backend and cell type hold the shipped defaults, adjustable per session. Backends: PyVista (general-purpose, the default for every mesh) and matplotlib (1d and 2d cells, on request or without PyVista); k3d and Plotly are dropped. |
| Cell types | One cell type per cell block, with a regular topology. No mixed-type blocks and no single-block round trip for now; a lossy one-way export to VTK, PyVista and meshio. |
| Methods on a subtree | Cell methods of `PolyData` (`id`, `topology()`, `celltypes()`, `volumes()`, `volume()`, ...) cascade over the cell blocks below, in one tree order, one row per cell. `topology()` is always a `JaggedArray` of global point ids; for a block that lacks the field, `cell_field()` raises by default or fills with `missing="fill"`. |
| Computed values | No dependency or staleness tracking in `mesh`, and no computation on read. Packages compute through explicit methods and decide which intermediate results to keep; fields that can go stale are declared with `persist=False` and never saved. Keeping saved results consistent with their inputs is the user's responsibility. |
| File formats | No default backend: the suffix of the file name chooses the format (HDF5: `.h5` or `.hdf5`), and a missing or unknown suffix raises. Files carry a format marker that `load` checks first. `.npz` needs no dependency; HDF5 is the optional extra `hdf5` (`h5py`), chunked and compressed, with in-place updates. Both store one entry per field and round-trip everything. |
| Migrations | Each package ships its own migrations, for the fields it owns, and its own sample files, one per past schema version. `mesh` provides the decorator, the stored representation and the runner, but no migrations of other packages. |
| Classes on load | Cell types and `PolyData` subclasses are registered by label, with a `base_label` chain. Files record the class of every block and cell block, and `cls.load()` returns the saved classes (`PolyData.load` gives a `FemMesh`), raises if the saved root class isn't a subclass of `cls`, finds unregistered labels through the entry point `sigmaepsilon.mesh.classes`, and falls back along the `base_label` chain when a package is missing. `resolve="base"` uses only the classes of `mesh`. `label` means only this class attribute; a block is named by `attrs.name`. |
| Frames | Core state of each block, in shared, per-cell or indexed form; every cell block has frames (default: the point cloud frame). Given at construction, read-only afterwards. |
| Vector and tensor fields | Deferred, with everything about transformations (section 3.14). |
| Rigid motion | `mesh.rotate` is removed; it comes back, if at all, with vector and tensor fields. |
| Classes and names | One role per class (section 3.15). `PolyCell` is merged into `CellData`, and cell types subclass `CellData`. `PolyData` holds `pointdata` and `celldata`, both data holders, and `config`. The cell type is the nested `Geometry`, the declared fields the nested `Schema`, the stored values `core` and `data`. `container` is the only back-reference. Old names stay as warning aliases until the next major version. |

---

## 5. Open decisions

- **D1.** Deferred with vector and tensor fields (section 3.14): which fields move with the body.
- **D2.** Withdrawn.
- **D3.** Withdrawn: the data of a block is always regular (section 3.8).
- **D4.** Withdrawn: each package ships its own migrations and sample files (section 3.13).
- **D5.** Withdrawn: `mesh` doesn't track staleness; values that can go stale aren't saved (section 3.11).
- **D6.** Deferred with vector and tensor fields (section 3.14): compact storage of symmetric tensors.
- **D7.** Withdrawn: there is no default backend; the suffix of the file name chooses the format, and a name without one raises (section 3.12).
- **D8.** Withdrawn: frames are read-only after construction (section 3.3).
- **D9.** Deferred with vector and tensor fields (section 3.14): per-node frames.
- **D10.** Withdrawn: `label` means only the class attribute of the registry (section 3.10), which also restores the classes of blocks and cell blocks on load (section 3.12). `config.attrs` has no `label`; a block is named by `attrs.name` (section 3.6).

---

## 6. Suggested order of work

1. Fix the bugs in section 2.8 and pickling; add round-trip tests.
2. Merge `PolyCell` into `CellData` and rename the classes, attributes and constructor keywords (section 3.15), with warning aliases for the old names. The later steps build on this class layout, and `DataMeta` (step 3) replaces both metaclasses merged here.
3. Add `.data`, the nested `Schema` and the `DataMeta` metaclass, with a deprecation shim for the old string-keyed access and `_attr_map_`.
4. Add fields with `FieldSpec`s and validation, shared by core and user fields, frames as read-only block state in their three forms, and deprecate `PolyData.rotate`.
5. Add `to_dict` / `from_dict` with the npz and HDF5 backends, versioning and migrations.
6. Port `solid.fem` to schemas.
7. Replace the zipped Awkward record with per-field columns, hidden behind steps 3–5. Route the cell methods of `PolyData` through one cascading walk (section 3.9), which fixes the misaligned results of section 2.9.
8. Replace `config` with the new `block.config` (attributes, style, metadata; a breaking change for the next major version), add the backend, renderer and preset registries and `plot(backend=None)`, turn `pvplot` into a shortcut, and remove k3d and Plotly (section 3.10). This step is independent of steps 4–7 and can come early; the bugs of section 2.5 are fixed by it. The aliases of step 2 are removed in the same major version.
