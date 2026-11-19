# Data model proposal for `sigmaepsilon.mesh`

**Status:** draft, under discussion. Nothing in this document has been implemented yet.

**Scope:** how point and cell data, block metadata and computed values are stored in a
`PolyData` tree, how users attach their own (scalar, vector, tensor) data, how packages
such as `sigmaepsilon.solid.fem` extend the model, and how meshes are saved and loaded.

The findings in section 2 were checked by running code against `main` at `ad942db`.

---

## 1. Context and goals

A mesh is a tree of `PolyData` blocks (a `DeepDict`). In the common case, the root holds
the point data and each leaf block holds one block of cells, whose topology refers to
those points. Inner blocks may hold their own point data, so one mesh object can host
several meshes in the same space. Point data and cell data both hold data about their
points and cells. Today all of that data is stored with Awkward.

Goals:

- Store data efficiently.
- Store metadata for each block.
- Keep internal data (e.g. topology) next to data provided by the user.
- Let users attach any data to points or cells without name clashes with internal data.
- Store scalar, vector and tensor data, using the tools in `sigmaepsilon.math`. In the
  extreme case, both the components and the frame differ from cell to cell.
- Allow fully ragged data, so that the whole mesh can be a single cell block.
- Keep the model extensible: `sigmaepsilon.solid.fem` builds on it, and more packages may
  do so in the future.
- Save meshes and load them back, preferably to a single file.

---

## 2. Audit of the current implementation

### 2.1 Internal and user field names can clash

- The only protection is a naming convention, and it isn't consistent:
  - `PointData` uses double underscores (`__x`, `__id`, `__activity`), and its docstring
    warns users about them.
  - `CellData` uses single underscores (`_topology`, `_id`, `_frames`, `_t`).
  - `solid.fem` adds a third convention (`_K_`, `_density_`, `_shapes_`), and some of its
    internal fields have no prefix at all (`loads`, `mass`, `fixity`, `forces`, `reactions`).
- `cd.db["_id"] = zeros` silently overwrites the global cell ids.
- Reading and writing by key don't agree. `AkWrapper.__getitem__` translates public names
  through `_attr_map_`, but `__setitem__` doesn't. After `cd.db["nodes"] = my_array`,
  reading `cd.db["nodes"]` returns the topology, and the user's array can't be reached by
  that key.
- `CellData(fields={"t": ..., "id": ...})` silently remaps the user's `t` and `id` onto
  the internal fields.
- Internal names leak into `fields`, `to_dataframe` and parquet output.

### 2.2 Subclasses can't reliably extend the attribute map

`ABCMeta_AkWrapper` merges `_attr_map_` so that the **base class wins**. In `solid.fem`,
`CellData` declares `"activity": "_activity_"`, but the merged map resolves to the mesh's
`_activity`.

### 2.3 Attribute access falls through to Awkward

- `PolyCell` falls through to `CellData`, which falls through to the Awkward array.
- `pd.temperature = arr` doesn't store data: it quietly becomes a plain Python attribute,
  so it never reaches exports or plots.
- Which attributes exist depends partly on Awkward's own API, and error messages are hard
  to read.

### 2.4 Awkward is mostly used as a dict of regular columns

- Every field the code relies on is a regular array: coordinates, topology, frames, nodal
  distribution factors. Each access goes through `.to_numpy()`.
- Writing into a returned array in place works today because the layout is regular. It
  silently stops working once a column becomes optional or ragged.
- Per-cell frames are always stored expanded to (nE, 3, 3), even when every cell shares
  one frame (2160 bytes for 30 cells with one frame in the test).
- All fields of a block are zipped into one record. Combining blocks with different
  fields produces union and optional types on export (`var * int64`, `?float64`).

### 2.5 Block metadata lives among the child blocks

- `mesh['A']['material'] = 'steel'` works, but it sits in the same dict as the child blocks.
- Because the tree auto-creates missing keys, a typo like `mesh['A']['materal']` silently
  creates an empty child block.
- `config` (plotting settings) is a third place where per-block state is kept.

### 2.6 Ids depend on where a block sits in the tree

Global point and cell ids come from the root's index managers. They are rewritten when a
block joins a tree and deleted when it leaves (`_reindex`, `__after_join_parent__`,
`__after_leave_parent__`). A block's stored data changes whenever it moves, and copying
or merging needs re-indexing.

### 2.7 Nothing round-trips through a file

- `pickle` returns a mesh with `pd=None` and `cd=None`. `DeepDict.__reduce__` passes only
  the dict items, and point and cell data are instance attributes.
- `PolyData.to_parquet` writes two flat files with all blocks concatenated. The tree,
  frames, cell types and metadata are lost, and there is no `PolyData.from_parquet`.
- `to_ak` goes through Python lists: 2.9 s for 216k H8 cells.
- The remote branches `154-save-to-one-file` and `feature/save_to_one_file` (last commit
  2025-07-26) contain no save or load implementation. Their changes are build, plotting
  and type-hint updates.

### 2.8 Bugs found along the way

- `CellData.set_nodal_distribution_factors` always raises `KeyError: '_ndf'`. It looks up
  `_attr_map_` with the internal key, and its fallback indexes with `.id` instead of
  `_id`.
- `AkWrapper.to_dict` and `to_list` call `fields.extend(args)`, which modifies the
  caller's list (and fails for tuples).

---

## 3. Proposal

### 3.1 Separate namespaces instead of naming conventions

Each `PointData` and `CellData` has two kinds of fields:

- **Core fields** belong to a package (`mesh:x`, `mesh:topology`, `mesh:id`,
  `mesh:activity`, `mesh:frames`, `fem:K`, ...). They are declared in the class's
  `Schema` (section 3.5) and reached through generated properties (`pd.x`, `cd.topology`).
  They never appear among user fields.
- **User fields** live in `pd.data` / `cd.data`, a `MutableMapping` with no naming rules.
  `pd["temperature"]` stays as a shortcut for `pd.data["temperature"]`.

This is how PyVista (`point_data`), meshio and xarray (`coords` vs `data_vars`) handle it.
Clashes become impossible and no prefixes are needed. The `__getattr__` fall-through to
Awkward is removed.

Every package gets its own namespace for core keys, so a future package (say
`solid.dynamics`) can add fields without ever clashing with `fem` or with users. In a
saved file these are qualified keys or HDF5 groups.

### 3.2 Fields

A field is a typed object, not a bare column:

```
Field
├─ values     ndarray | JaggedArray   leading "item" axes + trailing component axes
├─ kind       what the values are, and which class they are returned as
├─ frame      FrameSpec (only for kinds that transform with a frame)
└─ location   point | cell | cell-node | quadrature-point
```

**Kinds** (replacing the earlier idea of a numeric `rank`, which suggests that a vector is
a "1d tensor"):

| kind | Transforms with a frame | Returned as |
|---|---|---|
| `"array"` | no (ids, stiffness matrices, flags) | `ndarray` / `JaggedArray` |
| `"scalar"` | no (frame-invariant) | `ndarray` |
| `"vector"` | yes | `Vector` |
| `"tensor2"`, `"tensor2x3"`, `"tensor4"`, ... | yes | `Tensor2`, `Tensor2x3`, `Tensor4`, ... |
| `"frame"` | – | `ReferenceFrame` (e.g. per-cell frames) |

- Kinds live in a registry. A package can add one with its own transformation rule, e.g.
  `fem` registering an element stiffness matrix that transforms with a block-diagonal
  direction-cosine matrix.
- How many trailing component axes a field has is derived from the class. Users never
  pass a number.
- `sigmaepsilon.math` has no `Scalar` class, so scalars are plain arrays.

**Layout rule for values:** ragged dimensions only appear among the leading item axes; the
trailing component axes are always regular. For example, per-cell-node stresses on mixed
cells have type `n * var * 3 * 3`.

**Uniform fields:** a field may hold a single value that stands in for every item (a
thickness for a whole part, one frame for all cells). It is expanded only when needed.

### 3.3 Frames

`sigmaepsilon.math` already handles per-item frames: a `ReferenceFrame` built from axes of
shape (n, dim, dim) works with `Tensor2(..., bulk=True)`, and `show()` into a single frame
works too. A frame is fully described by its class, its `axes` (always relative to the
ambient frame, with no parent chain) and its `name`.

**Frame registry.** Each mesh holds a registry of live frame objects, each under an id.
Stacked per-item axes are still a single frame object.

**FrameSpec.** Each field stores a small, serializable description of its frame:

```
FrameSpec
├─ source   "registry" | "field" | "host"
├─ ref      registry frame id | qualified field key (e.g. "mesh:frames") | None
└─ mapping  "shared" | "per_item" | "indexed"
```

- `"host"` means the frame of the hosting point data. Fields in the host frame move with
  the mesh frame and their components don't change.
- `"field"` points at another field, e.g. stresses given in each element's local frame
  (`mesh:frames`) without copying the frames.
- `"indexed"` stores k distinct frames plus an integer index per item, e.g. 10⁵ beams with
  3 orientations.

**Getting the frame back.** `cd.data.get("stress").frame` or `cd.data.frame("stress")`
returns the identical object (`is F`) within a session. After saving and loading you get
the same class, axes and name, and sharing is preserved: one object per registry entry,
so fields that shared a frame share one again. Identity with variables outside the mesh
can't survive a reload.

**Fields register with their frames.** In `math`, setting a frame's `axes` in place (which
`orient` does) re-expresses every tensor registered to that frame, so the physical
quantity stays the same. A field stored as bare components plus a frame reference would
break that: rotating the frame afterwards would silently change the physics. So `Field`
implements the same small interface tensors do (`show(dcm=...)` and an `array` setter) and
registers itself through `_register_tensorial_`. The store holds the field strongly, so
the frame's weak reference stays valid.

**Change of basis vs rigid motion.** These are two different operations:

| Operation | Example | Components | Physical quantity |
|---|---|---|---|
| Change of basis | `frame.orient(...)` | re-expressed | unchanged (`math` semantics) |
| Rigid motion | `mesh.rotate(...)`, `mesh.move(...)` | carried along | moves with the body |

Recommendation: fields follow `math` for a change of basis. That keeps a field stored in a
mesh behaving exactly like a `Tensor` stored anywhere else in the ecosystem.

Today `PolyData.rotate` turns the points and then calls `cd._rotate_` on the cell frames,
which is a rigid motion. Once fields are registered with frames, rotating the cell frames
in place would re-express the stresses so they stay physically fixed while the body turns.
So rigid motion needs one addition to `math`: rotating a frame while **carrying** its
registered tensors, e.g. `frame.orient(..., carry=True)` or a `with frame.carrying():`
context. `mesh.rotate` then:

- carries every field that has its own frame (registry frames, per-cell frames),
- transforms the components of vector and tensor fields in the host frame by the same
  rotation,
- leaves scalars and plain arrays alone.

Not every vector should move with the body: a gravity load given in the global frame
shouldn't turn when the part is rotated. That needs a per-field flag,
`motion="follow" | "fixed"` (open decision D1).

### 3.4 Access API

`cd.data` is a full `MutableMapping`: `[]`, `in`, `del`, `keys()`, `items()`. On top of
that, `get` and `set` take options for edge cases:

```python
s = cd.data["stress"]            # math object for tensorial kinds, arrays otherwise
s.show(mesh.frame)               # into the global frame

cd.data.get("stress", default=None, *,
            as_="auto" | "array" | "tensor",
            frame=target,        # return components transformed into a target frame
            resolve=True,        # fall back to uniform / cascaded values
            copy=False)

cd.data.set("stress", value, *,
            kind=None, frame=None, location=None,
            uniform=False, overwrite=True)

cd.data.frame("stress")          # the field's frame object
```

- **Decided:** `cd.data["stress"]` returns a math object by default. Raw components come
  from `cd.data.get("stress", as_="array")`.
- `get`'s second positional argument stays `default`, so it behaves like `dict.get`.
- `set` infers kind and frame from a `Vector`/`Tensor` value. The keyword arguments are
  only needed for plain arrays.
- The field owns its math object, whose array is a view onto the stored buffer. So
  `cd.data["s"] is cd.data["s"]`, the object stays registered with its frame, and nothing
  is rebuilt on each read.
- Ragged tensor fields don't fit `math` yet, because bulk tensors need a regular `ndarray`
  (open decision D3).

### 3.5 A nested `Schema` per data class

```python
class CellData:
    class Schema(BaseSchema):
        namespace = "mesh"
        topology = FieldSpec(kind="array", location="cell", dtype="int64", ragged_ok=True)
        frames   = FieldSpec(kind="frame", location="cell", uniform_ok=True)

class FemCellData(CellData):
    class Schema(CellData.Schema):
        namespace = "fem"
        K       = FieldSpec(kind="array", location="cell", derived=True, persist=True,
                            depends_on=["mesh:topology", "point:mesh:x", "fem:C"])
        density = FieldSpec(kind="scalar", location="cell", uniform_ok=True)
```

- Ordinary Python inheritance merges the specs, and the subclass wins. This replaces
  `_attr_map_` and the metaclass, and fixes the merge order of section 2.2.
- `__init_subclass__` checks that a nested `Schema` subclasses the parent's `Schema`, so
  inherited fields can't be dropped by accident. It also generates the outer class's
  properties from the specs.
- Each field remembers the namespace of the class that declared it (`mesh:topology` and
  `fem:K` side by side).
- `FieldSpec` as a pydantic model gives JSON serialization for free; pydantic is already
  in the stack through `sigmaepsilon.math`.
- A saved file stores the schema of every namespace together with its `schema_version`
  (section 3.12).
- The user namespace gets specs too, inferred at `set` time (kind, location, FrameSpec).
  It is a dynamic schema, saved the same way.

### 3.6 Block metadata and cascading

Each block gets `block.meta`, a JSON-serializable dict (like xarray's `.attrs`). Only
`PolyData` instances are allowed as dict items, so typos raise an error instead of
creating blocks.

There are three kinds of data, and they cascade differently:

| Kind | Cascades? | Why |
|---|---|---|
| Point/cell field arrays | never | The parent's lengths have nothing to do with the child's. |
| Block attributes (`color`, `material`, `label`) | decided per key | `color` should cascade, `name` shouldn't. |
| Uniform fields (`t`, `frames`, `density`) | yes, as a fallback | A single value stands in for a whole array. |

The mechanism is a declared spec per attribute, similar to how CSS marks some properties
as inherited:

```python
AttrSpec("color", inherit=True, default=None)
AttrSpec("name", inherit=False)
```

- `block.meta` holds only the values set on that block.
- `block.attrs["color"]` returns the resolved value: the block itself, then its parent,
  and so on up to the root, then the spec default.
- A sentinel distinguishes "explicitly set to None" from "not set".
- Only local values are saved, never resolved ones.
- Reading a cell field like `t` follows the same chain: a per-cell array, then a uniform
  value on the block, then an inherited uniform value, then the default.
- Plotting `config` becomes a set of inheritable attributes.

### 3.7 Ids

Global ids are computed from per-block offsets when needed and cached when the mesh is
locked (`lock()`). A block's data no longer changes because it moved in the tree.

### 3.8 Ragged data and mixed cell types

Awkward stays as the engine for ragged data, but used differently:

- Each field is its own column, not part of one `ak.zip` record. Fields in the same block
  can have different ragged structures (per node vs per quadrature point), and the
  union/optional types on export disappear.
- A regular field is a NumPy array (zero-copy). A ragged field is offsets over regular
  trailing axes, with no optional or union types. Numba kernels receive
  `(offsets, flat_values)`.
- The ragged field type is `sigmaepsilon.math.linalg.sparse.JaggedArray` (already
  Awkward-backed), so the ecosystem has one ragged type.
- Ragged fields are saved as offsets + values datasets.

**Decided:** storage allows mixed cell types in one `CellData` through a `mesh:celltype`
column and ragged `mesh:topology`, as in VTK unstructured grids. Computation runs on
grouped same-type views (`cd.groups()`), which is what kernels and FEM elements work on.
Users rarely build mixed meshes directly; the main use is turning a mesh into a single
cell block for systematic downstream consumption.

**Lossless single-block round trip** (`to_single_block` / `from_single_block`;
`to_standard_form` becomes one step in this pair):

- Columns added by the export: `mesh:celltype`, ragged `mesh:topology`, and `mesh:block`
  (an index into a **block table** of block addresses and metadata).
- Fields missing from some blocks are filled with the spec default, plus a validity mask
  column instead of Awkward option types.
- Cascading attributes stay in the block table. `resolve_attrs=["color", ...]` can
  additionally turn chosen ones into per-cell columns.
- `from_single_block` rebuilds the tree from the block table.

### 3.9 Extensibility

- **Schemas and namespaces** (sections 3.1 and 3.5) are the main extension mechanism.
- **Separate storage from behaviour.** Today a `PolyCell` is the data, the geometry and
  the behaviour in one class, and `fem` stacks mixins on top of it
  (`ElementStiffnessMixin`, `ElementLoadsMixin`, ...). Instead, the cell type becomes a
  stateless behaviour class that works on a store.
- **Cell type registry.** Cell types are registered by label and version (`"H8@1"`), which
  loading needs anyway. A Python entry point (`sigmaepsilon.mesh.celltypes`) lets a saved
  file open FEM elements without importing `fem` by hand. Each cell type records a
  `base_label` (e.g. `fem:H8_Solid` → `mesh:H8`), so a file still loads as a plain mesh
  when the extending package isn't installed.
- **Kind registry** (section 3.2) for quantities with their own transformation rules.
- The tree events `DeepDict` already provides (`__after_join_parent__` etc.) remain the
  extension hooks.

### 3.10 Computed fields

**Decided:** computed fields are saved by default. Recomputing them later (e.g. element
matrices in `fem`) can be slow, and storage is cheaper than time. The risk this creates is
**staleness**: a saved `K` that no longer matches the geometry or material. So computed
fields record where they came from:

```python
K = FieldSpec(kind="array", location="cell", derived=True, persist=True,
              depends_on=["mesh:topology", "point:mesh:x", "fem:C", "meta:quadrature"])
```

- **In a session:** every field has a write counter. A computed field records the counters
  of its inputs when it is computed; comparing them is cheap.
- **When saving:** a fast hash of each input buffer (blake2/xxhash) is stored next to the
  computed field.
- **When loading:** if the hashes and the schema version match, the field is valid.
  Otherwise it is marked stale.
- **Reading a stale field:** see open decision D5.
- A migration may simply mark computed fields stale; that is often the easiest correct
  migration.
- Opt-out: `save(path, derived="keep" | "drop")`.

Size is still worth knowing. An H27 element stiffness matrix is 81×81 doubles, about
52 kB per cell, so 10⁵ cells is about 5 GB. HDF5 chunking and compression help, and
partial reads mean loading the mesh doesn't require loading `K`.

### 3.11 Persistence

A format-neutral `to_dict()` / `from_dict()` produces a description of the mesh plus its
arrays:

- the tree of block paths, with each block's local `meta`,
- per block: point and cell fields with their specs (including FrameSpec), and the cell
  type label and version,
- the frame registry: class name, frame name, axes, optional index,
- the schemas and versions of every namespace (section 3.12),
- the format version and package versions.

Backends are built on top of it:

| Backend | Single file | Dependencies | Notes |
|---|---|---|---|
| `.npz` + JSON description | yes | none | Good default, but loads everything at once. |
| HDF5 (`h5py`) | yes | optional | **Recommended.** Groups map onto the tree, metadata goes into attributes; compression and partial reads. |
| Zarr (zip store) | yes | optional | The same layout, cloud-friendly. |
| Parquet | no | already used | Tabular, so the tree only fits as one file per block. Keep it as a data export. |
| VTK `.vtm` / XDMF | multi-file | optional | Interop export only: frames and metadata don't survive. |

Sketch of the HDF5 layout:

```
/                                attrs: format_version, packages, schemas (JSON)
/frames/<frame id>/axes          attrs: class, name
/frames/<frame id>/index         only for "indexed" frames
/blocks/<block path>/            attrs: meta (JSON, local values only)
    point/mesh:x                 attrs: spec (JSON)
    point/data/<user field>      attrs: spec (JSON)
    cell/                        attrs: celltype ("H8@1"), base_label
    cell/mesh:topology           regular: one dataset
    cell/mesh:topology/offsets   ragged: offsets + values
    cell/mesh:topology/values
    cell/fem:K                   attrs: spec, input hashes
/opaque/...                      fields kept from unknown namespaces or newer versions
```

`pickle` and `deepcopy` are fixed separately with `__getstate__` / `__setstate__`.

### 3.12 Versioning and migrations

**Four things are versioned separately**, because they change for different reasons and
different packages own them:

| What | Owner | Changes when |
|---|---|---|
| `format_version` | `mesh` io | The file layout changes (tree, frame registry, ragged arrays). |
| Schema version per namespace (`mesh`, `fem`, ...) | the package declaring that `Schema` | A stored field is renamed, changes type, or changes meaning. |
| Cell-type version (`"H8@1"`) | whoever registers the cell type | Node ordering or local-coordinate conventions change. |
| Package versions | – | Informational only, recorded for diagnostics. |

The schema version is independent of the package version. Adding an optional field needs
no migration. A rename, retype or change of meaning bumps the version and requires one.

**Migrations:**

```python
@migration(namespace="fem", from_version=2, to_version=3)
def rename_density(block: StoredBlock) -> StoredBlock:
    block.fields.rename("fem:rho", "fem:density")
    return block
```

- Migrations work on the **stored representation**: a lazy, dict-like view of the arrays,
  specs, metadata and frame registry. They never touch live classes, because classes keep
  changing and a migration that imports today's `FemCellData` breaks later (the rule used
  by Alembic and Django migrations).
- They run as a chain (v1→v2→v3), in memory on load. An explicit
  `sigmaepsilon.mesh.io.upgrade(path)` rewrites the file in place.
- Migrations only go forward. Writing an older format isn't supported.
- Each namespace's migrations are tested against sample files: one small file per past
  schema version, kept in `tests/data/`.
- The user namespace never gets migrations. It is stored as-is with its inferred specs.

**Load policy for each namespace:**

| Situation | Default | Options |
|---|---|---|
| Versions match | load | – |
| File is older and a migration path exists | migrate in memory | `migrate=False` → error |
| File is older and no path exists | error, naming the missing step | – |
| File is newer than the code | keep those fields **opaque** and warn | `on_newer="error"` |
| Namespace's package not installed | keep opaque and warn | `on_missing="error"` |

Opaque fields are preserved unchanged and written back on save, the way protobuf keeps
unknown fields. Opening and saving a file with an older or smaller install never loses
data. Cell types from a missing package fall back to their `base_label`.

---

## 4. Decisions taken

| Topic | Decision |
|---|---|
| Field type | `kind` naming the class (scalar, vector, tensor2, ...), not a numeric rank. |
| Access | `cd.data["x"]` returns a math object for tensorial kinds; `get`/`set` with options next to `[]`. |
| Schema | Nested `Schema` class per data class, serializable, merged by inheritance. |
| Mixed cell types | Allowed in storage; computation on grouped same-type views; lossless single-block round trip. |
| Computed fields | Saved by default, with provenance and staleness checks. |
| Frames on change of basis | Follow `math` (recommended; confirmation pending together with D1/D2). |

---

## 5. Open decisions

- **D1. Which vectors move with the body.** Should a vector or tensor field move with the
  body by default under rigid motion (`motion="follow"`), with `"fixed"` as an opt-in for
  things like global loads? Recommendation: `follow` for fields with their own frame, and
  an explicit choice required for fields in the host frame.
- **D2. Carry operation in `math`.** Is it acceptable to add a carry operation to frames
  (`orient(..., carry=True)` or a context manager)? Recommendation: change `math` rather
  than work around it in `mesh`.
- **D3. Ragged tensor fields.** Return a `math` tensor over the flat values plus `offsets`,
  or give `math` a ragged bulk mode? The first is quick; the second is cleaner long-term.
- **D4. Migration ownership.** Each package ships its own migrations and sample files
  (recommended), or a central registry in `mesh`?
- **D5. Stale computed fields.** Recompute automatically on read when a compute function
  is registered (recommended), or always raise so recomputation is explicit?
- **D6. Symmetric tensors.** Should they have an optional compact (Voigt) storage form?
  Not discussed yet.
- **D7. Default backend.** HDF5 as the recommended single-file format with `.npz` as the
  dependency-free fallback, or `.npz` first? Not discussed yet.

---

## 6. Suggested order of work

1. Fix the bugs in section 2.8 and pickling; add round-trip tests.
2. Add the `.data` and `.meta` namespaces and the nested `Schema`, with a deprecation shim
   for the old string-keyed access and `_attr_map_`.
3. Add fields with kinds and FrameSpec, the frame registry, and the carry operation in
   `math`.
4. Add `to_dict` / `from_dict` with the npz and HDF5 backends, versioning and migrations.
5. Add computed-field provenance; port `solid.fem` to schemas.
6. Replace the zipped Awkward record with per-field columns, hidden behind steps 2–4.
