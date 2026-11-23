# Implementation plan for the data model proposal

**Implements:** [data-model-proposal.md](data-model-proposal.md). Section numbers (§) refer to the proposal.

**Status:** in progress (current phase: P0). Progress is tracked per work package in section 2.

**Baseline:** `main` at `ad942db`, version 3.4.3, Python ≥ 3.10.

---

## 1. Strategy

### 1.1 Two release tracks

The proposal has non-breaking changes, which can ship with deprecation warnings, and breaking ones, which it puts in the next major version. The plan splits the work along that line:

| Track | Releases | Content | Compatibility |
|---|---|---|---|
| **3.x** | 3.5 – 3.8 | bug fixes, class layout, fields and schemas, frames, ids, methods on a subtree | old names, keywords and string access keep working and warn |
| **4.0** | 4.0 | block configuration, plotting backends, persistence and versioning, removal of k3d, Plotly, `rotate` and every alias | breaking, no shims |

Persistence goes into 4.0 rather than 3.x. The saved format holds the block configuration (§3.12), and that configuration only exists from 4.0 on (§3.6 replaces today's `config` without a shim). Saving files in 3.x would mean either a format without configuration or a format version bump in the next release, and both leave files behind that need a migration from day one.

### 1.2 One change to the order of §6

§6 adds the access API on top of the Awkward record (step 3) and replaces the record with per-field columns later (step 7). This plan does both in phase P3: `core` and `data` are built over the new storage straight away. Building them over the zipped record first would mean writing the uniform values, the spec validation and the read-only views twice, and the record is what causes the problems of §2.4 in the first place. Awkward stays a dependency, for `JaggedArray` and the `to_ak` export.

### 1.3 Working rules

- One PR per work package (the `Pn.m` items below), against `dev`. Each PR keeps the full test suite green, with and without `NUMBA_DISABLE_JIT`.
- Behaviour the proposal changes is first pinned by a test of today's behaviour, so the PR that changes it shows up as a deliberate edit of that test.
- Every public symbol goes through `__all__` and gets a NumPy-style docstring with a runnable example (`--doctest-modules` is on).
- Docs notebooks compute the numbers they show and declare every variable they use; `docs/source/notebooks/data_model.ipynb` is updated in every phase that changes what it describes.
- `solid.fem` is ported in step with each phase (section 4), against an editable install of the `mesh` branch, so a broken extension point shows up before a release.
- The example of §3.16 becomes an end-to-end test, `tests/design/test_full_picture.py`. Every line starts as `xfail`, and each phase turns on the lines it implements, so the test shows how far the implementation has got.

### 1.4 Releasing without breaking dependants

The promise to users is Semantic Versioning: code that runs on 3.4 without warnings keeps running on every 3.x release, and 4.0 removes only what 3.x warned about. The rules below keep that promise for the packages of the namespace and for code outside it.

#### Deprecations

- Every deprecation goes through one helper in `sigmaepsilon/mesh/_deprecation.py`, which warns with `stacklevel` pointing at the caller's line. The message names the replacement, the version that removes the old form (always 4.0), and the page of the migration guide (P8.2), e.g. ``"`PolyCell` is deprecated, use `CellData`; it is removed in 4.0. See <url>."``
- Renamed names and keywords warn with `DeprecationWarning`. Python shows it for calls made from `__main__` (scripts, notebooks) and in test runs, so it reaches users and the developers of dependent packages without flooding the users of those packages. Changes of behaviour (a fall-through that will raise, a default that will go away, block creation on read) warn with `FutureWarning`, which is always shown, because they change results rather than names.
- Nothing is removed in 4.0 that hasn't warned in at least one 3.x release. 3.8 is the transition release: everything P8.1 removes warns in 3.8, and every replacement exists in 3.8. Code that runs on 3.8 without warnings runs on 4.0.
- `mesh`'s own tests turn its deprecation warnings into errors (P2), so no internal code uses an old name.

#### Dependent packages

- The packages of the namespace cap their dependency on `mesh` at the next major version (`sigmaepsilon-mesh>=3.x,<4.0.0` in `solid.fem`, `solid.fourier` and `solid.material`), so publishing 4.0 never breaks an installed release of them. Third-party dependencies (NumPy, SciPy, ...) stay without upper caps, so the packages still install next to preinstalled ones (e.g. on Databricks); the cap only applies within the namespace, whose major releases are planned together. Each dependant lifts its cap to `<5.0.0` in a release made after it passes against the 4.0 candidate (P8.5).
- Each dependant gets a CI job that installs `mesh` from `dev` and runs its test suite with `mesh`'s `DeprecationWarning` and `FutureWarning` turned into errors (`-W error::DeprecationWarning:sigmaepsilon.mesh`, likewise for `FutureWarning`), so a use of an old name fails the job while 3.x still accepts it. The job is allowed to fail until the dependant is ported, and required afterwards.
- Packages outside the namespace are reached through the warnings, the changelog and the announcement below.

#### Release candidates

- 4.0 is published as `4.0.0rc1` (and further candidates as needed) to PyPI, not only to TestPyPI, because dependencies only resolve there. pip and uv skip pre-releases unless asked (`--pre`, `--prerelease=allow`), so nobody gets a candidate by accident.
- The candidate is final when the test suites of `solid.fem`, `solid.fourier` and `solid.material` pass against it, and their docs notebooks run.

#### After 4.0

- A `maint/3.x` branch, cut from the 3.8 tag, takes bug fixes for six months after 4.0 and publishes 3.8.x releases. The CircleCI publish job gets a branch filter for it.
- Read the Docs keeps a `3.x` version next to `stable` and `latest`, so 3.x users find their documentation.
- Pickles are not compatible across major versions: a pickle written by 3.x that refers to a removed class or alias doesn't load in 4.0. The changelog and the migration guide say so, and point to the file format of P7 as the way to keep meshes across versions.

#### Communication

- The changelog lists every deprecation under `Deprecated` in the release that adds it, and every removal under `Removed` in 4.0, each with its replacement.
- A pinned GitHub discussion announces the plan when 3.6 (the first release with deprecations) is published, links the migration guide as it grows, and announces the candidate and the date of 4.0.

---

## 2. Phases

Each phase lists its work packages, the files touched, the tests that close it, and its exit criteria.

**Tracking progress.** The `Status` column of each work package is `–` (not started), `in progress (#PR)` or `done (#PR)`, with a link to the pull request. The heading of a phase says `not started`, `in progress` or `released in x.y`, and the status line at the top of the file names the current phase. Update the row in the PR that does the work, so the plan and the code never disagree.

### P0. Preparation (no release) — in progress

| # | Work | Files | Status |
|---|---|---|---|
| P0.1 | Add `pydantic>=2` as a direct dependency (today it only comes through `sigmaepsilon.math`). Add `typing_extensions` for `Self` (Python 3.10). The `hdf5` extra with `h5py` is added in P7.5, where it is first used. | `pyproject.toml`, `uv.lock` | in progress |
| P0.2 | Characterization tests for every finding of §2 that the proposal changes, each marked with the section it comes from. They assert today's behaviour where it is going to change, and are `xfail(strict=True)` where they assert the target behaviour of a bug. | `tests/design/test_audit_*.py` (new) | – |
| P0.3 | Benchmarks for the numbers quoted in §2: `to_ak` on 216k H8 cells (2.9 s), memory of shared frames (2160 bytes for 30 cells), and `topology()` / `volumes()` on a large mixed subtree. Plain scripts, not part of CI. | `benchmarks/` (new) | – |
| P0.4 | Inventory of what `solid.fem` and `solid.material` use from `mesh`: `_attr_map_`, `_wrapped`, `PolyCell`, `data_class`, `.db`, `_init_config_`, `pd`/`cd`, `coords=`/`topo=`. Kept as a checklist in the fem port PRs. | – | – |

**Exit:** characterization tests merged; the inventory lists every use site.

### P1. Bugs and pickling — release 3.5 (§2.7, §2.8, §6 step 1) — not started

| # | Work | Files | Status |
|---|---|---|---|
| P1.1 | `CellData.set_nodal_distribution_factors`: look up the key correctly and index with `_id`. | `data/celldata.py` | – |
| P1.2 | `AkWrapper.to_dict` / `to_list`: copy `fields` instead of extending the caller's list. | `data/akwrapper.py` | – |
| P1.3 | `IndexManager.generate`: advance the counter only by the ids generated beyond the queue. `recycle`: raise `ValueError` for ids not in use, negative ids and duplicates. | `indexmanager.py` | – |
| P1.4 | `PointCloud.closest` / `furthest`: return `self.inds[id]`. `index_of_furthest`: infer the frame from the shape of the input like `index_of_closest`; fix its docstring. | `space/pointcloud.py` | – |
| P1.5 | `pickle` and `deepcopy` of `PolyData`, `PointData` and cells through `__getstate__` / `__setstate__` (and `__reduce__` on `PolyData`, since `DeepDict.__reduce__` passes only the items). | `data/polydata.py`, `data/pointdata.py`, `data/celldata.py`, `data/polycell.py` | – |
| P1.6 | `pvplot` works on a copy of the resolved config instead of popping `color` from the stored one (§2.5, third bullet). The rest of §2.5 is fixed by P6. | `plotting/pvplot.py` | – |

**Tests:** the `xfail` tests of P0.2 for §2.7 and §2.8 pass and lose their marker; round-trip tests for `pickle` and `deepcopy` of a nested mesh with point data, cell data, frames and user fields.

**Exit:** release 3.5.

### P2. Class layout — release 3.6 (§3.15, §6 step 2) — not started

| # | Work | Files | Status |
|---|---|---|---|
| P2.1 | One set of type variables, `PointDataT`, `CellDataT`, `PolyDataT`, invariant, defined once in `typing/`. Old names (`PointDataLike`, `PolyCellLike`, `PolyDataLike`, `MeshDataLike`, `CellDataLike`) re-exported as aliases. | `typing/data.py`, `typing/__init__.py`, `data/celldata.py`, `data/polycell.py` | – |
| P2.2 | Merge `PolyCell` into `CellData`: every method of `PolyCell` moves to `CellData`, `CellData` becomes the class cell types subclass. Add `CellData1d`, `CellData2d`, `CellData3d` and make every cell type subclass the one of its dimension. `PolyCell` stays as a deprecated alias of `CellData`; `.db` returns the instance itself with a warning, so `cd.db.db` still reaches the cell data, and Awkward attributes still resolve through the `__getattr__` fall-through until P3.4 deprecates it; `data_class` is removed (the only user, fem, is ported in P2.6). | `data/celldata.py`, `data/polycell.py` (alias module), `cells/*.py` | – |
| P2.3 | One metaclass `DataMeta` (deriving from `ABCMeta_Weak`) in place of `ABCMeta_AkWrapper` and `ABCMeta_PolyCell`. In this phase it still merges `_attr_map_` (with the subclass winning, fixing §2.2) and registers cell types by `label`, recording `base_label` from the inheritance. Registering a label twice raises. | `data/meta.py` (new), `typing/abcakwrapper.py`, `typing/abcpolycell.py` (aliases) | – |
| P2.4 | Renames with deprecated aliases: `PolyData.pd`/`cd` → `pointdata`/`celldata`; constructor keywords `pd=`/`cd=` → `pointdata=`/`celldata=`, `coords=` → `x=`, `topo=` → `topology=`; `_point_class_` → `pointdata_class`; `_point_array_class_` → `PointData.pointcloud_class` (with `pointdata.pointcloud()`); `PolyCellGeometry1d/2d/3d` → `CellGeometry1d/2d/3d`; `PolyCellProtocol`, `GeometryProtocol` → `CellDataProtocol`, `CellGeometryProtocol`. `CellDataProtocol` drops `pointdata` and `container` stays the only back-reference. | `data/*.py`, `geometry/geometry.py`, `typing/*.py`, every internal call site | – |
| P2.5 | Container registry for `PolyData` subclasses (`"PolyData"`, `"LineData"`, `"TriMesh"`), with `label` and `base_label`. Done with `__init_subclass__`, not a metaclass, so subclasses in other packages (fem's `FemMesh` uses its own `ABCMeta_FemMesh`) don't hit a metaclass conflict. | `data/registry.py` (new), `data/polydata.py`, `data/linedata.py`, `data/trimesh.py` | – |
| P2.6 | Port fem's cell classes to subclass `mesh`'s cell types directly, with fem's data class as a common base (`class H8Solid(FemCellData, H8)`). | `solid.fem` (section 4) | – |

**Tests:** every cell type is an instance of `CellData` and of the right `CellDataNd`; the registries hold every shipped label with the right `base_label`; each alias warns once with `DeprecationWarning` and returns the same object as the new name; the mesh test suite and fem's test suite pass, with the aliases filtered as errors in `mesh`'s own tests (`filterwarnings = error::DeprecationWarning` for `sigmaepsilon.mesh`), so no internal code still uses an old name.

**Exit:** release 3.6; fem released against it.

### P3. Fields, schemas and storage — release 3.7 (§3.1, §3.2, §3.4, §3.5, §3.8 storage, §6 steps 3, 4 and 7) — not started

| # | Work | Files | Status |
|---|---|---|---|
| P3.1 | `FieldSpec` (pydantic, JSON round trip) with `kind`, `location`, `at`, `dtype`, `shape`, `default`, `persist`. Spec inference from a first value (§3.2, "An inferred spec ..."). Validation of a value against a spec: item axes per location (points; cells; cells × nodes of the cell type; cells × rows of `at`), safe casts, component shape, uniform detection by shape. `sigmaepsilon.math` tensor objects raise `TypeError`. | `data/fields.py` (new) | – |
| P3.2 | Private storage of a data holder: a dict of NumPy arrays or single uniform values, plus user-field specs. `Fields`, `CoreFields`, `UserFields` as views over it (§3.4). `get(resolve=True)` expands uniform values on read; reads return arrays that share memory with storage, `copy=True` returns a copy. | `data/fields.py` | – |
| P3.3 | Nested `Schema` merged by `DataMeta`: reused names, changed specs of inherited fields and clashes with class attributes raise `TypeError` naming the first declaring class; diamonds allowed. Generated properties for core fields. `Schema.fields()`, `owner()`, `to_json()`. Owner package found by walking `__module__` up to the first module that defines `__schema_version__`; a data class with new fields and no such module raises. A subclass that still declares `_attr_map_` gets a `DeprecationWarning`, and its entries are honoured by the shim of P3.5 until 4.0. | `data/meta.py`, `data/schema.py` (new), `sigmaepsilon/mesh/__init__.py` (`__schema_version__ = 1`) | – |
| P3.4 | Rebuild `PointData` (`x`, `activity`) and `CellData` (`topology`, `activity`), `CellData1d` (`A`), `CellData2d` (`t`) on the new storage. Constructor keywords are the schema names. The `__getattr__` fall-through to Awkward and setting an undeclared attribute (`pd.temperature = arr`) keep working in 3.x but warn with `FutureWarning`, naming `pd.data`; from 4.0 on both raise `AttributeError` (§3.1, P8.1). `pd["name"]` stays a shortcut for `pd.data["name"]`. `AkWrapper` stays as a deprecated base class with no storage of its own; its export methods move to a mixin over the new storage. | `data/pointdata.py`, `data/celldata.py`, `data/akwrapper.py` | – |
| P3.5 | Deprecation shim for string access by internal name (`cd.db["_topology"]`, `pd["__x"]`, `_attr_map_` lookups from subclasses): mapped to the core field with a warning. Writes of internal names into `data` are rejected rather than mapped, since that is the clash of §2.1. | `data/compat.py` (new) | – |
| P3.6 | Exports over per-field columns: `to_dataframe`, `to_parquet`, `to_ak` list user fields and declared core fields under their plain names, with no internal names (§2.1, last bullet), and without going through Python lists (§2.7, `to_ak`). | `data/export.py` (new), `data/polydata.py` | – |
| P3.7 | `to_master(x, frame)` on cell data: containing cells and master coordinates, built on `locate` and `glob_to_loc` (§3.2, "Utilities"). Document the master-coordinate convention of every cell type in its class docstring. | `data/celldata.py`, `cells/*.py` | – |
| P3.8 | `pull` on point and cell data over the new storage, for `"cell"` and `"node"` fields. | `data/pointdata.py`, `data/celldata.py` | – |

**Tests:** one test module per rule of §3.2 (the code blocks of §3.1, §3.2 and §3.5 become tests nearly line by line); schema merging with diamonds, reuse, changed specs and attribute clashes; specs JSON round trip; uniform fields and the ambiguous first write; the P0.3 benchmark for `to_ak` shows the speed-up. fem's `_attr_map_` and `_wrapped` uses are gone (P0.4 checklist).

**Exit:** release 3.7; fem declares its fields in nested `Schema`s.

### P4. Frames — release 3.7 (§3.3, §6 step 4) — not started

Can be developed in parallel with P3.

| # | Work | Files | Status |
|---|---|---|---|
| P4.1 | Frame storage in three forms, `shared`, `per_cell`, `indexed` (with `frame_index`), never expanded to `(nE, 3, 3)`. A block created without frames gets the frame of its point cloud. | `data/frames.py` (new), `data/celldata.py` | – |
| P4.2 | `cd.frames` and `pd.frame` return a new frame object; assignment raises `AttributeError`. Frames of `sigmaepsilon.math` copy their axes on construction, so the returned frame is independent of the block (section 5, point 5). | `data/frames.py`, `data/pointdata.py` | – |
| P4.3 | Deprecate `PolyData.rotate`, `spin` and `CellData._rotate_` with a warning that names §3.14; removed in 4.0. | `data/polydata.py`, `data/celldata.py` | – |
| P4.4 | Kernels that need per-cell frames (Jacobians, local coordinates of 1D cells) take the frame storage and expand on the fly or per chunk. | `data/celldata.py`, `utils/` | – |

**Tests:** the examples of §3.3; memory of a shared frame is one set of axes (P0.3); indexed frames with 10⁵ beams and 3 orientations give the same Jacobians as expanded frames.

### P5. Ids and methods on a subtree — release 3.8 (§3.7, §3.9, §6 step 7, second half) — not started

| # | Work | Files | Status |
|---|---|---|---|
| P5.1 | Global ids from per-block offsets in tree order (depth first, insertion order), computed on demand and cached while the mesh is locked. Remove `_reindex`, the id parts of `__after_join_parent__` / `__after_leave_parent__`, and the use of the root's `pim` / `cim` for ids. The `pim` and `cim` properties are public, so they stay in 3.8 with a `DeprecationWarning` and go in 4.0 (`IndexManager` stays as a public utility). | `data/polydata.py`, `data/pointdata.py`, `data/celldata.py` | – |
| P5.2 | `@cascade(reduce=...)` decorator and one shared walk over `cellblocks(inclusive=True)`, with the filter arguments of `cellblocks()` and caching while locked. | `data/cascade.py` (new) | – |
| P5.3 | Cascading methods: `id`, `topology()` (always a `JaggedArray` of global point ids, adding each source's point offset), `celltypes()`, `lengths()`, `areas()` (`NaN` where not defined), `volumes()`, `centers()`, `cells_coords()`, `cell_field(name, missing=, fill_value=)`, `number_of_cells()`, `length()`, `area()`, `volume()`. `cell_ids()` becomes a warning alias of `id`. | `data/polydata.py` | – |
| P5.4 | Volume inputs: 2D cells need `t`, 1D cells need `A`, read through the field chain (per-cell array, uniform value). In 3.8 a missing input still counts as 1.0 but warns with `FutureWarning`; in 4.0 it raises `ValueError` naming the block and the field, and the chain gains the fallback to `config.attrs` (P6.3). | `data/celldata.py` | – |
| P5.5 | Check that `JaggedArray` of `sigmaepsilon.math` holds offsets over a regular array with regular trailing axes (`n * var * 4`) and hands `(offsets, flat_values)` to kernels. If it doesn't, extend it in `sigmaepsilon.math` first (section 4). | `sigmaepsilon.math/linalg/sparse/jaggedarray.py` | – |
| P5.6 | A read of a missing key that creates a block warns with `FutureWarning`, naming the tuple-key form `mesh["A", "Part1"] = ...`; it raises from 4.0 on (P6.6). | `data/polydata.py` | – |

**Tests:** the alignment checks of §3.16 ("subtree: checks") as a test module; the §2.9 characterization tests flip; moving a block in the tree leaves its stored arrays untouched; ids before and after `lock()` agree.

**Exit:** release 3.8, the last 3.x release with new features.

### P6. Block configuration and plotting — release 4.0 (§3.6, §3.10 plotting, §6 step 8) — not started

| # | Work | Files | Status |
|---|---|---|---|
| P6.1 | Pydantic models `Attrs` (with `Cascade()` markers, `extra="allow"`), `Style` (`extra="forbid"`), `BlockConfig` (`attrs`, `style`, `backend_styles`, `named_styles`, `meta`); `config_class` on `PolyData`. Local values only through `model_fields_set`; a subclass that redeclares an attribute name raises. Values under undeclared keys of `Attrs` must be JSON-serializable, checked when they are set; keys that aren't valid Python names are reached through `get` / `set` methods of `Attrs`. | `data/blockconfig.py` (new) | – |
| P6.2 | Replace `PolyData.config`, `_init_config_`, `_get_config_` and the `("pv", "default")` keys. Port `LineData` and `TriMesh` defaults to presets (P6.4). | `data/polydata.py`, `data/linedata.py`, `data/trimesh.py`, `plotting/pvplot.py` | – |
| P6.3 | `resolved_attrs()`; `resolved_style(backend, name)` returning a new `Style`, with the lookup order of §3.6 (block override, block neutral, each ancestor the same way, preset, base default); named styles; backend options cascading per option. Volume inputs fall back to `config.attrs.t` / `A` and raise when the chain ends (P5.4). | `data/polydata.py`, `data/blockconfig.py` | – |
| P6.4 | Plotting registries: backends (with availability from `config.py`), renderers by backend and cell type with lookup by label, `base_label` chain, then dimension; presets keyed the same way, mutable for the session through `presets()`, never saved. Entry point group `sigmaepsilon.mesh.renderers`. | `plotting/registry.py` (new), `plotting/renderers/` (new) | – |
| P6.5 | PyVista renderers for 1D, 2D and 3D cells and matplotlib renderers for 1D and 2D cells, built from today's `pvplot`, `triplot` and `parallel` code. `plot(backend=None, style="default", skip_unsupported=False, **kwargs)` with the backend rules of §3.10; `pvplot` as a shortcut. | `plotting/`, `data/polydata.py` | – |
| P6.6 | Only `PolyData` blocks as items, and strict reads (section 5, point 3). `__setitem__` raises `TypeError` for anything else, naming `block.config.attrs` and `block.config.meta` as the places for other data (§3.6). `__getitem__` raises `KeyError` for a missing key whether or not the mesh is locked, with close matches from `difflib` (`'Solds' (did you mean 'Solids'?)`); nested blocks are created in one step with a tuple key, `mesh["A", "Part1"] = PolyData(...)`, which `DeepDict.__setitem__` already resolves through `__missing__`. No change to `deepdict` is needed. | `data/polydata.py` | – |
| P6.7 | Remove k3d and Plotly: `k3dplot`, `to_k3d`, `plotting/k3dplot.py`, `io/to_k3d.py`, `plotting/plotly/`, `LineData.__plot_plotly__`, the `__hask3d__` and `__hasplotly__` flags, `tests/plotting/test_plotly.py`, the dependency group entries and `sphinx-plotly-directive`. `docs/source/notebooks/plotting.ipynb` switches to PyVista. | as listed in §3.10 | – |

**Tests:** the examples of §3.6 and §3.10 as tests (resolution order, nearer neutral beats farther override, presets last, metadata never cascades, validation errors); the §2.5 characterization tests flip; renderer tests with the `agg` matplotlib backend and PyVista off-screen.

### P7. Persistence, versioning and migrations — release 4.0 (§3.12, §3.13, §3.10 classes, §6 step 5) — not started

| # | Work | Files | Status |
|---|---|---|---|
| P7.1 | `to_dict()` / `from_dict()`: tree of block paths with container class and `base_label` chain, configuration (local values), point and cell fields with specs (`persist=False` skipped), cell type label and chain, frames in their form, merged schemas with owners and schema versions, format marker and versions. | `io/description.py` (new) | – |
| P7.2 | Class resolution on load (§3.12, "Classes on load"): saved class if it subclasses `cls`, `TypeError` otherwise, block by block; unregistered labels through the entry point group `sigmaepsilon.mesh.classes`, read lazily; fallback along `base_label` with a warning; `resolve="base"`. | `data/registry.py`, `io/description.py` | – |
| P7.3 | Backend registry by suffix (case-insensitive, one backend per suffix); `save` / `load` choose by suffix and raise on a missing or unknown one; format marker checked first. | `io/backends/__init__.py` (new) | – |
| P7.4 | `.npz` backend: one member per field, the description as a JSON member. | `io/backends/npz.py` (new) | – |
| P7.5 | HDF5 backend (`h5py`, new extra `hdf5`, flag `__hash5py__` in `config.py`): the layout sketched in §3.12, datasets chunked along the item axis, gzip per chunk. | `io/backends/hdf5.py` (new) | – |
| P7.6 | Versioning and migrations: `StoredBlock` (lazy, dict-like view of a stored block that exposes only the fields of one owner), `@migration(package, from_version, to_version)`, the runner that chains them on load, the load policy table of §3.13 (`migrate`, `on_newer`, `on_missing`), opaque fields kept and written back, `sigmaepsilon.mesh.io.upgrade(path)`. | `io/migrations.py` (new) | – |
| P7.7 | Sample files: one small file per schema version of `mesh`, in `tests/data/`, written by a script kept next to them, and a test that loads each. | `tests/data/`, `tests/io/` | – |
| P7.8 | Lossy one-way exports stay (`to_vtk`, `to_pv`, meshio), now built on the subtree methods of P5 (global point ids, `celltypes()`, fields every block has). `to_parquet` stays a data export. | `io/to_vtk.py`, `io/to_pyvista.py` | – |

**Tests:** round trip of every example mesh of the test suite through both backends, compared field by field, spec by spec, frame by frame and config by config; class resolution cases of §3.12 with a dummy extension package installed in the test environment through an entry point (a small package under `tests/fixtures/` installed in editable mode by the `test` group); missing package, newer file and migration chains with synthetic owners.

### P8. Removals and release — release 4.0 — not started

| # | Work | Status |
|---|---|---|
| P8.1 | Remove every alias of P2 and P3.5, `_attr_map_` support, `AkWrapper`, the `__getattr__` fall-through and undeclared attribute assignment (both raise `AttributeError`), `pim` / `cim`, `cell_ids()`, `rotate`, `spin`, the 1.0 volume default, block creation on read, `PolyCell`, `data/akwrapper.py`, `data/polycell.py`, `typing/abcakwrapper.py`, `typing/abcpolycell.py`. | – |
| P8.2 | Migration guide in the docs: a table of old and new names (from §3.15), the config change, plotting backends, files. | – |
| P8.3 | Rewrite `docs/source/notebooks/data_model.ipynb`, `data.ipynb`, `mesh_structure.ipynb`, `mesh_composition.ipynb`, `plotting.ipynb`, `io.ipynb` against 4.0. Every example that creates a parent block implicitly (`mesh["A"]["Part1"] = ...` without `mesh["A"]`) switches to the tuple key, starting with the `PolyData` docstring. | – |
| P8.4 | Set the proposal's status to implemented and link the release. | – |
| P8.5 | Publish `4.0.0rc1` to PyPI; fix what the dependants' suites find, with further candidates as needed (section 1.4). | – |
| P8.6 | Cut `maint/3.x` from the 3.8 tag, add its branch filter to the CircleCI publish job, and add the `3.x` version on Read the Docs (section 1.4). | – |

**Exit:** `mesh` 4.0; `fem`, `solid.fourier` and `solid.material` released against it with their caps lifted to `<5.0.0`, `fem` the same day.

---

## 3. Dependency graph

```text
P0 ─ P1 ─ P2 ─┬─ P3 ─┬─ P5 ─┐
              ├─ P4 ─┘      ├─ P7 ─ P8
              └─ P6 ────────┘
```

- P3 and P4 can be developed in parallel once P2 is merged.
- P6 depends on P2 for the container registry (renderer and preset lookup by label). Two items reach further: the volume fallback of P6.3 builds on P5.4, and the strict reads of P6.6 replace the warning of P5.6. Everything else in P6 can start early, following §6, on a `release/4.0` branch kept rebased on `dev`.
- P7 needs P3 (specs), P4 (frame forms), P5 (one walk order) and P6 (configuration to save).

---

## 4. Work in other repositories

| Repository | Needed for | Work |
|---|---|---|
| `sigmaepsilon.math` | P5.5 | `JaggedArray` with regular trailing axes, if it lacks them. |
| `sigmaepsilon.deepdict` | P1.5 | Optional: a `__reduce__` that subclasses can extend with instance state, instead of overriding it in `PolyData`. Not blocking. |
| `sigmaepsilon.solid.fem` | P2, P3, P5, P6, P7 | Cells subclass `mesh`'s cell types (P2); fields in nested `Schema`s, `__schema_version__ = 1`, `K` and other caches with `persist=False`, own `activity` flag dropped or renamed (§2.2) (P3); remove its own id and walk code (P5); `BeamStyle` / `BeamConfig`, renderers and presets, `LineMesh._init_config_` removed (P6); entry points `sigmaepsilon.mesh.classes` and `sigmaepsilon.mesh.renderers`, its own sample files (P7). |
| `sigmaepsilon.solid.material` | P6, P8 | Check `bernoulliframe.py`'s use of `LineSection` and `triplot_mpl_mesh` against the new plotting layer. Capped at `<4.0.0`; CI job against `mesh` `dev`; cap lifted to `<5.0.0` after it passes against the 4.0 candidate (section 1.4). |
| `sigmaepsilon.solid.fourier` | P2, P8 | The package itself doesn't import `mesh`; its tests, docs notebooks and `apps/` use `grid`, `gridQ4`, `triangulate`, `Q4_to_T3`, `triplot_mpl_data` and `PointData` (check for `coords=`, renamed in P2.4). Capped at `<4.0.0`; CI job against `mesh` `dev`; cap lifted to `<5.0.0` after it passes against the 4.0 candidate (section 1.4). |

---

## 5. Decisions taken while planning

These came up while mapping the proposal onto the code. None changes a decision of §4 of the proposal; they fill in what it leaves open.

1. **Major version of the aliases — decided.** The aliases of P2 and P3.5 ship in 3.6–3.8 with deprecation warnings and are removed in 4.0, together with the configuration change (P8.1).
2. **Reading an unset core field — decided.** A core field that has no value yet reads as `None`, through the property, `get()` and `core[...]` alike (§3.5 shows `cd.K` returning `None`). The rule of §3.4 that reads return an `ndarray` applies to fields that have a value.
3. **Typos in keys — decided.** From 4.0 on, reading a missing key of a `PolyData` raises `KeyError`, also on an unlocked mesh, and suggests close matches; nested blocks are created with a tuple key (P6.6). In 3.8, a read that creates a block warns with `FutureWarning` and names the tuple-key form (P5.6). Items other than `PolyData` blocks are rejected (P6.6): data of the user goes to undeclared keys of `config.attrs`, data of other tools to `config.meta`, values per point or cell to `data`. `lock()` keeps its role of caching ids and subtree results (§3.7).
4. **Volume default in 3.x — decided.** A missing `t` or `A` still counts as 1.0 in 3.8 but warns with `FutureWarning`; from 4.0 on it raises `ValueError` (P5.4).
5. **Read-only frame views — settled by the code.** `ArrayWrapper.__init__` in `sigmaepsilon.math` always copies the axes (`np.array(buf, dtype=float)`), and `orient` assigns a new array instead of writing in place. So `cd.frames` returns a frame over a copy, and nothing done to it can reach the block; no change to `math` is needed. The cost is one copy per access, about 7 MB for 10⁵ per-cell frames, so internal kernels read the frame storage directly (P4.4) and never go through `cd.frames`.
6. **`activity` in fem — decided.** fem drops its own `activity` field and uses the `activity` declared by `mesh`: both mean the same, and fem's assembly already reads `mesh.cell_activity()`. A flag with another meaning gets its own name (like `dof_activity`).

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| The class merge of P2 touches every cell type and every fem element at once. | P2 changes no behaviour; fem's suite runs against each P2 PR; aliases keep old imports working. |
| Losing the `__getattr__` fall-through breaks user code that reads Awkward attributes through a cell block. | P3.4 warns on every fall-through that succeeds, naming the attribute, and P8.1 removes it in 4.0. The proposal removes it outright; the warning period is an addition of this plan, to keep the 3.x promise of section 1.1. |
| Uniform values and read-only frame views break kernels that assume full arrays. | Kernels receive expanded arrays through `get(resolve=True)`; only storage stays compact. Benchmarks of P0.3 track the cost. |
| The 4.0 branch drifts from `dev` while P3–P5 land. | Rebase `release/4.0` after every 3.x release. P6 mostly touches configuration and plotting, which P3–P5 leave alone; the overlaps are P6.3 / P5.4 and P6.6 / P5.6, both in `PolyData`. |
| Saved files lock in mistakes of the first format. | P7 lands as the last feature before 4.0, after P3–P6 settle; format version 1 is tested against the sample files before release. |
