# What `solid.fem` and `solid.material` use from `mesh`

**Implements:** P0.4 of [data-model-implementation-plan.md](data-model-implementation-plan.md). Section numbers (§) refer to [data-model-proposal.md](data-model-proposal.md).

**Snapshot:** `sigmaepsilon.solid.fem` at `origin/dev` `6ce77ab`, `sigmaepsilon.solid.material` at `origin/main` `d427cb6`.

**Use:** each port PR in `solid.fem` copies the items of its phase into its description and ticks them off. An item is done when the use is gone or moved to the new API, and `solid.fem`'s suite passes against the phase branch of `mesh` with `mesh`'s `DeprecationWarning` and `FutureWarning` turned into errors (section 1.4 of the plan).

**Method:** a regular-expression search of `src/`, `tests/` and `docs/` for every name, keyword and hook the plan renames, merges or removes, then a reading of each hit in `src/`. Locations below are `file:line` under `src/sigmaepsilon/solid/fem/`. Hits in `tests/` and `docs/` are counted, not listed: they follow the source and are fixed with it. Notebooks under fem's `sandbox/` are left out. Keyword arguments are matched in Black's spelling (`topo=x`), so assignments such as `pd = PointData(...)` aren't counted. Nested assignments in notebooks (`mesh["A"]["B"] = ...`) weren't searched, because the notebook JSON escapes the quotes.

---

## 1. `solid.fem`

### P1 (release 3.5): nothing to port

fem doesn't call `set_nodal_distribution_factors`, `ndf`, `IndexManager`, `PointCloud.closest`/`furthest`, the `to_dict`/`to_list` exports or `pickle`. P1 changes no API it uses.

### P2 (release 3.6): class layout and renames

- [ ] **`PolyCell` → `CellData`** (P2.2): imported and subclassed in 19 source files.
  - The cell classes: `cells/andes.py:11`, `bernoulli2.py:1`, `bernoulli3.py:1`, `cook.py:8`, `cst.py:1`, `h27.py:1`, `h8.py:1`, `lst.py:1`, `q4.py:1`, `q8.py:1`, `q9.py:3`, `tet10.py:1`, `tet4.py:1`, `timoshenko2.py:1`, `timoshenko3.py:1`.
  - The data layer and the boundary conditions: `data/finiteelement.py:33,73,77`, `ebc/body2body.py:8,29,31,53,54,90,91`, `ebc/node2node.py:8,57,58`, `utils/ebc.py:9,25`.
  - P2.6 rebases the cell classes on `mesh`'s cell types (`class H8Solid(FemCellData, H8)`).
- [ ] **`data_class`** (P2.2, removed without an alias): `data/finiteelement.py:140` sets `data_class = CellData[FD, PD]`, fem's own cell data. The merged `CellData` takes over its role in P2.6.
- [ ] **`.db`** (P2.2, then returns the instance itself with a warning): 40 uses in 11 source files. Most read or write fields of the wrapped cell data through `self.db.<field>`.
  - The cell data: `data/finiteelement.py:164–246,475`, `data/finiteelement1d.py:85,86,219`, `data/shellelement.py:300–306,402,458–487`, `data/element_results.py:412–416,498`, `data/membraneelement.py:149`, `data/plateelement.py:39`.
  - Other modules: `cells/allman84.py:282,284`, `data/celldata.py:228`, `data/mesh.py:378`, `data/pointdata.py:102,152`, `model/truss.py:54,61`.
  - Tests: 18 hits in 3 files.
- [ ] **`ABCMeta_PolyCell`** (P2.3, alias of `DataMeta`): `typing/abcfemcell.py:3,90`. `ABCMeta_FemCell` subclasses it and checks `Model` and `Geometry` when a class is created. `DataMeta` must stay subclassable for this, or fem moves the checks to `__init_subclass__`.
- [ ] **Type variables** (P2.1): `PointDataLike` and friends in `typing/celldata.py:15,19,20` and `typing/finiteelement.py:24,29,30`. Port them to `PointDataT`, `CellDataT` and `PolyDataT`.
- [ ] **Geometry and protocol names** (P2.4):
  - `PolyCellGeometry`, `PolyCellGeometry1d` and `PolyCellGeometry3d` in `approximator.py:8,17` and `cells/h20.py:8,19`. Rename to `CellGeometry*`.
  - `GeometryProtocol` and `PolyCellProtocol` in `typing/abcfemcell.py:4,121,124`, `typing/celldata.py:5,19` and `data/element_results.py:19,80`. Rename to `CellGeometryProtocol` and `CellDataProtocol`.
- [ ] **`_point_class_`** (P2.4 → `pointdata_class`): `data/mesh.py:112`, where `FemMesh._point_class_ = PointData`.
- [ ] **`.pd` / `.cd`** (P2.4 → `pointdata` / `celldata`): 37 source hits in `data/mesh.py` (24) and `structure.py` (13), plus one in `benchmarks/models.py:661`. Tests: 69 hits in 15 files. Docs: 16 hits in 10 files.
- [ ] **Keywords `coords=` / `topo=`** (P2.4 → `x=` / `topology=`), also `pd=` / `cd=` (→ `pointdata=` / `celldata=`):
  - Source: `coords=`/`topo=` in 13 files: `benchmarks/models.py:237,239,623`, `benchmarks/plates.py:171,173`, `cells/truss.py:31,59`, `cookbook/linstat_membrane_q9.py:107,108`, `data/finiteelement.py:160`, `data/finiteelement1d.py:363,446`, `data/mesh.py:98–101` (docstring), `data/pointdata.py:68` (docstring), `data/surfaceelement.py:193`, `data/volumeelement.py:116`, `examples.py:95,99`, `model/truss.py:261`, `structure.py:75,76`.
  - Tests and docs: `coords=`/`topo=` has 92 hits in 19 test files and 84 in 25 docs files. `pd=`/`cd=` has 2 hits in tests and 6 in docs.
  - `FiniteElement.__init__` (`data/finiteelement.py:158`) already accepts both `topo` and `topology`; keep `topology` only.
- [ ] **Container registry** (P2.5): `FemMesh` and `LineMesh` register through `__init_subclass__`. They need a `label`, and no metaclass change: `FemMesh` keeps `ABC_FemMesh`.

### P3 (release 3.7): fields, schemas and storage

- [ ] **`_attr_map_`** (P3.3 → nested `Schema`):
  - `data/celldata.py:25` declares 10 fields: `body-loads`, `strain-loads`, `density`, `activity`, `fixity`, `K`, `C`, `M`, `B` and `f`, with `_..._` internal names.
  - `data/pointdata.py:76` declares 7 fields: `loads`, `mass`, `fixity`, `dofsol`, `shapes`, `reactions` and `forces`. Except `shapes`, the internal names equal the public ones, so they are unprotected (§2.1).
  - `body-loads` and `strain-loads` aren't valid Python names. The schema needs new names, such as `body_loads`; that is a breaking change for fem users who pass these keys in `fields`.
  - `K` and other caches get `persist=False` (section 4 of the plan).
- [ ] **`activity`** (decision 6 of the plan): fem's `"activity": "_activity_"` loses to `mesh`'s `_activity` today (§2.2; pinned by `tests/design/test_audit_attr_map.py`). fem drops its own field and uses `mesh`'s. `assemble/linstat.py:80` already reads `mesh.cell_activity()`.
- [ ] **`_dbkey_*` class properties** (P3.5 shim, removed in 4.0): 42 in `data/celldata.py:39–272` and 11 in `data/pointdata.py:96–248`, plus `cells/allman84.py:281` and `data/element_results.py:412`. They go away with the `Schema`. Tests: 2 hits.
- [ ] **`_wrapped`, reaching into the Awkward record** (P3.4): `data/celldata.py:82–272` (32 hits), `data/pointdata.py:195–298` (17), `data/element_results.py:102,414,416`, `structure.py:594`.
  - `structure.py:594` and `data/element_results.py:102` test whether a field exists with `pd.__class__._attr_map_["dofsol"] in pd._wrapped.fields`. Replace with a membership test on the fields view (§3.4).
  - Tests: 2 hits.
- [ ] **`__getattr__`** (P3.4): `data/shellelement.py:135` defines a wrapper around `db` that resolves extra fields and `has_<field>` names. Check it against the end of the Awkward fall-through: the wrapped `db` no longer answers Awkward attributes.

### P4 (release 3.7): frames

- [ ] **Assignment of expanded frames** (P4.1, P4.2: `cd.frames` becomes read-only, frames are set in one of three forms):
  - `structure.py:649` sets `celldata.frames = repeat(block.frame.show(), nE)` or `default_frames()`. The shared form replaces the `repeat`.
  - `data/shellelement.py:301,306` sets `self.db.frames` to computed mean-plane frames (per-cell form) and defines a `frames` setter.
  - Tests: 1 hit.
- [ ] **`rotate` / `spin`** (P4.3, warns in 3.7, removed in 4.0): not used in `src/`. Tests: 2 hits in 2 files.

### P5 (release 3.8): ids and methods on a subtree

- [ ] **Ids** (P5.1): fem doesn't use `pim`, `cim`, `cell_ids()` or `_reindex`. Nothing to port.
- [ ] **`topology()`** (P5.3: on `PolyData` it always returns a `JaggedArray` of global point ids): 19 calls.
  - The calls: `cells/bernoulli.py:144`, `cells/cook.py:31`, `data/element_loads.py:360`, `data/element_stiffness.py:177,338`, `data/finiteelement.py:365,415,419,440`, `data/lineelement.py:78`, `data/linemesh.py:52,97`, `data/mesh.py:224,283,311`, `data/shellelement.py:309,317`, `plotting/mpl/section_plot.py:64`, `utils/ebc.py:32`.
  - Check each: the ones on cell data keep a regular array, while the ones on a `FemMesh` get the cascading result.
- [ ] **`cells_coords`** (P5.3): `data/mesh.py:232,240` and `cells/utils/super.py:7,34`. It fails on mixed subtrees today (§2.9).
- [ ] **`areas()` / `lengths()`** (P5.3: `areas()` returns `NaN` where not defined, aligned with `id`): `cells/bernoulli.py:140,143`, `data/element_stiffness.py:428`, `data/finiteelement1d.py:220`.
- [ ] **Shared walk and locking** (P5.2): `cellblocks` (18 hits in 2 files), `source()` (`data/element_transform.py:73,127`, `ebc/body2body.py:68,71`, `model/solid3d.py:115`), `root` (20 hits in 7 files), `lock()` (`_nbc/femload.py:45`, `data/loaddata.py:52`, `structure.py:95,142`) and `blocks_of_cells` (`data/mesh.py:65`). They keep their names; check that ids and cached results agree before and after `lock()`.
- [ ] **Volume inputs** (P5.4: a missing `t` or `A` warns in 3.8 and raises in 4.0): fem's shells and plates set `t` through `FiniteElement.__init__` (`thickness`/`t`). Check that beams and trusses always set `A`.
- [ ] **Implicit creation of parent blocks** (P5.6 warns, P6.6 raises): the `FemMesh` docstring (`data/mesh.py:99–101`) uses `pd['A']['Part1'] = ...`. Switch to `pd['A', 'Part1'] = ...`.

### P6 (release 4.0): block configuration and plotting

- [ ] **`_init_config_`** (P6.2, removed): `data/linemesh.py:106–111` repeats `LineData`'s defaults line by line (§2.5). They become a preset of the plotting layer (P6.4).
- [ ] **Plotly** (P6.7, removed):
  - `data/linemesh.py:3–6` imports `__hasplotly__` and `plot_lines_plotly`.
  - `LineMesh.plot` (`data/linemesh.py:24`) defaults to `backend="plotly"` and uses `plot_lines_plotly` at `:63–94`. Switch the default to PyVista or matplotlib.
  - `data/surfacemesh.py:1` imports `__hasplotly__` from `sigmaepsilon.mesh.config`.
- [ ] **matplotlib helpers** (P6.5): `plotting/mpl/section_plot.py` and the imports of `triplot`, `triplot_data`, `triplot_mpl_data` and `triplot_mpl_mesh` from `sigmaepsilon.mesh.plotting`. Check them against the renderer registry; the free functions may stay as shortcuts.

### P7 (release 4.0): persistence

- [ ] **Entry points** `sigmaepsilon.mesh.classes` and `sigmaepsilon.mesh.renderers` in fem's `pyproject.toml`; `__schema_version__ = 1`; sample files (section 4 of the plan). Nothing to port before P7.

### Public API imported by fem

These names must keep working (directly or as warning aliases) through 3.x:

| Module | Names |
|---|---|
| `sigmaepsilon.mesh` | `PolyData`, `TopologyArray`, `grid`, `triangulate` |
| `.cells` | `H8`, `H27`, `L2`, `L3`, `QuadraticLine` (another name for `L3`), `Q4`, `Q8`, `Q9`, `T3`, `T6`, `TET4`, `TET10`, `W6`, `W18`, `PolyCell` |
| `.data` | `CellData`, `PointData`, `PolyCell` |
| `.cellapproximator` | `LagrangianCellApproximator` |
| `.config` | `__hasmatplotlib__`, `__hasplotly__` |
| `.extrude` | `extrude_T3_W6` |
| `.geometry`, `.geometry.geometry` | `PolyCellGeometry`, `PolyCellGeometry1d`, `PolyCellGeometry3d` |
| `.grid` | `grid`, `gridH8` |
| `.plotting`, `.plotting.plotly` | `triplot`, `triplot_data`, `triplot_mpl_data`, `triplot_mpl_mesh`, `plot_lines_plotly` |
| `.space` | `CartesianFrame`, `PointCloud`, `StandardFrame`, `frames_of_lines` |
| `.typing` | `GeometryProtocol`, `PointDataProtocol`, `PolyCellProtocol`, `PolyDataProtocol` |
| `.typing.abcpolycell` | `ABCMeta_PolyCell` |
| `.utils` | `cells_coords`, `jacobian_matrix_bulk`, `points_of_cells` |
| `.utils.cells.t3` | `shp_T3` |
| `.utils.numint` | `Quadrature`, `Q`, `Gauss_Legendre_*` (line, quad, tri, tet, hex) |
| `.utils.space` | `frames_of_surfaces`, `index_of_closest_point` |
| `.utils.topology`, `.utils.topology.tr` | `L2_to_L3`, `H8_to_H27`, `H8_to_L2`, `H8_to_TET4`, `Q4_to_Q8`, `Q4_to_Q9`, `Q4_to_T3`, `T3_to_T6`, `T6_to_T3` |
| `.utils.tri` | `area_tri`, `glob_to_nat_tri`, `loc_to_glob_tri`, `loc_to_nat_tri`, `nat_to_loc_tri` |

fem also imports some of these under local names (`H8 as H8Cell`, `W6 as Wedge6`, `W18 as Wedge18`, `PointData as Base`); those names are fem's own and need no alias in `mesh`.

---

## 2. `solid.material`

The package itself imports two names: `LineSection` from `sigmaepsilon.mesh.domains.section` and `triplot_mpl_mesh` from `sigmaepsilon.mesh.plotting`, both in `src/sigmaepsilon/solid/material/frame/bernoulliframe.py`.

- [ ] **P6**: check `bernoulliframe.py` against the new plotting layer (section 4 of the plan). Neither name is renamed by the proposal.
- [ ] **Docs notebooks**: 7 uses of `.db` in 1 notebook (P2.2), `coords=`/`topo=` in 2 notebooks (5 hits, P2.4), and `config` (2 hits) and Plotly or k3d (1 hit) in 1 notebook (P6). Its tests don't use `mesh`.
