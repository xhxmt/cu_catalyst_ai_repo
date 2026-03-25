## Goal

Integrate 226 rows of O and OH adsorption energy data from `data/full_dataset.csv`
(MamunHighT2019, BEEF-vdW, FCC-111, 36 metals) into the existing ML pipeline alongside
the current 375-row CO dataset, producing a ~550-row multi-adsorbate training set
(after SP-metal exclusion) with d-metal coverage expanding from 11 → ~25 metals.

Execute in Option C sequence: validate O+OH loader in isolation (I-1) before merging
with CO data (I-2).

---

## Assumptions

- `full_dataset.csv` is FCC (111) slab data — all rows get `facet = "111"` and
  `dft_functional = "BEEF-vdW"`.
- SP metals (Al, Ga, Bi, Cd, Hg, In, Sn, Tl, Zn) are excluded from both I-1 and I-2
  because they have no meaningful d-band center.
- The existing Ruban 1997 d-band scale (all negative, Fermi-level referenced) is used
  throughout; Nørskov vacuum-referenced positive values are NOT mixed in.
- `_ELEMENT_DATA` already covers most d-metals. Only Zr, Hf, Y are d-metals missing
  from it entirely.
- No existing A–H experiment is modified; all changes are behind new config keys or in
  new files.

---

## Plan

### Step 1 — Extend `_ELEMENT_DATA` with missing d-metals [~5 min]

**Files**: `src/cu_catalyst_ai/features/element_features.py`

**Change**: Add three d-metal entries using Ruban 1997 / literature Fermi-level
values for Zr (−3.26 eV), Hf (−3.05 eV), Y (−2.45 eV). These are d-metals present
in the Mamun dataset but currently missing from the lookup table, which causes their
rows to get NaN for all element features.

```python
"Zr": {"d_band_center": -3.26, "work_function": 4.05, "electronegativity": 1.33, "atomic_radius_pm": 175, "d_electron_count": 2},
"Hf": {"d_band_center": -3.05, "work_function": 3.90, "electronegativity": 1.30, "atomic_radius_pm": 167, "d_electron_count": 2},
"Y":  {"d_band_center": -2.45, "work_function": 3.10, "electronegativity": 1.22, "atomic_radius_pm": 190, "d_electron_count": 1},
```

**Verify**:
```bash
uv run python -c "
from cu_catalyst_ai.features.element_features import get_element_features
for el in ['Zr', 'Hf', 'Y']:
    print(el, get_element_features(el))
"
uv run pytest tests/test_element_features.py -v
```

---

### Step 2 — Extend `_SURFACE_DBAND_MAP` for new metals' (111) face [~5 min]

**Files**: `src/cu_catalyst_ai/features/basic_features.py`

**Change**: Add `(element, "111")` entries for all d-metals that are in the Mamun
dataset but NOT yet in `_SURFACE_DBAND_MAP`. These reuse the `d_band_center` values
from `_ELEMENT_DATA` (Ruban 1997 bulk = (111) close-packed approximation):

```python
("Sc", "111"): -2.13,   # [L] Ruban 1997 / _ELEMENT_DATA
("Ti", "111"): -2.76,   # [L]
("V",  "111"): -2.65,   # [L]
("Cr", "111"): -2.55,   # [L]
("Mn", "111"): -2.18,   # [L]
("Fe", "111"): -2.29,   # [L]
("Zr", "111"): -3.26,   # [L] newly added
("Nb", "111"): -3.20,   # [L] already in _ELEMENT_DATA
("Mo", "111"): -2.95,   # [L]
("Tc", "111"): -2.72,   # [L]
("Hf", "111"): -3.05,   # [L] newly added
("Ta", "111"): -3.25,   # [L]
("W",  "111"): -2.86,   # [L]
("Re", "111"): -2.58,   # [L]
("Os", "111"): -2.22,   # [L]
("Y",  "111"): -2.45,   # [L] newly added
```

Metals already in `_SURFACE_DBAND_MAP` (Cu, Ag, Au, Co, Fe, Ir, Ni, Pd, Pt, Rh, Ru)
are NOT touched.

**Verify**:
```bash
uv run python -c "
from cu_catalyst_ai.features.basic_features import _SURFACE_DBAND_MAP
for el in ['Ti', 'Zr', 'Hf', 'Mo', 'Re']:
    print(el, _SURFACE_DBAND_MAP.get((el, '111')))
"
uv run pytest tests/test_features.py -v
```

---

### Step 3 — Write `mamun_loader.py` [~10 min]

**Files**: `src/cu_catalyst_ai/dataio/mamun_loader.py` [NEW]

**Change**: New module with one public function `load_mamun_ooh()` that:
1. Reads `data/full_dataset.csv`.
2. Filters `ads_1` to `{"O", "OH"}`.
3. Filters pure d-metals only (regex `^[A-Z][a-z]?$` + exclusion list of SP metals).
4. Maps columns to canonical schema:
   - `reaction_energy` → `adsorption_energy`
   - `surface_composition` → `element`
   - `ads_1` → `adsorbate`
   - `facet = "111"` (hard-coded; MamunHighT2019 is FCC 111)
   - `dft_functional = "BEEF-vdW"` (hard-coded)
   - `target_definition = "adsorption_energy_ev_multi_v1"`
   - `provenance = "MamunHighT2019|10.1021/acscatal.0c01752|2020"`
   - `unit_adsorption_energy = "eV"`
   - `catalyst_id` = `f"mamun_{element}_{adsorbate}_{site_1}_{rowindex}"`
5. Retains only canonical columns (same list as `cathub_fetch._CANONICAL_COLUMNS`).
6. Drops rows with NaN `adsorption_energy`.
7. Logs count by adsorbate and count of dropped SP-metal rows.

```python
SP_METALS = frozenset({"Al", "Ga", "Bi", "Cd", "Hg", "In", "Sn", "Tl", "Zn", "Pb", "Ge"})
```

**Verify**:
```bash
uv run python -c "
from cu_catalyst_ai.dataio.mamun_loader import load_mamun_ooh
df = load_mamun_ooh('data/full_dataset.csv')
print(df.shape, df['adsorbate'].value_counts().to_dict())
print(df['element'].unique().tolist())
print(df.columns.tolist())
print(df.isna().sum())
"
```

---

### Step 4 — Register new target definition YAML [~3 min]

**Files**: `configs/target/adsorption_energy_ev_multi_v1.yaml` [NEW]

**Change**:
```yaml
name: adsorption_energy_ev_multi_v1
column: adsorption_energy
canonical_unit: eV
unit_column: unit_adsorption_energy
required_adsorbate: null   # accepts O, OH, CO — no adsorbate filtering

supported_unit_conversions:
  eV: 1.0
  meV: 0.001
  kJ/mol: 0.010364

hard_invalid_rules:
  avg_neighbor_distance_le_zero: true
  coordination_number_lt_zero: true
  adsorption_energy_non_numeric: true

review_bounds:
  adsorption_energy_abs_max: 12.0   # O can reach ~-8 eV; widen from 10 to 12
  surface_energy_min: 0.0
  electronegativity_min: 0.0
  electronegativity_max: 4.0
```

Note: `required_adsorbate: null` means `cli.py` line 163 reads `None`, and
`validate_target_definition()` is called with `required_adsorbate="None"` — need to
verify the None handling in `target_validator.py`. Actually `cli.py` line 163 does
`str(... or "CO")` which will convert `None` to `"CO"`. **This must be fixed** in
Step 6 (CLI change).

**Verify**:
```bash
uv run python -c "
import yaml
with open('configs/target/adsorption_energy_ev_multi_v1.yaml') as f:
    print(yaml.safe_load(f))
"
```

---

### Step 5 — Add `add_adsorbate_ohe()` to `basic_features.py` [~8 min]

**Files**: `src/cu_catalyst_ai/features/basic_features.py`

**Change**: Add new function after `add_gcn()`:

```python
def add_adsorbate_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot encoded adsorbate columns: is_CO, is_O, is_OH.

    Noop when 'adsorbate' column is absent (preserves backward compat).
    """
    if "adsorbate" not in df.columns:
        return df
    df = df.copy()
    for ads in ("CO", "O", "OH"):
        df[f"is_{ads}"] = (df["adsorbate"] == ads).astype(int)
    return df
```

Also update `_run_featurize` in `cli.py` to call `add_adsorbate_ohe()` (Step 6).

**Verify**:
```bash
uv run python -c "
import pandas as pd
from cu_catalyst_ai.features.basic_features import add_adsorbate_ohe
df = pd.DataFrame({'adsorbate': ['CO', 'O', 'OH', 'CO']})
print(add_adsorbate_ohe(df))
# Noop test
df2 = pd.DataFrame({'electronegativity': [1.9]})
assert 'is_CO' not in add_adsorbate_ohe(df2).columns
"
uv run pytest tests/test_features.py -v
```

---

### Step 6 — CLI merge hook + fix required_adsorbate None handling [~15 min]

**Files**: `src/cu_catalyst_ai/cli.py`, `configs/data/cathub.yaml`

**Change A — `_run_featurize`**: Add `add_adsorbate_ohe()` call after `add_surface_dband()`:
```python
from cu_catalyst_ai.features.basic_features import add_adsorbate_ohe
enriched_df = add_adsorbate_ohe(enriched_df)
```

**Change B — `_run_clean`**: Fix the `required_adsorbate` read on line 163:
```python
# OLD:
required_ads = str(_cfg_get(cfg, "target.required_adsorbate") or "CO")
# NEW:
_raw_ads = _cfg_get(cfg, "target.required_adsorbate")
required_ads = None if _raw_ads is None else str(_raw_ads)
```
And update `validate_target_definition()` call to pass `required_adsorbate=required_ads`.
Also update `target_validator.py` to skip adsorbate check when `required_adsorbate is None`.

**Change C — `_run_fetch` (cathub branch)**: After fetching CO parquet, check for
optional `extra_data.mamun_path` config key. If present and non-null, call
`load_mamun_ooh()` and concat to the raw dataframe before writing:
```python
mamun_path = _cfg_get(cfg, "data.extra_data.mamun_path")
if mamun_path:
    from cu_catalyst_ai.dataio.mamun_loader import load_mamun_ooh
    mamun_df = load_mamun_ooh(str(mamun_path))
    raw_df = pd.concat([raw_df, mamun_df], ignore_index=True)
    logger.info("Appended %d Mamun O/OH rows; total: %d", len(mamun_df), len(raw_df))
```

For I-1 (O+OH only), a **new data config** `configs/data/mamun_ooh.yaml` will use
`source_name: mamun` and bypass the cathub-fetch entirely, just calling
`load_mamun_ooh()` directly. This keeps the fetch stage clean.

Actually simpler: use a new source type `"mamun"` in `_run_fetch`, which directly calls
`load_mamun_ooh()` and writes the parquet. This is the cleanest approach for I-1.

**Verify**:
```bash
# target_validator None handling
uv run python -c "
from cu_catalyst_ai.clean.target_validator import validate_target_definition
import pandas as pd
df = pd.DataFrame({'target_definition': ['adsorption_energy_ev_multi_v1', 'adsorption_energy_ev_multi_v1'],
                   'adsorbate': ['O', 'OH'], 'adsorption_energy': [-1.5, -2.0],
                   'unit_adsorption_energy': ['eV', 'eV']})
out = validate_target_definition(df, 'adsorption_energy_ev_multi_v1', required_adsorbate=None)
print(out.get('review_reason', 'no flag'))
assert 'review_reason' not in out.columns, 'O/OH rows should NOT be flagged'
print('OK - no rows flagged')
"
uv run pytest tests/ -v -k "not test_cathub"
```

---

### Step 7 — New experiment configs [~5 min]

**Files**:
- `configs/data/mamun_ooh.yaml` [NEW] — source for I-1
- `configs/data/cathub_multi.yaml` [NEW] — source for I-2 (cathub + mamun)
- `configs/features/cathub_multi_ads.yaml` [NEW]

**`configs/data/mamun_ooh.yaml`**:
```yaml
source_name: mamun
mamun_path: data/full_dataset.csv
raw_output: data/raw/mamun/mamun_ooh_raw.parquet
cleaned_output: data/interim/mamun_ooh_cleaned.parquet
processed_output: data/processed/mamun_ooh_model_table.parquet
review_output: data/interim/mamun_ooh_review.parquet
target_definition: adsorption_energy_ev_multi_v1
```

**`configs/features/cathub_multi_ads.yaml`**:
```yaml
name: cathub_multi_ads
use_columns:
  - electronegativity
  - d_band_center
  - gcn
  - is_CO
  - is_O
  - is_OH
categorical_columns: []
```

**Verify**:
```bash
uv run python -c "
import yaml
for p in ['configs/data/mamun_ooh.yaml', 'configs/features/cathub_multi_ads.yaml']:
    print(p, yaml.safe_load(open(p)))
"
```

---

### Step 8 — Run I-1: O+OH only [~5 min runtime]

```bash
uv run python -m cu_catalyst_ai.cli \
  data=mamun_ooh \
  features=cathub_multi_ads \
  model=rf \
  target=adsorption_energy_ev_multi_v1 \
  task=baseline
```

**Pass/fail threshold**: R² ≥ 0.20 on test set. If R² < 0.20, stop — investigate
loader or d-band feature issues before proceeding to I-2.

Check output:
```bash
cat reports/tables/metrics.csv
# Expect: element column has ≥10 unique metals in cleaned parquet
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/interim/mamun_ooh_cleaned.parquet')
print(df['element'].value_counts())
print(df['adsorbate'].value_counts())
"
```

---

### Step 9 — Run I-2: CO+O+OH merged [~varies]

Update `configs/data/cathub_multi.yaml` to include `extra_data.mamun_path` pointing to
`data/full_dataset.csv`. The fetch stage will concat CO+O+OH.

```bash
uv run python -m cu_catalyst_ai.cli \
  data=cathub_multi \
  features=cathub_multi_ads \
  model=rf \
  target=adsorption_energy_ev_multi_v1 \
  task=baseline

uv run python -m cu_catalyst_ai.cli \
  data=cathub_multi \
  features=cathub_multi_ads \
  model=gpr \
  target=adsorption_energy_ev_multi_v1 \
  task=baseline
```

**Additional validation** — CO-subset R²:
```bash
uv run python -c "
import pandas as pd
from sklearn.metrics import r2_score
pred = pd.read_parquet('data/processed/predictions.parquet')
co = pred[pred['adsorbate'] == 'CO']
print('CO-subset R2:', r2_score(co['adsorption_energy'], co['predicted']))
"
```

---

## Risks & Mitigations

| # | Risk | Mitigation |
|---|------|------------|
| R1 | SP metals cause NaN in d_band_center | Excluded in `load_mamun_ooh()` via `SP_METALS` set |
| R2 | `required_adsorbate: null` YAML → Python `None` → `str(None or "CO")` = `"CO"` | Fix in `cli.py` line 163 (Step 6 Change B) + `target_validator.py` |
| R3 | `validate_target_definition` flags O/OH rows as adsorbate mismatch | Fixed by `required_adsorbate=None` skip logic in `target_validator.py` |
| R4 | Energy range: O adsorption can reach −8 eV (beyond old ±10 limit) | New target YAML sets `adsorption_energy_abs_max: 12.0` |
| R5 | `_SURFACE_DBAND_MAP` missing new metals → fallback to `d_band_center` from `_ELEMENT_DATA` | Step 2 adds `(element, "111")` entries; Step 1 adds Zr/Hf/Y to `_ELEMENT_DATA` |
| R6 | `add_adsorbate_ohe()` called on CO-only data without `adsorbate` column | Function is noop when column absent; A–H configs won't have `is_CO/O/OH` in `use_columns` anyway |
| R7 | Backward compat: A–H experiments | New configs are in separate YAML files; existing cathub.yaml is untouched |

---

## Rollback Plan

- Steps 1–2 (dict edits): revert with `git checkout src/cu_catalyst_ai/features/`.
- Step 3 (new file): `git rm src/cu_catalyst_ai/dataio/mamun_loader.py`.
- Steps 4, 7 (new YAML files): `git rm configs/target/adsorption_energy_ev_multi_v1.yaml configs/data/mamun_ooh.yaml configs/features/cathub_multi_ads.yaml`.
- Step 5 (new function in basic_features.py): revert with `git checkout src/cu_catalyst_ai/features/basic_features.py`.
- Step 6 (cli.py changes): revert with `git checkout src/cu_catalyst_ai/cli.py src/cu_catalyst_ai/clean/target_validator.py`.
- Existing experiments (A–H, CO data) are unaffected in all failure scenarios.
- Full revert: `git stash` or `git checkout HEAD~1` if on a feature branch.
