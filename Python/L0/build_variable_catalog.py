"""
Build a file/variable catalog for the ARISE-SAI climate-index netCDFs.

Layout assumed (root = folder containing the scenario folders):

    root/<scenario>/<metric>/bioclim_indicators_<NNN>.nc                  (seasonal_cycle)
    root/<scenario>/<metric>/<extreme_high|extreme_low>/extreme_indices_<NNN>.nc  (extreme_events)
    root/<scenario>/<metric>/precip_indices_<NNN>.nc                      (precip_indices)

Usage (from a different directory than the data):
    python build_catalog.py /path/to/root
or in a notebook:
    from build_catalog import build_file_catalog, explode_to_variables
"""
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Configuration: plain dictionaries. A dict maps a key -> a value, so
#    FAMILIES["seasonal_cycle"] returns that family's list of variables.
#    Keeping the "rules" here (not buried in loops) means a change in one
#    place propagates everywhere.
# ---------------------------------------------------------------------------

SCENARIOS = ["SSP245", "ARISE_SAI_1p0", "ARISE_SAI_1p5"]

# Ensemble members used for every scenario and metric (the files are named
# ..._006.nc to ..._010.nc). Decision: members 6-10 only, none dropped.
MEMBERS = [6, 7, 8, 9, 10]

# Per-family settings: variables in each file, and how the file is named.
FAMILIES = {
    "seasonal_cycle": {
        "variables": ["mean", "max_month", "min_month", "range", "seasonality"],
        "filename": "bioclim_indicators_{member:03d}.nc",
    },
    "extreme_events": {
        "variables": ["annual_mean", "frequency", "duration",
                      "mean_intensity", "max_intensity"],
        "filename": "extreme_indices_{member:03d}.nc",
    },
    "precip_indices": {
        "variables": ["CDD", "CWD", "PRCPTOT", "R96pTOT"],
        "filename": "precip_indices_{member:03d}.nc",
    },
}

# Per-metric settings: which family, and which extreme subfolders exist.
# [None] means "no extreme subfolder" (file sits directly in the metric folder).
METRICS = {
    "CONCLD": {"family": "seasonal_cycle", "extremes": [None]},
    "O3":     {"family": "seasonal_cycle", "extremes": [None]},
    "FSDS":   {"family": "extreme_events", "extremes": ["high", "low"]},
    "SOLLD":  {"family": "extreme_events", "extremes": ["high", "low"]},
    "SOLSD":  {"family": "extreme_events", "extremes": ["high", "low"]},
    "TSMN":   {"family": "extreme_events", "extremes": ["low"]},
    "TSMX":   {"family": "extreme_events", "extremes": ["high"]},
    "PRECT":  {"family": "precip_indices", "extremes": [None]},
}

# Ensemble members to flag as unreliable: {(scenario, metric, member): "reason"}.
# Use "*" as the metric to flag a member for every metric in that scenario.
# Flagged files stay in the catalog (so nothing silently vanishes) but get
# usable=False. Examples:
#   ("ARISE_SAI_1p5", "*", 7): "drift in control period"   # member 7, all metrics
FLAGGED = {}   # empty: no members dropped


def flag_reason(scenario, metric, member):
    """Look up a flag for this file: specific metric first, then the '*' wildcard."""
    for key in [(scenario, metric, member), (scenario, "*", member)]:
        if key in FLAGGED:
            return FLAGGED[key]
    return None


# ---------------------------------------------------------------------------
# 2. File-level catalog: one row per expected file.
#    We build the list from the rules above and then check the disk, rather
#    than globbing the disk. That way a missing file shows up as
#    exists=False instead of just being absent.
# ---------------------------------------------------------------------------

def build_file_catalog(root):
    root = Path(root)
    rows = []
    for scenario in SCENARIOS:
        for metric, spec in METRICS.items():
            fam = FAMILIES[spec["family"]]
            for extreme in spec["extremes"]:
                for member in MEMBERS:
                    folder = root / scenario / metric
                    if extreme is not None:
                        folder = folder / f"extreme_{extreme}"
                    path = folder / fam["filename"].format(member=member)
                    reason = flag_reason(scenario, metric, member)
                    rows.append({
                        "scenario": scenario,
                        "metric": metric,
                        "family": spec["family"],
                        # "none" (a string) rather than None, so groupby never
                        # has to deal with missing values
                        "extreme": extreme if extreme is not None else "none",
                        "member": member,
                        "path": str(path),
                        "variables": fam["variables"],   # list, one per file
                        "exists": path.exists(),
                        "usable": reason is None,
                        "flag_reason": reason,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Variable-level catalog: one row per variable per file.
#    DataFrame.explode() turns a list column into one row per list item.
# ---------------------------------------------------------------------------

def make_var_id(metric, extreme, variable):
    """One unique, readable name per variable: CONCLD_mean, TSMX_high_frequency, ..."""
    parts = [metric]
    if extreme != "none":
        parts.append(extreme)
    parts.append(variable)
    return "_".join(parts)


def explode_to_variables(file_catalog):
    out = (file_catalog
           .explode("variables")
           .rename(columns={"variables": "variable"})
           .reset_index(drop=True))
    out["var_id"] = [make_var_id(m, e, v) for m, e, v
                     in zip(out["metric"], out["extreme"], out["variable"])]
    return out


# ---------------------------------------------------------------------------
# 4. Sanity checks
# ---------------------------------------------------------------------------

def report_missing(file_catalog):
    missing = file_catalog.loc[~file_catalog["exists"]]
    print(f"{len(file_catalog)} expected files, {len(missing)} missing")
    if len(missing):
        print(missing[["scenario", "metric", "extreme", "member", "path"]]
              .to_string(index=False))


def find_unexpected_files(root, file_catalog):
    """.nc files on disk that the catalog doesn't know about (typos, stray files)."""
    on_disk = {str(p) for p in Path(root).rglob("*.nc")
               if p.relative_to(root).parts[0] in SCENARIOS}
    return sorted(on_disk - set(file_catalog["path"]))


if __name__ == "__main__":
    root = sys.argv[1]
    files = build_file_catalog(root)
    report_missing(files)
    extra = find_unexpected_files(root, files)
    print(f"{len(extra)} unexpected .nc files on disk")
    for p in extra[:20]:
        print("  ", p)

    variables = explode_to_variables(files)
    print(f"{len(files)} file rows -> {len(variables)} variable rows")
    files.to_pickle("file_catalog.pkl")        # pickle keeps the list column intact
    variables.to_csv("variable_catalog.csv", index=False)