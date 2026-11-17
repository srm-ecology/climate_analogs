"""
make_averages.py -- annual ensemble-mean series + period means, one name per variable

Reads the file catalog from build_catalog.py (which files exist, which are usable)
and writes, under <out_dir>:

  annual_ensmean/<scenario>_annual_ensmean.nc
      every variable, every year, averaged across ensemble members   (year, lat, lon)
  period_means/<scenario>_<period>.nc
      every variable averaged over the period's years                (lat, lon)
  reference_members/SSP245_present_members.nc      [only with --save-reference-members]
      per-member annual values for the present period         (member, year, lat, lon)
  variable_index.csv   one row per variable name; add a "keep" column and filter on it
  file_qc.csv          year coverage + attribute checks for every input file
  member_usage.csv     which members went into each average

Every variable is stored under a unique name (e.g. CONCLD_mean, TSMX_high_frequency),
so choosing a subset for the Mahalanobis step is just:  ds[list_of_names]

Usage (run from a folder that is NOT inside the data folders):
    python make_averages.py /path/to/L1 /path/to/output
    python make_averages.py /path/to/L1 /path/to/output --scenarios SSP245 --metrics PRECT

Quick xarray vocabulary used below:
  Dataset   = a bundle of named variables sharing the same dimensions/coordinates
              (one netCDF file opens as one Dataset).
  DataArray = a single variable, e.g. ds["mean"], with its dims (year, lat, lon),
              coordinate values, and an .attrs dictionary of metadata.
  dim       = a named axis. ds.mean(dim="member") collapses that axis by averaging.
"""
import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from build_variable_catalog import SCENARIOS, build_file_catalog, explode_to_variables, make_var_id

# ---------------------------------------------------------------------------
# Settings. A dict maps a key to a value: PERIODS["present"]["years"] -> (2015, 2034).
# Present exists only for SSP245 (the ARISE runs start later), future for all three.
# To add a period, add another entry here; nothing else in the script needs to change.
# ---------------------------------------------------------------------------
PERIODS = {
    "present": {"years": (2015, 2034), "scenarios": ["SSP245"]},
    "future":  {"years": (2050, 2069), "scenarios": ["SSP245", "ARISE_SAI_1p0", "ARISE_SAI_1p5"]},
}
COMPRESS = True   # zlib compression in the output files (needs the netCDF4 backend).
                  # Set to False if writing fails with an "unexpected encoding" error.


# ---------------------------------------------------------------------------
# Step 0: quality-control scan. Opens each file lazily (metadata only) and checks
# that years, variable names and the scenario/member/variable attributes match
# what the folder structure says.
#
# Why: the catalog is built from folder and file NAMES. This confirms the
# CONTENTS agree (e.g. a file in the SSP245 folder really says scenario = SSP245),
# and shows which years each file covers before any averaging happens.
# ---------------------------------------------------------------------------
def scan_files(files):
    rows = []
    # itertuples() yields one row at a time; columns become attributes (r.path, r.member...)
    for r in files.itertuples():
        # "with" closes the file automatically when the block ends
        with xr.open_dataset(r.path) as ds:
            years = ds["year"].values          # .values = plain numpy array of the coordinate
            rows.append({
                "scenario": r.scenario, "metric": r.metric, "extreme": r.extreme,
                "member": r.member, "path": r.path,
                "first_year": int(years.min()), "last_year": int(years.max()),
                "n_years": len(years),
                # expected variable names (from the catalog) that are absent from the file
                "missing_vars": [v for v in r.variables if v not in ds.data_vars],
                # .attrs is a dict of file metadata. f"{6:03d}" formats 6 as "006"
                # (3 digits, zero-padded), the way members are stored in the attributes.
                "attr_scenario_ok": ds.attrs.get("scenario") == r.scenario,
                "attr_member_ok": ds.attrs.get("member") == f"{r.member:03d}",
                "attr_variable_ok": ds.attrs.get("variable") == r.metric,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 1: load every member for one (scenario, metric, extreme) and stack them
# along a new "member" dimension. join="exact" makes xarray raise an error if the
# members don't share identical years/lat/lon, instead of silently padding with NaN.
# That error is wanted: it means two members are not on the same footing.
# ---------------------------------------------------------------------------
def load_members(group):
    # open_dataset is lazy: it reads only metadata until the values are needed
    datasets = [xr.open_dataset(p) for p in group["path"]]
    try:
        # concat creates the new "member" axis. .load() then reads the values into
        # memory, which must happen BEFORE the files are closed below.
        stacked = xr.concat(datasets, dim="member", join="exact").load()
    finally:
        # "finally" runs even if concat raised an error, so file handles never leak
        for ds in datasets:
            ds.close()
    # label the new axis with member IDs ("006", "007", ...) so you can select by name later
    return stacked.assign_coords(member=[f"{m:03d}" for m in group["member"]])


def period_mean(annual, start, end, label):
    """Average the annual maps over start..end. Warns (does not fail) if years are missing."""
    # .sel with a slice selects by coordinate VALUE (years), not position, and includes
    # both endpoints, so slice(2015, 2034) is 20 years.
    sub = annual.sel(year=slice(start, end))
    n_expected = end - start + 1
    n_found = sub.sizes["year"]
    if n_found != n_expected:
        # e.g. CONCLD files starting in 2016 would show up here as 19 of 20 years
        print(f"  WARNING {label}: only {n_found} of {n_expected} years in {start}-{end}")
    # keep_attrs=True stops xarray from discarding the variable metadata during the mean
    out = sub.mean(dim="year", keep_attrs=True)
    # record how many years were actually averaged, so it is never a mystery later
    for name in out.data_vars:
        out[name].attrs["years_used"] = f"{int(sub['year'].min())}-{int(sub['year'].max())}"
        out[name].attrs["n_years_used"] = n_found
    return out


def save_nc(ds, path):
    path.parent.mkdir(parents=True, exist_ok=True)     # create folders if they don't exist
    # encoding is a dict of per-variable storage settings: variable name -> settings dict
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars} if COMPRESS else None
    ds.to_netcdf(path, encoding=encoding)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(root, out_dir, scenarios, metrics, members, save_ref_members, scan_only):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Catalog: which files should exist, which do, which are usable ---------
    catalog = build_file_catalog(root)
    missing = catalog[~catalog["exists"]]               # ~ means "not" for True/False columns
    print(f"{len(catalog)} expected files; {len(missing)} missing; "
          f"{(~catalog['usable']).sum()} flagged unusable")
    if len(missing):
        print(missing[["scenario", "metric", "extreme", "member"]].to_string(index=False))

    # Keep only files that exist and are usable, then apply any command-line filters.
    # Each line narrows the table with a True/False mask.
    files = catalog[catalog["exists"] & catalog["usable"]]
    files = files[files["scenario"].isin(scenarios)]
    if metrics:
        files = files[files["metric"].isin(metrics)]
    if members:
        files = files[files["member"].isin([int(m) for m in members])]
    print(f"Using {len(files)} files")

    # --- Variable index: written once, same for every scenario ---------------
    # Built from the full catalog (not the filtered files) so the index always lists
    # all variables, even in a small test run.
    index = (explode_to_variables(catalog)
             .drop_duplicates("var_id")[["var_id", "metric", "extreme", "variable", "family"]]
             .reset_index(drop=True))
    index["keep"] = ""          # fill in later, e.g. 1 to keep
    index.to_csv(out_dir / "variable_index.csv", index=False)

    # --- QC scan: read the problems before trusting the averages -------------
    qc = scan_files(files)
    qc.to_csv(out_dir / "file_qc.csv", index=False)
    print("\nYear coverage (rows = files sharing the same first/last year):")
    # groupby + size() counts files per combination. If one metric starts a year later
    # than the others, it shows up here as a separate row.
    print(qc.groupby(["scenario", "metric", "extreme", "first_year", "last_year"]).size().to_string())
    # | means "or" between True/False columns; ~ flips the attr checks so True = problem
    bad = qc[(qc["missing_vars"].str.len() > 0) | ~qc["attr_scenario_ok"]
             | ~qc["attr_member_ok"] | ~qc["attr_variable_ok"]]
    if len(bad):
        print(f"\nQC PROBLEMS in {len(bad)} files (see file_qc.csv):")
        print(bad[["path", "missing_vars", "attr_scenario_ok",
                   "attr_member_ok", "attr_variable_ok"]].to_string(index=False))
    if scan_only:
        return

    # --- Main loop: scenario -> metric/extreme group -> members ----------------
    usage_rows = []
    for scenario in scenarios:
        print(f"\n=== {scenario} ===")
        # One small Dataset per metric/extreme is collected here, then merged into a single
        # Dataset per scenario at the end. Building the pieces first and merging once is
        # simpler (and faster) than growing one big Dataset inside the loop.
        annual_parts = []
        period_parts = {p: [] for p in PERIODS}         # dict comprehension: {"present": [], "future": []}
        ref_parts = []

        scen_files = files[files["scenario"] == scenario]
        # groupby with two columns yields ((metric, extreme), rows) pairs; the "for (a, b), c"
        # form unpacks the key tuple into two variables. Each group is one folder of members.
        for (metric, extreme), group in scen_files.groupby(["metric", "extreme"]):
            member_ids = [f"{m:03d}" for m in group["member"]]
            label = f"{scenario} {metric} {extreme}"
            print(f"{label}: {len(member_ids)} members")

            stacked = load_members(group)                    # dims: (member, year, lat, lon)

            # Rename e.g. 'frequency' -> 'TSMX_high_frequency' so names are unique across files.
            # Keep the mapping so we can record each variable's original name as an attribute.
            new_names = {v: make_var_id(metric, extreme, v) for v in stacked.data_vars}
            stacked = stacked.rename(new_names)

            # Ensemble mean for every year: the "member" axis disappears, leaving
            # (year, lat, lon). float32 halves the file size; precision is far beyond
            # what the climate indices need.
            annual = stacked.mean(dim="member", keep_attrs=True).astype("float32")

            # Attach provenance to each variable. These travel with the data inside the
            # netCDF file, so a variable can always be traced to its source.
            for original, name in new_names.items():
                annual[name].attrs.update({
                    "metric": metric, "extreme": extreme, "variable": original,
                    "family": group["family"].iloc[0], "scenario": scenario,
                    "members": ",".join(member_ids), "n_members": len(member_ids),
                    # baseline period used to define extreme thresholds (extreme files only)
                    "clim_reference": stacked.attrs.get("clim_reference", "NA"),
                })
            annual_parts.append(annual)

            # Period means come from the annual means just computed, not from re-reading files.
            for period, spec in PERIODS.items():
                if scenario in spec["scenarios"]:
                    start, end = spec["years"]
                    period_parts[period].append(period_mean(annual, start, end, label))

            # Optional: keep the individual members for the present period. This is what you
            # would use for interannual variability (ICV), since averaging members first
            # smooths out year-to-year variability.
            if save_ref_members and scenario == "SSP245":
                start, end = PERIODS["present"]["years"]
                ref_parts.append(stacked.sel(year=slice(start, end)).astype("float32"))

            usage_rows.append({
                "scenario": scenario, "metric": metric, "extreme": extreme,
                "n_members": len(member_ids), "members": ",".join(member_ids),
                "first_year": int(annual["year"].min()), "last_year": int(annual["year"].max()),
            })

        if not annual_parts:
            continue                                        # nothing found for this scenario

        # --- Write outputs for this scenario ---------------------------------
        # Combine all metrics into one Dataset per scenario. Names are unique, so no clashes.
        annual_all = xr.merge(annual_parts)
        annual_all.attrs = {"scenario": scenario,
                            "description": "Mean across ensemble members; annual values retained"}
        save_nc(annual_all, out_dir / "annual_ensmean" / f"{scenario}_annual_ensmean.nc")

        for period, parts in period_parts.items():
            if parts:
                merged = xr.merge(parts)
                start, end = PERIODS[period]["years"]
                merged.attrs = {"scenario": scenario, "period": f"{start}-{end}",
                                "description": "Mean across years, then across ensemble members"}
                save_nc(merged, out_dir / "period_means" / f"{scenario}_{period}.nc")

        if ref_parts:
            # Outer join on "member": a member flagged for one metric only shows up as NaN there
            ref = xr.merge(ref_parts)
            ref.attrs = {"scenario": scenario, "period": "present",
                         "description": "Per-member annual values (for interannual variability)"}
            save_nc(ref, out_dir / "reference_members" / f"{scenario}_present_members.nc")

    # Record which members went into each average, for reproducibility
    pd.DataFrame(usage_rows).to_csv(out_dir / "member_usage.csv", index=False)
    print("\nDone.")


# This block runs only when the file is executed as a script (python make_averages.py ...),
# not when its functions are imported into a notebook. argparse turns the command-line words
# into the arguments that run() needs.
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder containing the scenario folders (e.g. .../L1)")
    ap.add_argument("out_dir", help="where to write the outputs")
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--metrics", nargs="+", default=None, help="e.g. PRECT TSMX (default: all)")
    ap.add_argument("--members", nargs="+", default=None, help="e.g. 006 007 (default: all usable)")
    ap.add_argument("--save-reference-members", action="store_true",
                    help="also save per-member annual values for the SSP245 present period")
    ap.add_argument("--scan-only", action="store_true", help="run the QC scan and stop")
    a = ap.parse_args()
    run(a.root, a.out_dir, a.scenarios, a.metrics, a.members,
        a.save_reference_members, a.scan_only)