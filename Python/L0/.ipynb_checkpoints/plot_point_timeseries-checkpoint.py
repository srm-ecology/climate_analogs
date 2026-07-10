"""
Plot per-scenario time series at a single lat/lon point, for one or more of the ARISE-SAI vs
SSP2-4.5 indicator variable groups (temp/TSMX, o3/O3, concld/CONCLD, precip/PRECT).

Example (East Lansing, MI):
    python3 plot_point_timeseries.py --lat 42.7 --lon -84.5 --name "East Lansing"
"""
# ---------------------------------------------------------------------------
# Import packages and initialisation 
# ---------------------------------------------------------------------------
import argparse  # CLI argument parsing (--lat, --lon, --name, --groups, --out-dir)
import glob  # filesystem globbing to find ensemble-member netCDF files
import os  # path joining/creation for locating data and writing output
import re  # regex to pull the 3-digit member id out of each filename

import matplotlib.pyplot as plt  # figure/axes creation and saving the PNG output
import numpy as np  # array alignment, NaN-aware reductions (mean/min/max) across members
import xarray as xr  # netCDF loading and label-based (lat/lon/year) data selection

DATA_ROOT = "/mnt/research/nasabio/data/climate/L1"
OUT_DIR = "/mnt/home/f0113797/Documents/climate_analogs/Python/L0/plots"

SAI_START = 2035

SCENARIOS = ["SSP245", "ARISE_SAI_1p5", "ARISE_SAI_1p0"]
SCENARIO_LABELS = {
    "SSP245": "SSP2-4.5 (no SAI)",
    "ARISE_SAI_1p5": "ARISE-SAI-1.5",
    "ARISE_SAI_1p0": "ARISE-SAI-1.0",
}
SCENARIO_COLORS = {
    "SSP245": "tab:red",
    "ARISE_SAI_1p5": "tab:blue",
    "ARISE_SAI_1p0": "tab:green",
}

# ---------------------------------------------------------------------------
# Per-variable-group configuration (row labels / panel titles
# ---------------------------------------------------------------------------
GROUPS = {
    "temp": dict(
        rel_path="TSMX/extreme_high",
        glob="extreme_indices_*.nc",
        variables=["frequency", "duration", "mean_intensity", "max_intensity"],
        row_labels=["Frequency", "Duration", "Mean intensity", "Peak intensity"],
        panel_titles=None,
    ),
    "o3": dict(
        rel_path="O3",
        glob="bioclim_indicators_*.nc",
        variables=["mean", "max_month", "min_month", "range", "seasonality"],
        row_labels=["Annual mean", "Max month", "Min month", "Annual range", "Seasonality"],
        panel_titles=None,
    ),
    "concld": dict(
        rel_path="CONCLD",
        glob="bioclim_indicators_*.nc",
        variables=["mean", "max_month", "min_month", "range", "seasonality"],
        row_labels=["Annual mean", "Max month", "Min month", "Annual range", "Seasonality"],
        panel_titles=None,
    ),
    "precip": dict(
        rel_path="PRECT",
        glob="precip_indices_*.nc",
        variables=["CDD", "CWD", "PRCPTOT", "R95pTOT"],
        row_labels=["CDD", "CWD", "PRCPTOT", "R95pTOT"],
        panel_titles=["Consecutive Dry Days", "Consecutive Wet Days",
                      "Annual Total Precip", "Heavy Precip (>P95)"],
    ),
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
MEMBER_IDS = {"006", "007", "008", "009", "010"}


def member_id(path):
    m = re.search(r"(\d{3})\.nc$", path)
    return m.group(1) if m else None


def member_files(scenario, group):
    pattern = os.path.join(DATA_ROOT, scenario, group["rel_path"], group["glob"])
    files = sorted(glob.glob(pattern))
    return [f for f in files if member_id(f) in MEMBER_IDS]


def open_member(path):
    return xr.open_dataset(path)


def var_units(ds, var):
    return ds[var].attrs.get("units", "")


def nearest_point_series(ds, var, lat0, lon0):
    """Nearest-gridcell time series for `var`, at (lat0, lon0). Longitude is converted to the
    dataset's 0-360 convention before selection."""
    lon0_360 = lon0 % 360
    return ds[var].sel(lat=lat0, lon=lon0_360, method="nearest")


def resolve_grid_point(group, lat0, lon0):
    """Actual (lat, lon) of the nearest grid cell"""
    files = member_files("SSP245", group)
    ds = open_member(files[0])
    point = nearest_point_series(ds, group["variables"][0], lat0, lon0)
    resolved = (float(point.lat), float(point.lon))
    ds.close()
    return resolved


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def compute_point_time_series(group, lat0, lon0):
    """Per-scenario, per-variable: years, ensemble-mean point series, min, max across members.
    """
    series = {v: {} for v in group["variables"]}
    for s in SCENARIOS:
        files = member_files(s, group)
        per_member = {v: [] for v in group["variables"]}
        for f in files:
            ds = open_member(f)
            years = ds.year.values[:-1]
            for v in group["variables"]:
                pt = nearest_point_series(ds, v, lat0, lon0).values[:-1]
                per_member[v].append((years, pt))
            ds.close()
        n_members = len(per_member[group["variables"][0]])
        for v in group["variables"]:
            all_years = sorted(set().union(*[set(y) for y, _ in per_member[v]]))
            aligned = []
            for years, pt in per_member[v]:
                lookup = dict(zip(years, pt))
                aligned.append([lookup.get(y, np.nan) for y in all_years])
            arr = np.array(aligned, dtype=float)
            coverage = np.sum(np.isfinite(arr), axis=0)
            keep = coverage == n_members
            all_years = np.array(all_years)[keep]
            arr = arr[:, keep]
            series[v][s] = dict(
                years=all_years,
                mean=np.nanmean(arr, axis=0),
                lo=np.nanmin(arr, axis=0),
                hi=np.nanmax(arr, axis=0),
            )
    return series


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_point_fig(prefix, group, lat0, lon0, location_name, out_dir):
    grid_lat, grid_lon = resolve_grid_point(group, lat0, lon0)
    series = compute_point_time_series(group, lat0, lon0)

    all_nan = all(
        not np.isfinite(series[v][s]["mean"]).any()
        for v in group["variables"] for s in SCENARIOS
    )
    if all_nan:
        print(f"[{prefix}] no valid data at ({lat0}, {lon0}) -> nearest grid cell "
              f"({grid_lat:.2f}, {grid_lon:.2f}) is masked (e.g. ocean point for a land-only "
              f"variable, or vice versa). Skipping.")
        return

    nvar = len(group["variables"])
    nrows, ncols = (2, 2) if nvar <= 4 else (2, 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    titles = group["panel_titles"] or group["row_labels"]
    for i, v in enumerate(group["variables"]):
        ax = axes[i]
        for s in SCENARIOS:
            d = series[v][s]
            ax.plot(d["years"], d["mean"], color=SCENARIO_COLORS[s], label=SCENARIO_LABELS[s], linewidth=1.6)
            ax.fill_between(d["years"], d["lo"], d["hi"], color=SCENARIO_COLORS[s], alpha=0.2, linewidth=0)
        ax.axvline(SAI_START, color="gray", linestyle="--", linewidth=1.2, label=f"SAI start ({SAI_START})")
        ax.set_title(titles[i], fontsize=13, fontweight="bold")
        ax.set_xlabel("Year", fontsize=11, fontweight="bold")
        files = member_files("SSP245", group)
        ds0 = open_member(files[0])
        unit = var_units(ds0, v)
        ds0.close()
        ax.set_ylabel(unit, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")

    for j in range(nvar, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"{location_name} ({lat0:.2f}, {lon0:.2f})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    slug = location_name.lower().replace(" ", "_").replace(",", "")
    out_path = os.path.join(out_dir, f"point_{slug}_{prefix}_temporal_evolution.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{prefix}] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, required=True, help="Latitude, e.g. 42.7")
    parser.add_argument("--lon", type=float, required=True,
                         help="Longitude in -180..180, e.g. -84.5 (converted to 0-360 internally)")
    parser.add_argument("--name", type=str, default=None, help="Location name for titles/filenames")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS.keys()), choices=list(GROUPS.keys()),
                         help="Variable groups to plot (default: all)")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    location_name = args.name or f"({args.lat:.2f}, {args.lon:.2f})"
    os.makedirs(args.out_dir, exist_ok=True)
    for prefix in args.groups:
        make_point_fig(prefix, GROUPS[prefix], args.lat, args.lon, location_name, args.out_dir)

if __name__ == "__main__":
    main()
