
import xarray as xr
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import detrend
from scipy.spatial import cKDTree
from scipy.stats import chi

base_path = "/mnt/research/nasabio/data/climate/L1/analog_testing"

# Focal location 
lat, lon = 42.73, -84.48           # East Lansing, MI
lon360 = lon % 360                 # 275.52, matching the dataset's 0-360 convention

trunc_sd=0.1 # Remove PC of local ICV whose variance is below 0.01


# A = n x K matrix of n spatially gridded observations of K climate variables (mean over time; reference period)
# precip = xr.open_dataset(Path(base_path, "SSP245_present_PRECT.nc")).to_dataframe()
temp = xr.open_dataset(Path(base_path, "SSP245_present_TSMX_extreme_high.nc")).to_dataframe()
# A = pd.concat([precip, temp], axis=1)
A_temp = temp.dropna()
A = np.asarray(A_temp, dtype=float)
# cdd = ds["CDD"]
# cdd.plot()

# B = n x K matrix of n spatially gridded observations of K climate variables (mean over time; future period)
# precip_f = xr.open_dataset(Path(base_path, "SSP245_future_PRECT.nc")).to_dataframe()
temp_f = xr.open_dataset(Path(base_path, "ARISE_SAI_1p5_future_TSMX_extreme_high.nc")).to_dataframe()
# B = pd.concat([precip_f, temp_f], axis=1)
B_temp = temp_f.dropna()

B = np.asarray(B_temp, dtype=float)


# C = T x K matrix of T concurrent annual observations of K climate variables at focal location 
temp_a = xr.open_dataset(Path(base_path, "SSP245_present_TSMX_annual.nc"))
pixel = temp_a.sel(lat=lat, lon=lon360, method="nearest")
plat, plon = float(pixel.lat), float(pixel.lon)
C = pixel.reset_coords(drop=True).to_dataframe()
C = np.asarray(C, dtype=float)
print(C)
T = C.shape[0]

B_focal = B_temp.loc[[(plat, plon)]]                 # focal cell's future normals (1 x K)

# Step 1: Express climate data as standardized anomalies of reference period ICV and ICV proxy j
## Need to linearly detrend local time series data for ICV calculation. 
C_detr = detrend(C, axis=0, type="linear") + C.mean(axis=0)

sd = C.std(axis=0, ddof=1)
A1, B1, C1 = A / sd, B / sd, C / sd

# Step 2: Extract the principal components of the reference period ICV and project all data onto these PCs
center = C1.mean(axis=0)
_, S, Vt = np.linalg.svd(C1 - center, full_matrices=False)
pc_sd = S / np.sqrt(T - 1)                       # = prcomp()$sdev
n_pcs = int(np.sum(pc_sd > trunc_sd))            # truncation rule
if n_pcs == 0:
    raise ValueError("No PCs exceed the truncation threshold")
rot = Vt[:n_pcs].T                               # (K, n_pcs) rotation matrix
X = (A1 - center) @ rot                          # analog pool in PC space
Y = (B1 - center) @ rot                          # focal future in PC space

# Step 3a: Express PC scores as standardized anomalies of reference interannual variability 
X /= pc_sd[:n_pcs]
Y /= pc_sd[:n_pcs]

# Step 3b: Find the sigma dissimilarity of each projected condition with its best analog (Euclidean nearest neighbor) in the observed analog pool. 
dist, idx = cKDTree(X).query(Y, k=1, workers=-1)

# Convert distance (chi dist., df = n_pcs) to sigma (chi dist., df = 1).
# sf/isf (survival function) instead of cdf/ppf keeps precision in the far tail,
# where cdf rounds to exactly 1.0 and ppf would return inf. (Beyond a distance of
# roughly 38 even sf underflows to 0 and sigma is returned as inf; irrelevant in
# practice since maps cap at ~4 sigma.)
sigma = chi.isf(chi.sf(dist, df=n_pcs), df=1)
 



from novelty import analog_surface
from mapping import plot_sigma_map
import matplotlib.pyplot as plt

surf = analog_surface(A_temp.values, B_focal.values[0], C_detr)

plot_sigma_map(surf["sigma"], A_temp.index, title="ARISE SAI 1.5 TSMX Extreme High Analogs",
               focal=(plat, plon), cmap="viridis_r",
               label="Dissimilarity to East Lansing's future (σ)",
               extent=(-130, -60, 20, 60))     # lon_min, lon_max, lat_min, lat_max
plt.savefig("SAI_1p5_TSMX_Extreme_High_EastLansing.png", dpi=200, bbox_inches="tight")


sig = surf["sigma"]
sig = np.where(np.isposinf(sig), 1e6, sig)     # inf = off the scale, so it belongs in ">4σ"

bins   = [0, 1, 2, 3, 4, np.inf]
labels = ["0–1σ", "1–2σ", "2–3σ", "3–4σ", ">4σ"]
binned = pd.cut(sig, bins=bins, labels=labels, right=False)   # bins are [0,1), [1,2), ...

counts = pd.Series(binned).value_counts(sort=False)           # sort=False keeps bin order
table = pd.DataFrame({
    "cells":   counts,
    "percent": (100 * counts / len(sig)).round(2),
})
table.loc["Total"] = [counts.sum(), table["percent"].sum()]
print(table)
