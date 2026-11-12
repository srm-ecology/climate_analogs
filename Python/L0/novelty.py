"""
Sigma dissimilarity / Mahalanobis nearest-neighbour distance (Mahony et al. 2017),
Python port of Novelty_NA_SampleCode.R.

Layout
------
1. I/O helpers   : NetCDF -> tidy pandas objects (xarray is imported lazily so the
                   maths below can be imported and tested without it)
2. Maths         : detrend_keep_mean(), sigma_dissimilarity()

Notation (same as the paper / R script)
    A : (n_cells, K)  reference-period normals for every candidate analog cell
    B : (m_cells, K)  future-period normals for the focal cell(s)
    C : (T_years, K)  annual values at the focal cell (reference period), detrended
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.spatial import cKDTree
from scipy.stats import chi


# --------------------------------------------------------------------------
# 1. I/O helpers
# --------------------------------------------------------------------------
def _pick_data_var(ds):
    """Return the one real data variable in a Dataset, ignoring *_bnds etc."""
    cands = [v for v in ds.data_vars if "bnds" not in v and ds[v].ndim >= 2]
    if len(cands) != 1:
        raise ValueError(f"Expected exactly one data variable, found {cands}")
    return ds[cands[0]]


def load_normals(base_path, scenario, period, suffixes, transforms=None):
    """
    Build an (n_cells x K) DataFrame of period-mean climate, one column per variable.

    suffixes   : dict  {column name: file-name suffix}
                 e.g. {"PRECT": "PRECT", "TSMX": "TSMX_extreme_high"}
    transforms : dict  {column name: function}, e.g. {"PRECT": np.log}
    """
    import xarray as xr

    transforms = transforms or {}
    cols = {}
    for var, suffix in suffixes.items():                      # dict -> (key, value) pairs
        path = Path(base_path, f"{scenario}_{period}_{suffix}.nc")
        da = _pick_data_var(xr.open_dataset(path)).squeeze(drop=True)
        da = da.transpose("lat", "lon")                       # same cell order in every file
        s = da.to_series()                                    # MultiIndex (lat, lon)
        cols[var] = transforms.get(var, lambda x: x)(s)       # transform if one is registered
    return pd.DataFrame(cols).dropna()                        # dropna removes ocean / masked cells


def load_annual_at_point(base_path, scenario, period, suffixes, lat, lon360, transforms=None):
    """
    Build the (T_years x K) matrix C for the grid cell nearest (lat, lon360).
    Returns (C DataFrame indexed by year/time, actual_lat, actual_lon).
    """
    import xarray as xr

    transforms = transforms or {}
    cols, actual = {}, None
    for var, suffix in suffixes.items():
        path = Path(base_path, f"{scenario}_{period}_{suffix}.nc")
        da = _pick_data_var(xr.open_dataset(path))
        pt = da.sel(lat=lat, lon=lon360, method="nearest")
        actual = (float(pt.lat), float(pt.lon))
        s = pt.reset_coords(drop=True).to_series()            # index = time
        cols[var] = transforms.get(var, lambda x: x)(s)
    return pd.DataFrame(cols), actual


def load_annual_all(base_path, scenario, period, suffixes, cell_index, transforms=None):
    """
    Annual series for EVERY cell in cell_index (a (lat, lon) MultiIndex, e.g. A.index).
    Returns array (T_years, n_cells, K), in the cell order of cell_index.
    """
    import xarray as xr

    transforms = transforms or {}
    layers = []
    for var, suffix in suffixes.items():
        path = Path(base_path, f"{scenario}_{period}_{suffix}.nc")
        da = _pick_data_var(xr.open_dataset(path)).transpose("time", "lat", "lon")
        df = da.stack(cell=("lat", "lon")).to_pandas()        # rows = years, cols = (lat, lon)
        df = df.reindex(columns=cell_index)                   # same cells, same order as A
        layers.append(transforms.get(var, lambda x: x)(df.values))
    return np.stack(layers, axis=2)


# --------------------------------------------------------------------------
# 2. Maths
# --------------------------------------------------------------------------
def detrend_keep_mean(C):
    """
    Remove the linear trend along axis 0 (time) of C, but add the mean back so the
    series stays in its original units. Works on (T, K) or (T, n_cells, K) arrays.
    Rows with NaN must be dropped first (see detrend_cells for gridded data).
    """
    C = np.asarray(C, dtype=float)
    return detrend(C, axis=0, type="linear") + C.mean(axis=0)


def detrend_cells(C_all):
    """Detrend every cell of a (T, n_cells, K) array; cells with any NaN stay all-NaN."""
    ok = ~np.isnan(C_all).any(axis=(0, 2))
    out = np.full_like(C_all, np.nan, dtype=float)
    out[:, ok, :] = detrend_keep_mean(C_all[:, ok, :])
    return out


def distance_to_sigma(dist, n_pcs):
    """
    Convert a distance on a chi distribution with df = n_pcs into 'sigma', i.e. the same
    percentile on a chi distribution with df = 1 (a half-normal). sf/isf instead of
    cdf/ppf keeps precision in the far tail (beyond a distance of ~38 sf underflows
    to 0 and sigma is inf; irrelevant when maps cap at ~4 sigma).
    """
    return chi.isf(chi.sf(dist, df=n_pcs), df=1)


def _whiten(A, B, C, trunc_sd=0.1):
    """Steps 1-3a: scale by ICV, rotate onto ICV principal components, whiten."""
    A, B, C = (np.asarray(x, dtype=float) for x in (A, B, C))
    if np.isnan(C).any():
        raise ValueError("C contains NaN; drop incomplete years first")
    T = C.shape[0]

    # Step 1: express everything in units of local ICV (std dev per variable; ddof=1 like R's sd)
    sd = C.std(axis=0, ddof=1)
    A1, B1, C1 = A / sd, B / sd, C / sd

    # Step 2: PCA of the ICV. SVD of the centred C gives the same rotation as R's prcomp().
    center = C1.mean(axis=0)
    _, S, Vt = np.linalg.svd(C1 - center, full_matrices=False)
    pc_sd = S / np.sqrt(T - 1)                       # = prcomp()$sdev
    n_pcs = int(np.sum(pc_sd > trunc_sd))            # truncation rule
    if n_pcs == 0:
        raise ValueError("No PCs exceed the truncation threshold")
    rot = Vt[:n_pcs].T                               # (K, n_pcs)
    X = (A1 - center) @ rot                          # analog pool in PC space
    Y = (B1 - center) @ rot                          # future conditions in PC space

    # Step 3a: express PC scores in units of ICV along each PC
    X /= pc_sd[:n_pcs]
    Y /= pc_sd[:n_pcs]
    return X, Y, n_pcs, pc_sd


def sigma_dissimilarity(A, B, C, trunc_sd=0.1, workers=-1):
    """
    Nearest-neighbour Mahalanobis distance and sigma dissimilarity of each row of B
    to its best analog among the rows of A, using the ICV in C as the yardstick.

    Returns a dict:
        mahalanobis  (m,)  NN distance in whitened, truncated PC space
        sigma        (m,)  same distance re-expressed as a 1-D half-normal 'sigma'
        analog_idx   (m,)  row position in A of each best analog
        n_pcs        int   number of PCs retained
        pc_sd        (K,)  std dev of every PC of the ICV (before truncation)
    """
    X, Y, n_pcs, pc_sd = _whiten(A, B, C, trunc_sd)
    # Step 3b: Euclidean NN in whitened space == Mahalanobis NN distance
    dist, idx = cKDTree(X).query(Y, k=1, workers=workers)
    return {"mahalanobis": dist, "sigma": distance_to_sigma(dist, n_pcs),
            "analog_idx": idx, "n_pcs": n_pcs, "pc_sd": pc_sd}


def analog_surface(A, b, C, trunc_sd=0.1):
    """
    ANALOG MAP: dissimilarity between ONE future climate b (K,) and the reference
    climate of EVERY cell in A. Low values = places whose present climate resembles b.
    Returns dict with mahalanobis (n,), sigma (n,), n_pcs.
    """
    X, Y, n_pcs, _ = _whiten(A, np.atleast_2d(b), C, trunc_sd)
    dist = np.linalg.norm(X - Y[0], axis=1)
    return {"mahalanobis": dist, "sigma": distance_to_sigma(dist, n_pcs), "n_pcs": n_pcs}


def novelty_grid(A, B, C_all, trunc_sd=0.1, step=1, progress_every=2000):
    """
    NOVELTY MAP (the R script's main loop): every cell is its own focal location.

    A, B   : (n, K) reference / future normals, same cell order
    C_all  : (T, n, K) detrended annual series; ICV for cell i is C_all[:, i, :]
    step   : only compute every `step`-th cell (e.g. 5) for a fast preview
    Returns dict of (n,) arrays: sigma, mahalanobis, analog_idx (NaN / -1 if skipped).
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    n, K = A.shape
    sigma = np.full(n, np.nan)
    mahal = np.full(n, np.nan)
    analog = np.full(n, -1)
    for count, i in enumerate(range(0, n, step)):
        Ci = C_all[:, i, :]
        Ci = Ci[~np.isnan(Ci).any(axis=1)]                # drop incomplete years
        if Ci.shape[0] < K + 2:                           # too short for a PCA
            continue
        try:
            r = sigma_dissimilarity(A, B[i:i + 1], Ci, trunc_sd, workers=1)
        except ValueError:
            continue
        sigma[i], mahal[i], analog[i] = r["sigma"][0], r["mahalanobis"][0], r["analog_idx"][0]
        if progress_every and count % progress_every == 0:
            print(f"cell {i} / {n}")
    return {"sigma": sigma, "mahalanobis": mahal, "analog_idx": analog}
