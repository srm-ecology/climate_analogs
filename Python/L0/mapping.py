"""Plot helpers for sigma dissimilarity maps (matplotlib only; no cartopy needed)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap


def to_grid(values, index):
    """
    Turn a flat array (one value per cell) plus its (lat, lon) MultiIndex into a 2-D
    array. Returns (lat, lon, grid) with grid[i, j] = value at lat[i], lon[j].
    Cells absent from the index (ocean) become NaN.
    """
    s = pd.Series(np.asarray(values, float), index=index)
    grid = s.unstack("lon").sort_index().sort_index(axis=1)   # rows = lat, cols = lon
    return grid.index.values, grid.columns.values, grid.values


# --------------------------------------------------------------------------
# Background outlines
# --------------------------------------------------------------------------
def _draw_geom(ax, geom, **kw):
    """Draw any shapely geometry (polygon, line, or multi-part) as plain lines on ax."""
    if hasattr(geom, "geoms"):                       # Multi* / GeometryCollection
        for g in geom.geoms:
            _draw_geom(ax, g, **kw)
    elif hasattr(geom, "exterior"):                  # Polygon: outer ring + holes
        ax.plot(*geom.exterior.xy, **kw)
        for ring in geom.interiors:
            ax.plot(*ring.xy, **kw)
    else:                                            # LineString
        ax.plot(*geom.xy, **kw)


def add_outline(ax, outline="auto", lon=None, lat=None, grid=None, resolution="50m",
                states=True, color="black", lw=0.8, halo=True):
    """
    Draw a background outline on a plain lon/lat matplotlib axis (-180..180 longitudes).

    outline : "auto"     -> try Natural Earth via cartopy, fall back to "mask"
              "cartopy"  -> Natural Earth country outlines (+ state/province lines)
              "mask"     -> outline of the land cells themselves (no extra libraries)
              path       -> any vector file geopandas can read (.shp, .geojson, ...)
              None       -> nothing
    """
    if outline is None:
        return
    # white stroke behind the line keeps it visible over dark AND light cells
    fx = [pe.withStroke(linewidth=lw + 1.6, foreground="white", alpha=0.8)] if halo else []
    if outline in ("auto", "cartopy"):
        try:
            from cartopy.io import shapereader
            layers = [("admin_0_countries", 1.0)]
            if states:
                layers.append(("admin_1_states_provinces_lines", 0.6))
            for name, scale in layers:
                path = shapereader.natural_earth(resolution=resolution, category="cultural", name=name)
                for geom in shapereader.Reader(path).geometries():
                    _draw_geom(ax, geom, color=color, lw=lw * scale, zorder=3, path_effects=fx)
            return
        except Exception as err:                     # ImportError, or no internet to download
            if outline == "cartopy":
                raise
            print(f"cartopy outline unavailable ({type(err).__name__}); using land-mask outline")
            outline = "mask"
    if outline == "mask":
        cs = ax.contour(lon, lat, np.isfinite(grid).astype(float), levels=[0.5],
                        colors=color, linewidths=lw, zorder=3)
        cs.set_path_effects(fx)
        return
    import geopandas as gpd                          # outline is a file path
    gpd.read_file(outline).boundary.plot(ax=ax, color=color, linewidth=lw, zorder=3,
                                         path_effects=fx)


def plot_sigma_map(values, index, title="", vmax=4.0, focal=None, extent=None,
                   lon180=True, outline="auto", alpha=0.8, cmap="mahony", label="Dissimilarity of best analog (σ)",
                   ax=None):
    """
    values : (n,) array, one value per cell
    index  : (lat, lon) MultiIndex in the same order as values (e.g. A.index)
    vmax   : colour scale tops out here; larger values get the 'over' colour
    focal  : optional (lat, lon) to mark with a star (lon in the dataset's convention)
    extent : optional (lon_min, lon_max, lat_min, lat_max) to zoom, in plotted coordinates
    lon180 : convert 0-360 longitudes to -180..180 for plotting
    alpha  : transparency of the coloured cells (1 = opaque); lower lets the outline show through
    outline: background outline; see add_outline ("auto", "cartopy", "mask", a file path, None)
    cmap   : "mahony" = light gray -> black as in the paper, or any matplotlib name
             (e.g. "viridis_r")
    """
    # +inf sigma means "too far for the chi conversion" (distance > ~38), i.e. off the
    # scale. pcolormesh treats inf as missing and leaves it blank, so swap it for a large
    # finite number, which gets the 'over' colour. NaN stays NaN (truly missing).
    values = np.where(np.isposinf(values), 1e6, values)
    lat, lon, grid = to_grid(values, index)
    if lon180:
        lon = ((lon + 180) % 360) - 180
        order = np.argsort(lon)
        lon, grid = lon[order], grid[:, order]
        if focal is not None:
            focal = (focal[0], ((focal[1] + 180) % 360) - 180)

    if cmap == "mahony":
        cmap = LinearSegmentedColormap.from_list("mahony", ["lightgray", "black"])
    else:
        cmap = plt.get_cmap(cmap).copy()
    cmap.set_over(cmap(1.0))        # values > vmax drawn in the top colour

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    mesh = ax.pcolormesh(lon, lat, grid, cmap=cmap, vmin=0, vmax=vmax, shading="nearest",
                         alpha=alpha)
    plt.colorbar(mesh, ax=ax, extend="max", label=label, shrink=0.8)
    add_outline(ax, outline, lon=lon, lat=lat, grid=grid)
    if focal is not None:
        ax.plot(focal[1], focal[0], marker="*", ms=14, mfc="red", mec="white", ls="none")
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    return ax