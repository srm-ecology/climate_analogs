#This is the entire function of get_location_data_all_pixels from all_data_for_location.ipynb, used for graphs_compare_future.ipynb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import geopandas as gpd
import earthpy as et
import xarray as xr
# Spatial subsetting of netcdf files
import regionmask
import rioxarray
import pyproj
from shapely.ops import transform
from shapely.geometry import Point
import glob
from csv import writer
# Plotting options
sns.set(font_scale=1.3)
sns.set_style("white")
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning
)
import earthpy as et







def convert_longitude(ds):
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    ds = ds.sortby('lon')
    return ds

def get_location_data_all_pixels(input_path, projection, *, target_lat = None, target_lon = None, shapefile = None, buffer = None):
    climate_scenario = ["ARISE_SAI_1p0", "ARISE_SAI_1p5", "SSP245"]
    climate_variables = ["SSP", "TSMX", "PRECT"]
    
    info_list = {}

    list_of_files = glob.glob(input_path + '/*.nc') 
    file_reprojected = []
    
    for each in list_of_files:
        file = xr.open_dataset(each)
        file = convert_longitude(file)
        file = file.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        file.rio.write_crs("EPSG:4326", inplace=True) #Set our reference system (mercator projection)
        file = file.rio.reproject(projection)
        file_reprojected.append(file)
        
    
    if (target_lat != None and target_lon != None and shapefile == None):
        point = Point(target_lon, target_lat)
        gdf_point = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
        gdf_projected = gdf_point.to_crs(projection) #CONUS coordinate.
        
        
        masked_files_list = []        
        if (buffer != None):          #Buffered point
            buffered_geom = gdf_projected.geometry.buffer(buffer) #apply buffer.
            for each in file_reprojected:  
                masked_data = each.rio.clip(buffered_geom.geometry, drop = True, all_touched=True) #Set true if you want to crop it. 
                masked_files_list.append(masked_data)
                
            return masked_files_list
            
        
        else:           #No buffer point
            for each in file_reprojected:
                point_data = each.rio.clip(gdf_projected.geometry, drop = True, all_touched=True)
                #point_data = file_reprojected.sel(lat=target_lat, lon=target_lon, method='nearest')
                masked_files_list.append(point_data)
            return masked_files_list
                    
    elif (target_lat == None and target_lon == None):  #Shapefile        
        
        masked_files_list = []        
        for each in file_reprojected:            
            mask = shapefile.to_crs(each.rio.crs)
            sliced_masked = each.rio.clip(
                mask.geometry,
                mask.crs,
                drop=True,
                all_touched=True
            )
            
            masked_files_list.append(sliced_masked)
        return masked_files_list

    else: 
        return("Please enter either a point or a shapefile correctly.")
    #Checking for correct inputs. 
