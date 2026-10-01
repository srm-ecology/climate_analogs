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
import glob

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



#Function pulling everything together:

#input_path should be the full path to the .nc files. AND the specific file names for the members. 
def create_avg_ncfile(members, start_year, end_year, input_path, unqiue_title, output_path, file_name):
    path_files = []
    time_sliced_files = []
    # Get the list of all files and directories

    list_of_files = []
    for each in members:
        if (each.endswith("p95_threshold.nc")):
            list_of_files.remove(each)
        else: 
            path = input_path + "/" + unqiue_title + each + ".nc"
            file = xr.open_dataset(path)
            path_files.append(file)
    
    #For every ensemble, collapse the "year" dimension by averaging all values across the years at every lat and lon. 
    for each in path_files[:]: 
        sliced = each.sel(year = slice(start_year, end_year))
        each_ensemble_avg = sliced.mean(dim='year')
        time_sliced_files.append(each_ensemble_avg)

    # As a check of the code, let's look at if the first ensemble came out correctly. 
    # sliced1 = path_files[0].sel(year = slice(start_year, end_year)) #Data over the specified years. 
    # print(sliced1['CDD'].values[:, 2, 0]) #Here is the first value of the 3rd row for each of the 20 years. 
    # print((sliced1['CDD'].values[:, 2, 0]).mean()) #And it's mean. 
    # print(time_sliced_files[0]['CDD'].values[2, 0]) #And compare that with the 3rd row, 1st column of the averaged data. 
    
    concat_files_avg=xr.concat(time_sliced_files, dim='ensemble')
    concat_files_avg = concat_files_avg.assign_coords({'ensemble' : members})
    concat_files_avg = concat_files_avg.mean(dim='ensemble')

    concat_files_avg.attrs["title_description"] = concat_files_avg.attrs["scenario"] + " " + concat_files_avg.attrs["variable"] + " " + str(start_year) + "-" + str(end_year)
    
    concat_files_avg.to_netcdf(output_path + "/" + file_name)
    
    return concat_files_avg


members = ['006', '007', '008', '009', '010']
members_no_008 = ['006', '007', '009', '010']

# create_avg_ncfile(members, 2050, 2069, "/mnt/research/nasabio/data/climate/L1/ARISE_SAI_1p5/PRECT", "precip_indices_", "/mnt/research/nasabio/data/climate/L1/future", "ARISE_SAI_1p5_PRECT.nc")


# Manual testing of the function 
start_year = 2050
end_year = 2069
input_path = "/mnt/research/nasabio/data/climate/L1/ARISE_SAI_1p5/PRECT"
unique_title = "precip_indices_"
output_path = "/mnt/research/nasabio/data/climate/L1/future"
file_name = "ARISE_SAI_1p5_PRECT.nc"


path_files = []
time_sliced_files = []
# Get the list of all files and directories

list_of_files = []
for each in members:
    if (each.endswith("p95_threshold.nc")):
        list_of_files.remove(each)
    else: 
        path = input_path + "/" + unique_title + each + ".nc"
        file = xr.open_dataset(path)
        path_files.append(file)

#For every ensemble, collapse the "year" dimension by averaging all values across the years at every lat and lon. 
for each in path_files[:]: 
    sliced = each.sel(year = slice(start_year, end_year))
    each_ensemble_avg = sliced.mean(dim='year')
    time_sliced_files.append(each_ensemble_avg)

# As a check of the code, let's look at if the first ensemble came out correctly. 
# sliced1 = path_files[0].sel(year = slice(start_year, end_year)) #Data over the specified years. 
# print(sliced1['CDD'].values[:, 2, 0]) #Here is the first value of the 3rd row for each of the 20 years. 
# print((sliced1['CDD'].values[:, 2, 0]).mean()) #And it's mean. 
# print(time_sliced_files[0]['CDD'].values[2, 0]) #And compare that with the 3rd row, 1st column of the averaged data. 

concat_files_avg=xr.concat(time_sliced_files, dim='ensemble')
concat_files_avg = concat_files_avg.assign_coords({'ensemble' : members})
concat_files_avg = concat_files_avg.mean(dim='ensemble')

first = path_files[0]
concat_files_avg.attrs = {
    "scenario": first.attrs["scenario"],
    "variable": "Mean annual " + first.attrs["variable"],
    "clim_reference": first.attrs["clim_reference"],
    "members": ",".join(str(m) for m in members),
    "title_description": f"{first.attrs['scenario']} {first.attrs['variable']} {start_year}-{end_year}",
}
concat_files_avg.to_netcdf(output_path + "/" + file_name)



################################### 
# Check attributes for all files 
from pathlib import Path
import pandas as pd
import xarray as xr

root = Path("/mnt/research/nasabio/data/climate/L1/ARISE_SAI_1p5")

# 1. Find every .nc file at any depth below root
nc_files = sorted(root.rglob("*.nc"))
print(f"Found {len(nc_files)} files")

# 2. Build a dictionary: {file path: {attribute name: value}}
attrs_by_file = {}
for f in nc_files:
    try:
        with xr.open_dataset(f) as ds:
            attrs_by_file[str(f.relative_to(root))] = dict(ds.attrs)
    except Exception as e:
        print(f"Could not open {f}: {e}")

# 3. Dictionary of dictionaries -> table (rows = files, columns = attributes)
values_df = pd.DataFrame.from_dict(attrs_by_file, orient="index")

# 4. Presence/absence table: 1 if the file has that attribute, 0 if not
presence_df = values_df.notna().astype(int)

presence_df.to_csv("attrs_presence.csv")
values_df.to_csv("attrs_values.csv")
