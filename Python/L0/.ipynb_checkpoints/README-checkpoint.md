# Climate Analogs Under SAI

### Code files breakdown:  
* `code_to_avg_files.ipynb` (1)
* `all_data_for_location.ipynb` (2)
* `graphs_compare_future.ipynb` (3)


### All work is in `/mnt/research/nasabio/data/climate/L1/present`. 
Lala’s initial cleaned ensemble data is in: 
* ARISE_SAI_1p0
* ARISE_SAI_1p5
* SSP245
Pre-analog data analysis (Sophia's work) is in folders: 
* future
* present
* avg_csv_output
* analog_calculations
* scenario_comparisons


## Data Folders Structure for Pre-Analog Data Analysis (Sophia's work)   
### `present`
Files are .nc files. Do not download and open. Used for later calculations.  
Only includes climate variables from SSP24.5. (Present data)   
Subfolder:   
* sliced_and_avged: This stores the outcome from the function `create_sliced_ncfile` in `code_to_avg_files.ipynb` (1). This function takes in ensemble files, slices them based from 2015 to 2034 (present period), then takes the average of the ensembles. (This includes multiple years.) This is .nc files. 
Files:   
* These files are the output of `create_avg_ncfile` in `code_to_avg_files.ipynb` (1). This function takes ensemble files and takes the average of each dataset over from 2015 to 2034 (present period), and then takes an average over all the ensembles. (There are no years because it’s an average.) This is .nc files. 

### `future`
Files are .nc files. Do not download and open. Used for later calculations.   
Subfolder:   
* sliced_and_avged: This stores the outcome from the function `create_sliced_ncfile` in `code_to_avg_files.ipynb` (1). This function takes in ensemble files, slices them based from 2050 to 2069 (future period), then takes the average of the ensembles. (This includes multiple years.) 
Files:   
* These files are the output of `create_avg_ncfile` in `code_to_avg_files.ipynb` (1). This function takes ensemble files and takes the average of each dataset over from 2050 to 2069 (future period), and then takes an average over all the ensembles. (There are no years because it’s an average.)


### `avg_csv_output`
add_mean_to_dataframe function will save to this folder. `add_mean_to_dataframe` is based on the results of `get_location_data_all_pixels` function in `all_data_for_location.ipynb` (2). 
Files are .csv files of the spatial and temporal average of all pixels at a location for multiple locations over different time frames. 

### `analog_calculations`
Point of the folder is to contain all necessary data for analog calculations. Yearly folders is for interannual variation calculations. (Likely only for present data.)
Subfolders:  
* present: 
    * yearly: stores .csv files. .csv files has years 2015-2034, and 3 scenarios. 
* future: 
    * yearly: stores .csv files. .csv files has years 2050-2069, and 3 scenarios. 

### `scenario_comparisons` 
This is used to store the stats (average and standard deviation) of comparing the present period with the 3 future scenarios. The .csv files include the average and standard deviation of all the pixels of a specific region, for present and 3 future periods.   
(The average data is the same as the average stored in `avg_csv_output`, if they are the same location and time.)
