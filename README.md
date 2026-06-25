# Fire Severity analysis tools
This repository contains code used to convert, extract and explore fire 
severity data. It was developed as part of a 10-day pilot staff exchange 
programme between DBCA Fire Science and the Australian SKA Regional Centre 
(AusSRC), which focused on a single complete region in the Perth Hills. 

*In no way is it a fully-functioning analysis suite. Currently we have some simple conversion scripts and some exploratory data analysis code written as a way ot examining where the direct future effort for a more robust and meaningful analysis.*

---


## Data conversion
### Event data files (RDS)
When starting with event-specific data files (perhaps multiple files per event), which do not contain spatial locality information (i.e., data are just measurments or derived quantities for ~some region~), we should use the `convert_events_to_dataframe_pickle.py` script. As input provide the directory where the corresponding RDS-format files are stored. 
```Bash
python convert_events_to_dataframe_pickle.py /path/to/rds/files
```
after which a file named `combined_event_data.pkl` will be created, and may then be directly read in an analysis script via the `pandas.read_pickel()` method.

### Monthly raster data files (TIF)
When starting from monthly rasterised data (currently assuming TIF format), where each image corresponds to a time-step, the pixel shape is consistent, and the layers represent different metrics, one can use the `convert_monthly_rasters_to_matrix.py` script. As input provide the directory where the corresponding TIF-format image files are stored.
```Bash
python convert_monthly_rasters_to_matrix.py /path/to/tif/image/files
```
Two files will be created:
  - `severity_rasters_i8.npy`
  - `severity_rasters_timestamps.npy`

which encode the severity measurements (as 8-bit integers) and timestamps (as `numpy.datetime64` objects). These may then simply be loaded into memory via `numpy.load` or, indeed, a mem-map via `numpy.lib.format.open_memmap()`. 