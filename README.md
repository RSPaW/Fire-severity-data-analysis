# Fire Severity analysis tools
This repository contains code used to convert, extract and explore fire 
severity data. It was developed as part of a 10-day pilot staff exchange 
programme between DBCA Fire Science and the Australian SKA Regional Centre 
(AusSRC), which focused on a single complete region in the Perth Hills. 

*In no way is it a fully-functioning analysis suite. Currently we have some simple conversion scripts and some exploratory data analysis code written as a way ot examining where the direct future effort for a more robust and meaningful analysis.*


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
  - `severity.npy`
  - `timestamps.npy`

which encode the severity measurements with fire-type information (as 8-bit integers) and timestamps (as `numpy.datetime64` objects). These may then simply be loaded into memory via `numpy.load` or, indeed, a mem-map via `numpy.lib.format.open_memmap()`.

## Exploration and initial analysis
### Event-based data
Looking at the event-level data across the full field, we execcute the `severity_ecotype_metrics_from_events.py` script and provide it the path to the `combined_event_data.pkl` created in the above data conversion step. 

These data files have information about the fire severity, time since last fire, severity of last fire and ecosystem types. Temporal information is technically included by virtue of being ascribed to specific events and time between subsequent events, although this is not exploited here. There is no spatial information available in the data form constructed here.

Three plots are generated, along with summary statistics for each ecosystem type. The analysis here specifically focuses on: most-recent fire severity vs. severity of previous burn, and most-recent fire severity vs. time since last burn. We als generate the correlation matrix for each ecosystem type to inspect the relationship between the three above variables. *(In many cases there are many pairs of events and so this is statistically interesting, but in some there are very few instances and so the outputs are hard to interpret.)*

### Monthly raster data
Looking at the monthly raster data, we are faced with a much larger data challenge in that we now have T samples of a grid with shape (Y, X), and thus have `T*Y*X` measurements (for the example data provided, quantity is 8.6 billion). A reasonable fraction of these data are "flagged" as being outside a region of interest, and the occurance of fires in the remaining valid pixels is very sparse. We inspect some general field statistics and look at how the severity measurements (per pixel) vary with time. Finally, we attempt a relatively naive PCA approach to extract spatial components, and then look at the spectrum of those components. In it's current form, it is hard to determine what is useful versus obfuscated by the data sparsity.

Dealing with the masked data or null measurements (i.e., where measurements were taken, but no fire-data was available) is a challenge, but critical. We take a reasonably naive approach, but considered thought on how to incorporate the null measurements could well be very insightful.

Running the `spatiotemporal_spectra.py` script, it will search in the current wordking directory for the `severity.npy` and `timestamps.npy` files, and move on to perform some cursory field exploration. It then prepares the data for a PCA decomposition, and generates example spatial reconstructions, temporal spectra, and spatial wave-mode spectra over time. It will generate 10 plots which are named based on what is being displayed. The most interesing are:
  - `severity_count_burns_map.png` which shows how many independent burns happen on a per pixel bases across the entire time-spand of the data set.
  - `severity_repeated_burn_map.png` which shows where more than 3 fires have been recorded for the same pixels.
  - `severity_activity_over_time.png` which shows the sum of the pixels over time and can be used to identify particulaly active and/or high-severity periods. *(Temporal analysis of this split by ecosystem type could be insightful.)*
  - `pca_variance_explained.png` which shows how much of the variance in the data is explained by each subsequent principle component. *(A slowly rising curve here is nominally bad - it means we need many components to describe the data, which defeats the point of using PCA to decrease the dimensionality of the problem.)*
- `eofN_temporal_spectrum.png` which looks at the Nth spatial component and examines its temporal scales via Lomb-Scargle periodogram (allowing for unevenly sampled data).
- `radial_spectra_over_time.png` which examines how the spatial scales decay over time.