# pyGC_FID_processing
Python class for processing multiple GC-FID files at once (outlier removal, baseline removal, alignment, peak picking)

# Usage
It is important to execute all of the steps below in order. If the quality of measurement allows it, the remove_outliers and subtract_baseline steps can be omitted

Import the object
```python
from cSpectrum import Spectra
```
Setup
```python
xcol = 'RT'
ycol = 'counts'

files = [
    "your", "list", "of", "files", "here"
]
```

Initiate Spectra object with a list of files 
```python
s = Spectra(list_path_files=files)
```
All keyword arguments from here on are optional, values chosen here are the default values.

Plotting the summed spectrum can be used at any point
```python
s.plt_summed(plt_all=False)
```
Start the processing:
Smooth the signals, remove outliers, bigger window_size gives smoother signals
```python
s.remove_outliers_all(window_length=101, diff_threshold=.001, plts=False)
```

![image](https://github.com/yaza11/pyGC_FID_processing/assets/116643078/e486aeed-496a-40e4-b36c-1608a9c37a58)

Remove the baseline from each spectrum by subtracting result from a minimum filter
```python
s.subtract_base_line_all(window_size_portion=1/50, plts=False)
```
![image](https://github.com/yaza11/pyGC_FID_processing/assets/116643078/3ebf2e45-0251-4b18-9347-4f831ed9eed1)

Align all spectra to the first one, determined as the lag at maximum crosscorrelation
```python
s.align_spectra(plts=False, max_time_offset=None)
```
Cut of injection peak after alignment (injection peak helps with stability for alignment)
```python
s.set_rt_window_all(window=(5, np.infty))
```


Sum up the spectra, sets the "rts" and "counts" attribute
```python
s.set_summed()
```
Search peaks in summed up spectrum, defines the "peaks", "peak_properties" and "peak_setting_parameters" attributes
```python
s.set_peaks(prominence=1e-3)
```
Search kernels as bigaussians in summed spectrum, sets the attribute "kernel_params"
```python
s.set_kernels()
```
Integrate area under peaks by multiplying kernels with spectra, defines the "line_spectra" attribute
```python
s.bin_spectrum()
```
Obtain the dataframe of binned spectra where columns represent the spectra and rows the picked times
```python
df = s.get_dataframe()
```
Save binned spectra to disc as csv or excel
```python
df.to_csv('your/path/here.csv')
# or
df.to_excel('your/path/here.xlsx')
```
The quality of the binning procedure can be controlled for each measurement, defines the "losses" attribute
```python
s.set_reconstruction_losses(idxs=None, plts=True, ylim=(0, 2e7))
```
![image](https://github.com/yaza11/pyGC_FID_processing/assets/116643078/21f9de93-1008-4f3f-92af-afdb00edd946)

