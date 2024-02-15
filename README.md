# pyGC_FID_processing
Python class for processing multiple GC-FIG files at once (outlier removal, baseline removal, alignment, peak picking)

# Example usage
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
Remove the baseline from each spectrum by subtracting result from a minim filter
```python
s.subtract_base_line_all(window_size_portion=1/50, plts=False)
```
Align all spectra to the first one, determined as the lag at maximum crosscorrelation
```python
s.align_spectra(plts=False, max_time_offset=None)
```
Sum up the spectra
```python
s.set_summed()
```
Search peaks in summed up spectrum
```python
s.set_peaks(prominence=1e-3)
```
Search kernels as bigaussians in summed spectrum
```python
s.set_kernels()
```
Integrate area under peaks by multiplying kernels with spectra
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
