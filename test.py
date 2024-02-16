# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from cSpectrum import Spectra

xcol = 'RT'
ycol = 'counts'

files = [
    "your", "files", "here"
]

# initiate Spectra object with a list of files 
s = Spectra(list_path_files=files)
# plotting the summed spectrum can be used at any point
s.plt_summed(plt_all=False)
# all keyword arguments from here on are optional, values chosen here are 
# the default values
# smooth the signals, remove outliers, bigger window_size gives smoother signals
s.remove_outliers_all(window_length=101, diff_threshold=.001, plts=True)
# remove the baseline from each spectrum by subtracting result from a minim filter
s.subtract_base_line_all(window_size_portion=1/50, plts=True)
# align all spectra to the first one, determined as the lag at maximum crosscorrelation
s.align_spectra(plts=False, max_time_offset=None)
# sum up the spectra
s.set_summed()
# search peaks in summed up spectrum
s.set_peaks(prominence=1e-3)
# search kernels as bigaussians in summed spectrum
s.set_kernels()
# integrate area under peaks by multiplying kernels with spectra
s.bin_spectrum()
# obtain the dataframe of binned spectra where columns represent the spectra and
# rows the picked times
df = s.get_dataframe()
# save binned spectra to disc as csv or excel
# df.to_csv('your/path/here.csv')
# or
# df.to_excel('your/path/here.xlsx')

# s.set_reconstruction_losses(plts=False, ylim=(0, 2e7))


