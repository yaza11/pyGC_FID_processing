# -*- coding: utf-8 -*-
"""This module allows the usage of functions from the R rtms package in python."""
import os

from typing import Iterable 
import re
import warnings

import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import pickle
from scipy.signal import find_peaks, convolve, correlate, correlation_lags
from scipy.optimize import curve_fit
from scipy.ndimage import minimum_filter, median_filter


class BaseClassSpectrum:
    def plot(self, *args, limits=None, hold=False, **kwargs):
        if len(args) == 0:
            args = ['+-']
        if limits is None:
            mask = np.ones_like(self.rts, dtype=bool)
        else:
            assert limits[0] < limits[1], 'left bound has to be smaller, e.g. limits=(600, 2000)'
            mask = (self.rts >= limits[0]) & (self.rts <= limits[1])
        rts = self.rts[mask]
        counts = self.counts[mask]
        fig, ax = plt.subplots()
        plt.plot(rts, counts, *args, **kwargs)
        plt.xlabel('retention time in min')
        plt.ylabel('Intensities')
        
        if hold:
            return fig, ax
        plt.show()
        
    def to_pandas(self):
        """Return mass and intensity as pandas dataframe."""
        df = pd.DataFrame({'RT': self.rts, 'counts': self.counts})
        return df
    
    def bigaussian_from_peak(self, peak_idx: int):
        """Find kernel parameters for a peak with the shape of a bigaussian."""
        assert hasattr(self, 'peaks'), 'call set_peaks first' 
        mz_idx = self.peaks[peak_idx]  # mz index of of center 
        mz_c = self.rts[mz_idx] # center of gaussian
        H = self.counts[mz_idx]
        # width of peak at half maximum
        FWHM_l = self.rts[
            (self.peak_properties["left_ips"][peak_idx] + .5).astype(int)
        ]
        FWHM_r = self.rts[
            (self.peak_properties["right_ips"][peak_idx] + .5).astype(int)
        ]
        # convert FWHM to standard deviation
        sigma_l = -(FWHM_l - mz_c) / (2 * np.log(2))
        sigma_r = (FWHM_r - mz_c) / (2 * np.log(2))        
        return mz_c, H, sigma_l, sigma_r
    
    @staticmethod
    def bigaussian(x: np.ndarray, x_c, H, sigma_l, sigma_r):
        """
        Evaluate bigaussian for mass vector based on parameters.

        Parameters
        ----------
        x : np.ndarray
            mass vector.
        x_c : float
            mass at center of peak
        H : float
            Amplitude of peak.
        sigma_l : float
            left-side standard deviation.
        sigma_r : float
            right-side standard deviation.

        Returns
        -------
        np.ndarray
            Intensities of bigaussian.

        """
        x_l = x[x <= x_c]
        x_r = x[x > x_c]
        y_l = H * np.exp(-1/2 * ((x_l - x_c) / sigma_l) ** 2)
        y_r = H * np.exp(-1/2 * ((x_r - x_c) / sigma_r) ** 2)
        return np.hstack([y_l, y_r])
    
    def set_peaks(self, prominence: float = .005, width=3, **kwargs):
        """
        Find peaks in summed spectrum using scipy's find_peaks function.

        Parameters
        ----------
        prominence : float, optional
            Required prominence for peaks. The default is 0.005 (in terms of 
            maximum normalized spectrum.
        width : int, optional
            Minimum number of points between peaks. The default is 3.
        **kwargs : dict
            Additional kwargs for find_peaks.

        Sets peaks and properties

        """
        prominence *= self.counts.max()
        
        self.peaks, self.peak_properties = find_peaks(
            self.counts, prominence=prominence, width=width, **kwargs
        )
        
        # save parameters to dict for later reference
        self.peak_setting_parameters = kwargs
        self.peak_setting_parameters['prominence'] = prominence
        self.peak_setting_parameters['width'] = width

    def bigaussian_fit_from_peak(self, peak_idx):
        """Find kernel parameters for a peak with the shape of a bigaussian."""
        assert hasattr(self, 'peaks'), 'call set_peaks first' 
        assert hasattr(self, 'kernel_params'), 'cant finetune without initial guess'
        rt_idx = self.peaks[peak_idx]  # mz index of of center 
        
        # width of peak at half maximum
        idx_l = (self.peak_properties["left_ips"][peak_idx] + .5).astype(int)
        idx_r = (self.peak_properties["right_ips"][peak_idx] + .5).astype(int)
        mask = slice(idx_l, idx_r)
        rt_c, H, sigma_l, sigma_r = self.kernel_params[peak_idx, :]
        if len(self.rts[mask]) < 4:
            return None
        
        bounds_l = [
            rt_c - sigma_l / 4, 
            H * .8, 
            sigma_l * .8,
            sigma_r * .8
        ]
        bounds_r = [
            rt_c + sigma_r / 4,    
            H * 1.2,
            sigma_l * 1.2,
            sigma_r * 1.2
        ]
        # print(bounds_l, bounds_r, self.kernel_params[peak_idx, :])
        # print(len(bounds_l), len(bounds_r), len(self.kernel_params[peak_idx, :]))
        try:
            params, _ = curve_fit(
                f=self.bigaussian, 
                xdata=self.rts[mask], 
                ydata=self.counts[mask], 
                p0=self.kernel_params[peak_idx, :],
                bounds=(bounds_l, bounds_r)
            )
        except Exception as e:
            warnings.warn(
                f'Unhandled exception in curve_fit for index {peak_idx}:' +
                str(e)
            )
            return None
        return params

    def set_kernels(self, fine_tune=True):
        """
        Based on the peak properties, find bigaussian parameters to 
        approximate spectrum. Creates kernel_params where cols correspond to 
        peaks and rows different properties. Properties are: m/z, vertical 
        shift, intensity at max, sigma left, sigma right
        """
        assert hasattr(self, 'peaks'), 'call set peaks first'
        
        y = self.counts.copy()
        
        self.kernel_params = np.zeros((len(self.peaks), 4))
        # start with heighest peak, work down
        idxs_peaks = np.argsort(self.peak_properties['prominences'])[::-1]
        mask_valid = np.ones(len(self.peaks), dtype=bool)
        for idx in idxs_peaks:
            rt_c, H, sigma_l, sigma_r = self.bigaussian_from_peak(idx)
            if (H <= 0) or (sigma_l <= 0) or (sigma_r <= 0):
                mask_valid[idx] = False
                continue
            else:
                self.kernel_params[idx, :] = rt_c, H, sigma_l, sigma_r
            if fine_tune:
                params = self.bigaussian_fit_from_peak(idx)
                if params is not None:
                    self.kernel_params[idx, :] = params
                else: 
                    mask_valid[idx] = False
                    continue
            self.counts -= self.bigaussian(
                self.rts, *self.kernel_params[idx, :]
            )
        # restore counts
        self.counts = y
        # delete invalid peaks
        self.kernel_params = self.kernel_params[mask_valid, :]
        self.peaks = self.peaks[mask_valid]
        props_new = {}
        for k, v in self.peak_properties.items():
            props_new[k] = v[mask_valid]
        self.peak_properties = props_new
        
    def get_estimated_spectrum(self):
        assert hasattr(self, 'kernel_params'), 'call set_kernels first'
        # calculate approximated signal by summing up kernels
        counts_approx = np.zeros_like(self.counts, dtype=float)
        
        for i in range(len(self.peaks)):
            y = self.bigaussian(self.rts, *self.kernel_params[i, :]).astype(float)
            counts_approx += y
        return counts_approx


class Spectrum(BaseClassSpectrum):
    def __init__(
            self, path_file = None, rts = None, counts = None, 
            limits: tuple[float] | None = None
    ):
        """
        Convert rtms spectrum to python.

        Parameters
        ----------
        rspectrum : rtms.Spectrum
            The rtms object to convet.
        limits : tuple[float] | None, optional
            mz limits to crop the spectrum as a tuple with upper and lower bound. 
            The default is None and will not crop the spectrum.

        Returns
        -------
        None.

        """
        assert (path_file is not None) or ((rts is not None) and (counts is not None)), \
            "provide either a path or retention times with counts"
            
        if path_file is not None:
            df = pd.read_csv(path_file, sep=',')
            self.rts = df['RT'].to_numpy()
            self.counts = df['counts'].to_numpy()
        else:
            self.rts = rts
            self.counts = counts
        if limits is not None:  # if limits provided, crop spectrum to interval
            mask = (self.rts >= limits[0]) & (self.rts <= limits[1])
            self.rts = self.rts[mask]
            self.counts = self.counts[mask]
            
    
        
    def resample(
            self, 
            delta_rt: float | Iterable[float] = 1e-4 / 3, 
            check_intervals = False
    ):
        """
        Resample mzs and intensities to regular intervals.
        
        Provide either the equally spaced mz values at which to resample or 
        the distance for regular intervals resampling.
        """
        if type(delta_rt) in (float, int):
            # create mzs spaced apart with specified precision
            # round to next smallest multiple of delta_mz
            smallest_rt = int(self.rts.min() / delta_rt) * delta_rt
            # round to next biggest multiple of delta_mz
            biggest_rt = (int(self.rts.max() / delta_rt) + 1) * delta_rt
            # equally spaced
            rts_ip = np.arange(smallest_rt, biggest_rt + delta_rt, delta_rt)
        else:
            if check_intervals:
                drt = np.diff(delta_rt)
                assert np.allclose(drt[1:], drt[0]), \
                    'passed delta_rt must either be float or list of equally spaced mzs'
            rts_ip = delta_rt
        # already same mzs, nothing todo
        if (len(self.rts) == len(rts_ip)) and np.allclose(self.rts, rts_ip):
            return
        # interpolate to regular spaced mz values
        ints_ip = np.interp(rts_ip, self.rts, self.counts)
        # overwrite objects mz vals and intensities
        self.rts = rts_ip
        self.counts = ints_ip
    
    def remove_outliers(
            self, window_length=101, diff_threshold=.001, plts=False
    ) -> np.ndarray:
        # smoothed = convolve(
        #     self.counts, np.ones(window_length) / window_length,
        #     mode='same'
        # )
        smoothed = median_filter(self.counts, size=window_length, mode='nearest')
        # diff = np.abs(self.counts - smoothed)
        # normalize
        # diff /= np.max(diff)
        # outliers = diff > diff_threshold

        if plts:        
            counts_before = self.counts.copy()
        # rts_new = self.rts[~outliers].copy()
        # counts_new = self.counts[~outliers].copy()
        
        self.counts = smoothed

        if plts:
            plt.figure()
            plt.plot(self.rts, counts_before, label='original')
            plt.plot(self.rts, smoothed, label='smoothed')
            # plt.scatter(
            #     self.rts[outliers], counts_before[outliers], 
            #     label='detected outliers'
            # )
            # plt.plot(self.rts, self.counts, label='denoised')
            plt.legend()
            plt.show()

    # def remove_outliers(
    #         self, quantile=95, plts=False
    # ) -> np.ndarray:
    #     diff = np.append(np.abs(np.diff(self.counts)), [0])
    #     # normalize
    #     thr = np.percentile(diff, q=quantile)
    #     outliers = diff > thr
        
    #     rts = self.rts.copy()
    #     counts = self.counts.copy()
        
    #     rts = self.rts.copy()
    #     self.rts = self.rts[~outliers]
    #     self.counts = self.counts[~outliers]
        
    #     self.resample(delta_rt=rts)
        
    #     if plts:
    #         plt.figure()
    #         plt.plot(rts, counts, label='original')
    #         plt.scatter(
    #             rts[outliers], counts[outliers], 
    #             label='detected outliers'
    #         )
    #         plt.plot(self.rts, self.counts, label='denoised')
    #         plt.legend()
    #         plt.show()

    def subtract_base_line(
            self, window_size_portion=1/50, plts=False, **kwargs
    ):        
        window_size = int(len(self.rts) * window_size_portion + .5)
        ys_min = minimum_filter(self.counts, size=window_size)
        
        if plts:
            plt.figure()
            plt.plot(self.rts, self.counts, label='signal')
            plt.plot(self.rts, self.counts - ys_min, label='baseline corrected')
            plt.plot(self.rts, ys_min, label='baseline')
            plt.xlabel('retention time in min')
            plt.ylabel('Counts')
            plt.legend()
            plt.show()
        
        self.counts -= ys_min
        
    def plt_summed(self, plt_kernels=False):
        assert hasattr(self, 'kernel_params'), 'call set_kernels first'
        # calculate approximated signal by summing up kernels
        counts_approx = np.zeros_like(self.counts, dtype=float)
        plt.figure()
        
        for i in range(len(self.peaks)):
            y = self.bigaussian(self.rts, *self.kernel_params[i, :]).astype(float)
            counts_approx += y
            if plt_kernels:
                plt.plot(self.rts, y)
        plt.plot(self.rts, self.counts, label='summed intensity')
        plt.plot(self.rts, counts_approx, label='estimated')
        plt.legend()
        plt.xlabel(r'retention time in min')
        plt.ylabel('Counts')
        plt.show()
        
    def xcorr(self, other, plts=False, max_time_offset=None):
        """
        Calculate crosscorrelation for a and b within a certain lag window.

        Parameters
        ----------
        a : Iterable
            DESCRIPTION.
        b : Iterable
            DESCRIPTION.
        max_time_offset : float | None, optional
            The maximal allowed time difference between the two spectra in 
            minutes. The default is 10 seconds. None will not restrict the search
            space.

        Returns
        -------
        lags : TYPE
            DESCRIPTION.
        corrs : TYPE
            DESCRIPTION.

        """
        diffs = np.diff(self.rts)
        assert np.allclose(diffs[0], diffs[1:]), \
            'equally spaced sampling required for xcorr'
        
        other = other.copy()
        other.resample(self.rts)
        a = self.counts
        b = other.counts
        N = len(b)

        # b = np.pad(b.copy(), maxlags)    
        lags = correlation_lags(N, N, mode='full')
        times = diffs[0] * lags
        # corrs = np.array([
        #     np.correlate(
        #         a, 
        #         b[maxlags + lag:maxlags + lag + N], 
        #         mode='valid'
        #     )[0] for lag in lags
        # ])
        corrs = correlate(a, b, mode='full')
        if max_time_offset is not None:
            mask = np.abs(times) <= max_time_offset
        else:
            mask = np.ones_like(corrs, dtype=bool)
        corrs[~mask] = 0
        idx = np.argmax(corrs)
        lag = lags[idx]
        time = times[idx]
        if plts:
            plt.figure()
            plt.plot(times[mask] * 60, corrs[mask])
            plt.plot(time * 60, corrs[idx], 'ro')
            plt.xlabel('time in seconds')
            plt.ylabel('Correlation')
            plt.title(f'{lag=}, time shift = {time*60:.1f} seconds')
            plt.show()
        return time
    
    def bin_spectrum(self):
        """Find intensities of compound based on kernels."""
        # weight is the integrated weighted signal
        # ideally this would take the integral but since mzs are equally 
        # spaced, we can use the sum (scaled accordingly), so instead of
        # line_spectrum[idx_peak] = np.trapz(weighted_signal, x=self.mzs)
        # take
        # line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz
        #
        # and instead of summing over peaks we can write this as matrix
        # multiplication
        #
        # equivalent to 
        # line_spectrum = np.zeros(N_peaks)
        # for idx_peak in range(N_peaks):
        #     weighted_signal = spectrum.intensities * bigaussians[idx_peak, :]
        #     line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz
        # precompute bigaussians
        N_peaks = len(self.peaks)
        
        dmt = np.abs(np.diff(self.counts)[0])
        kernels = np.zeros((N_peaks, len(self.rts)))
        for idx_peak in range(N_peaks):
            # x_c, H, sigma_l, sigma_r
            sigma_l = self.kernel_params[idx_peak, 2]
            sigma_r = self.kernel_params[idx_peak, 3]
            H = np.sqrt(2 / np.pi) / (sigma_l + sigma_r)  # normalization constant
            I0 = self.kernel_params[idx_peak, -1]
            kernels[idx_peak] = self.bigaussian(
                self.rts, 
                x_c=self.kernel_params[idx_peak, 0], 
                H=H,  # normalized kernels
                sigma_l=sigma_l, 
                sigma_r=sigma_r
            )
        kernels = kernels.T
        
        line_spectrum = (self.counts @ kernels) * dmt
        self.line_spectrum = line_spectrum
    
    def copy(self):
        new = Spectrum(rts=self.rts.copy(),counts=self.counts.copy())
        return new
        
class Spectra(BaseClassSpectrum):
    """Container for multiple Spectrum objects and binning."""
    def __init__(
            self, 
            list_path_files: list[str] = None,
            spectra: list[Spectrum] = None,
            limits: tuple[float] = None,
            delta_rt = 1e-4 / 3
    ):
        """
        Initiate the object. 
        
        Either provide a list of filesfrom which spectra will be analyzed or 
        a list of spectra.

        Parameters
        ----------
        list_path_files: list[str]
            List containing the files to be analized.
        spectra: list[Spectrum]
            List containing spectra classes
        limits : tuple[float], optional
            Limits for the retention times. The default is None.

        Returns
        -------
        None.

        """        
        assert (list_path_files is not None) or (spectra is not None), \
            'provide either a list of files or a list of spectra'
        
        if list_path_files is not None:
            self.list_path_files = list_path_files
            spectra = [
                Spectrum(path_file=file, limits=limits) 
                for file in list_path_files
            ]
        self.spectra = spectra
        # ensure uniform measurment points
        self.delta_rt = delta_rt
        self.resample_all(delta_rt=self.delta_rt)
        
    def resample_all(self, **kwargs):
        for spec in self.spectra:
            spec.resample(**kwargs)
        
    def remove_outliers_all(self, **kwargs):
        for spec in self.spectra:
            spec.remove_outliers(**kwargs)
    
    def subtract_base_line_all(self, **kwargs):
        for spec in self.spectra:
            spec.subtract_base_line(**kwargs)        
    
    def set_lag_table(self, maxlags=500):
        n = len(self.spectra)
        table = np.zeros((n, n))
        indices =  np.triu_indices_from(table)
        
        for i, j in zip(*indices):
            if i == j:
                continue
            spec_a = self.spectra[i]
            spec_b = self.spectra[j]
            dist = spec_a.xcorr(spec_b, maxlags=maxlags)
            table[i, j] = dist
            
        # fill lower triangle matrix
        self.lag_table = table - table.T
    
    def get_rts_extent(self):
        rt_min = np.infty
        rt_max = -np.infty
        for spec in self.spectra:
            # update min and max retention time
            if (rt_lower := spec.rts.min()) < rt_min:
                rt_min = rt_lower
            if (rt_upper := spec.rts.max()) > rt_max:
                rt_max = rt_upper
        return rt_min, rt_max
    
    def align_spectra(self, **kwargs):
        # align everything to first spectrum
        spec0 = self.spectra[0]
        
        for spec in self.spectra[1:]:
            time_shift = spec0.xcorr(spec, **kwargs)
            spec.rts += time_shift
            print(f'shifted spec by {time_shift*60:.1f} seconds')
            
        # resample
        rt_min, rt_max = self.get_rts_extent()
        rts = np.arange(rt_min, rt_max + self.delta_rt, self.delta_rt)
        self.resample_all(delta_rt=rts)
    
    def set_peaks_all(self, **kwargs):
        for spec in self.spectra:
            spec.set_peaks(**kwargs)
            
    def set_kernels_all(self, **kwargs):
        for spec in self.spectra:
            spec.set_kernels(**kwargs)
        
    def set_summed(self):
        rt_min, rt_max = self.get_rts_extent()
        rts = np.arange(rt_min, rt_max + self.delta_rt, self.delta_rt)
        counts = np.zeros_like(rts)
        for spec in self.spectra:
            mask = slice(
                np.argmin(np.abs(spec.rts.min() - rts)),
                np.argmin(np.abs(spec.rts.max() - rts)) + 1
            )
            counts[mask] += spec.counts
        self.rts, self.counts = rts, counts
        
    def bin_spectrum(self, **kwargs):
        def _bin_spectrum(spectrum, idx):
            """Find intensities of compound based on kernels."""
            # weight is the integrated weighted signal
            # ideally this would take the integral but since mzs are equally 
            # spaced, we can use the sum (scaled accordingly), so instead of
            # line_spectrum[idx_peak] = np.trapz(weighted_signal, x=self.mzs)
            # take
            # line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz
            #
            # and instead of summing over peaks we can write this as matrix
            # multiplication
            #
            # equivalent to 
            # line_spectrum = np.zeros(N_peaks)
            # for idx_peak in range(N_peaks):
            #     weighted_signal = spectrum.intensities * bigaussians[idx_peak, :]
            #     line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz
            line_spectrum = (spectrum.counts @ kernels) * self.delta_rt
            self.line_spectra[idx, :] = line_spectrum
        
        self.resample_all(delta_rt=self.rts)
        N_spectra = len(self.spectra)  # number of spectra 
        N_peaks = len(self.peaks)  # number of identified peaks
        self.line_spectra = np.zeros((N_spectra, N_peaks))  # result array
        
        # precompute bigaussians
        kernels = np.zeros((N_peaks, len(self.rts)))
        for idx_peak in range(N_peaks):
            # x_c, H, sigma_l, sigma_r
            sigma_l = self.kernel_params[idx_peak, 2]
            sigma_r = self.kernel_params[idx_peak, 3]
            H = np.sqrt(2 / np.pi) / (sigma_l + sigma_r)  # normalization constant
            kernels[idx_peak] = self.bigaussian(
                self.rts, 
                x_c=self.kernel_params[idx_peak, 0], 
                H=H,  # normalized kernels
                sigma_l=sigma_l, 
                sigma_r=sigma_r
            )
        kernels = kernels.T
        
        # iterate over spectra and bin according to kernels
        print(f'binning {N_spectra} spectra into {N_peaks} bins ...')
        time0 = time.time()
        for it, spectrum in enumerate(self.spectra):
            _bin_spectrum(spectrum, it)
            if it % 10 ** (np.around(np.log10(N_spectra), 0) - 2) == 0:
                time_now = time.time()
                time_elapsed = time_now - time0
                predict = time_elapsed * N_spectra / (it + 1)
                print(f'estimated time left: {(predict - time_elapsed):.1f} s')
        print('done binning spectra')
            
    def plt_summed(self, plt_all=False, plt_kernels=False):
        if not hasattr(self, 'counts'):
            self.set_summed()
        if hasattr(self, 'kernel_params'):
            counts_approx = self.get_estimated_spectrum()
        plt.figure()
        for spec in self.spectra:
            mask = slice(
                np.argmin(np.abs(spec.rts.min() - self.rts)),
                np.argmin(np.abs(spec.rts.max() - self.rts)) + 1
            )
            if plt_all:
                plt.plot(spec.rts, spec.counts)
        plt.plot(self.rts, self.counts, label='summed')
        if hasattr(self, 'kernel_params'):
            plt.plot(self.rts, counts_approx, label='approximated')
        plt.xlabel('retention time in minutes')
        plt.ylabel('Summed counts')
        plt.show()
        
    def get_dataframe(self):
        """Turn the line_spectra into the familiar df with R, x, y columns."""
        assert hasattr(self, 'line_spectra'), 'create line spectra with bin_spectra'
        
        if hasattr(self, 'list_path_files'):
            index = [os.path.basename(file) for file in self.list_path_files]
        else:
            index = None
        
        df = pd.DataFrame(
            data=self.line_spectra.copy(), 
            columns=np.around(self.kernel_params[:, 0], 4).astype(str),
            index=index
        )
    
        return df
    
#     def set_masses(self):
#         """Initiate masses and intensities summed spectrum."""
#         # round to next smallest multiple of delta_mz
#         smallest_mz = int(self.limits[0] / self.delta_mz) * self.delta_mz
#         # round to next biggest multiple of delta_mz
#         biggest_mz = (int(self.limits[1] / self.delta_mz) + 1) * self.delta_mz
#         # equally spaced
#         self.mzs =  np.arange(smallest_mz, biggest_mz + self.delta_mz, self.delta_mz)
#         self.intensities = np.zeros_like(self.mzs)
    
#     def initiate(self, reader, indices, limits):
#         """Set limits and masses."""
#         if indices is None:
#             if not hasattr(reader, 'indices'):
#                 reader.create_indices()    
#             indices = reader.indices
#         self.indices = indices
#         if limits is None:
#             if not hasattr(reader, 'metaData'):
#                 reader.set_meta_data()
#             reader.set_QTOF_window()
#             limits = reader.limits
#         self.limits = limits
#         self.set_masses()
    
#     def add_spectrum(self, spectrum: Spectrum):
#         """Add passed spectrum values to summed spectrum."""
#         # spectrum = spectrum.copy()
#         spectrum.resample(self.mzs)

#         self.intensities += spectrum.intensities
    
#     def add_all_spectra(self, reader, **kwargs):
#         """Add up all spectra found in the mcf file."""
#         N = len(self.indices)
#         print(f'adding up {N} spectra ...')
            
#         time0 = time.time()
#         # iterate over all spectra
#         for it, index in enumerate(self.indices):
#             spectrum = reader.get_spectrum(int(index)) 
#             self.add_spectrum(spectrum, **kwargs)
#             time_now = time.time()
#             if it % 10 ** (np.around(np.log10(N), 0) - 2) == 0:
#                 time_elapsed = time_now - time0
#                 predict = time_elapsed * N / (it + 1)
#                 print(f'estimated time left: {(predict - time_elapsed):.1f} s')
#         print('done adding up spectra')
        
#     def set_peaks(self, prominence: float | None = None, width=3, **kwargs):
#         """
#         Find peaks in summed spectrum using scipy's find_peaks function.

#         Parameters
#         ----------
#         prominence : float, optional
#             Required prominence for peaks. The default is None. This defaults
#             to 10 % of the median intensity
#         width : int, optional
#             Minimum number of points between peaks. The default is 3.
#         **kwargs : dict
#             Additional kwargs for find_peaks.

#         Sets peaks and properties

#         """
#         if prominence is None:
#             median = np.median(self.intensities)
#             prominence = .1 * median
        
#         self.peaks, self.peak_properties = find_peaks(
#             self.intensities, prominence=prominence, width=width, **kwargs
#         )
        
#         # save parameters to dict for later reference
#         self.peak_setting_parameters = kwargs
#         self.peak_setting_parameters['prominence'] = prominence
#         self.peak_setting_parameters['width'] = width
    
#     def gaussian_from_peak(self, peak_idx):
#         assert hasattr(self, 'peaks'), 'call set_peaks first' 
#         mz_idx = self.peaks[peak_idx]  # mz index of of center 
        
        
#         H = self.intensities[mz_idx] # corresponding height
#         # width of peak at half maximum
#         FWHM_l = self.mzs[
#             (self.peak_properties["left_ips"][peak_idx] + .5).astype(int)
#         ]
#         FWHM_r = self.mzs[
#             (self.peak_properties["right_ips"][peak_idx] + .5).astype(int)
#         ]
#         mz_c = (FWHM_l + FWHM_r) / 2
#         # convert FWHM to standard deviation
#         sigma_l = -(FWHM_l - mz_c) / (2 * np.log(2))
#         sigma_r = (FWHM_r - mz_c) / (2 * np.log(2))        
#         sigma = (sigma_l + sigma_r) / 2
#         return mz_c, H, sigma
    
#     def kernel_fit_from_peak(self, peak_idx):
#         """Find kernel parameters for a peak with the shape of a bigaussian."""
#         assert hasattr(self, 'peaks'), 'call set_peaks first' 
#         mz_idx = self.peaks[peak_idx]  # mz index of of center 
        
#         # width of peak at half maximum
#         idx_l = (self.peak_properties["left_ips"][peak_idx] + .5).astype(int)
#         idx_r = (self.peak_properties["right_ips"][peak_idx] + .5).astype(int)
#         mask = slice(idx_l, idx_r)
#         if hasattr(self, 'kernel_params') and np.any(self.kernel_params[peak_idx, :]):
#             mz_c, H, sigma, *sigma_r = self.kernel_params[peak_idx, :]
#             bounds_l = [
#                 mz_c - sigma / 4, 
#                 H * .8, 
#                 sigma * .8
#             ]
#             bounds_r = [
#                 mz_c + sigma / 4,    
#                 H * 1.2,
#                 sigma * 1.2
#             ]
#             if len(sigma_r) > 0:
#                 bounds_l.append(sigma_r[0] * .8)
#                 bounds_r.append(sigma_r[0] * 1.2)
#             params, _ = curve_fit(
#                 f=self.kernel_func, 
#                 xdata=self.mzs[mask], 
#                 ydata=self.intensities[mask], 
#                 p0=self.kernel_params[peak_idx, :],
#                 bounds=(bounds_l, bounds_r)
#             )
#         else:
#             return self.kernel_params[peak_idx, :]
        
#         return params
    
#     def bigaussian_from_peak(self, peak_idx: int, use_offset=False):
#         """Find kernel parameters for a peak with the shape of a bigaussian."""
#         assert hasattr(self, 'peaks'), 'call set_peaks first' 
#         mz_idx = self.peaks[peak_idx]  # mz index of of center 
#         mz_c = self.mzs[mz_idx] # center of gaussian
#         # height at center of peak - prominence
#         H = self.intensities[mz_idx] # corresponding height
#         # width of peak at half maximum
#         FWHM_l = self.mzs[
#             (self.peak_properties["left_ips"][peak_idx] + .5).astype(int)
#         ]
#         FWHM_r = self.mzs[
#             (self.peak_properties["right_ips"][peak_idx] + .5).astype(int)
#         ]
#         # convert FWHM to standard deviation
#         sigma_l = -(FWHM_l - mz_c) / (2 * np.log(2))
#         sigma_r = (FWHM_r - mz_c) / (2 * np.log(2))        
#         return mz_c, H, sigma_l, sigma_r
    
#     @staticmethod
#     def gaussian(x: np.ndarray, x_c, H, sigma):
#         return H * np.exp(-1/2 * ((x - x_c) / sigma) ** 2)
    
#     @staticmethod
#     def bigaussian(x: np.ndarray, x_c, H, sigma_l, sigma_r):
#         """
#         Evaluate bigaussian for mass vector based on parameters.

#         Parameters
#         ----------
#         x : np.ndarray
#             mass vector.
#         x_c : float
#             mass at center of peak
#         H : float
#             Amplitude of peak.
#         sigma_l : float
#             left-side standard deviation.
#         sigma_r : float
#             right-side standard deviation.

#         Returns
#         -------
#         np.ndarray
#             Intensities of bigaussian.

#         """
#         x_l = x[x <= x_c]
#         x_r = x[x > x_c]
#         y_l = H * np.exp(-1/2 * ((x_l - x_c) / sigma_l) ** 2)
#         y_r = H * np.exp(-1/2 * ((x_r - x_c) / sigma_r) ** 2)
#         return np.hstack([y_l, y_r])
    
#     @property
#     def kernel_func(self):
#         if self.kernel_shape == 'bigaussian':
#             return self.bigaussian
#         elif self.kernel_shape == 'gaussian':
#             return self.gaussian
        
#     @property
#     def kernel_func_from_peak(self):
#         if self.kernel_shape == 'bigaussian':
#             return self.bigaussian_from_peak
#         elif self.kernel_shape == 'gaussian':
#             return self.gaussian_from_peak
    
#     def set_kernels(self, use_bigaussian=False, fine_tune=False):
#         """
#         Based on the peak properties, find bigaussian parameters to 
#         approximate spectrum. Creates kernel_params where cols correspond to 
#         peaks and rows different properties. Properties are: m/z, vertical 
#         shift, intensity at max, sigma left, sigma right
#         """
#         assert hasattr(self, 'peaks'), 'call set peaks first'
#         if use_bigaussian: 
#             self.kernel_shape = 'bigaussian'
#             self.kernel_params = np.zeros((len(self.peaks), 4))
#         else:
#             self.kernel_shape = 'gaussian'
#             self.kernel_params = np.zeros((len(self.peaks), 3))
#         kernel_func_from_peak = self.kernel_func_from_peak
        
#         for idx in range(len(self.peaks)):
#             self.kernel_params[idx, :] = kernel_func_from_peak(idx)
#             if fine_tune:
#                 try:
#                     params = self.kernel_fit_from_peak(idx)
#                     self.kernel_params[idx, :] = params
#                 except RuntimeError as e:
#                     print(idx, e)
#                 except TypeError as e:
#                     print(idx, e)
#         # vertical shifts get taken care of by taking sum
        
#     def plt_summed(self, plt_kernels=False):
#         assert hasattr(self, 'kernel_params'), 'call set_kernels first'
#         # calculate approximated signal by summing up kernels
#         intensities_approx = np.zeros_like(self.intensities)
#         plt.figure()
        
#         for i in range(len(self.peaks)):
#             y = self.kernel_func(self.mzs, *self.kernel_params[i, :])
#             intensities_approx += y
#             if plt_kernels:
#                 plt.plot(self.mzs, y)
#         plt.plot(self.mzs, self.intensities, label='summed intensity')
#         plt.plot(self.mzs, intensities_approx, label='estimated')
#         plt.legend()
#         plt.xlabel(r'$m/z$ in Da')
#         plt.ylabel('Intensity')
#         plt.show()
        
#     def bin_spectra(self, reader: ReadBrukerMCF):
#         """For each spectrum find overlap between kernels and signal."""
#         def _bin_spectrum(spectrum, idx):
#             """Find intensities of compound based on kernels."""
#             if (len(spectrum.mzs) != len(self.mzs)) \
#                     or (not np.allclose(spectrum.mzs, self.mzs)):
#                 spectrum.resample(self.mzs)
#             # weight is the integrated weighted signal
#             # ideally this would take the integral but since mzs are equally 
#             # spaced, we can use the sum (scaled accordingly), so instead of
#             # line_spectrum[idx_peak] = np.trapz(weighted_signal, x=self.mzs)
#             # take
#             # line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz
#             #
#             # and instead of summing over peaks we can write this as matrix
#             # multiplication
#             #
#             # equivalent to 
#             # line_spectrum = np.zeros(N_peaks)
#             # for idx_peak in range(N_peaks):
#             #     weighted_signal = spectrum.intensities * bigaussians[idx_peak, :]
#             #     line_spectrum[idx_peak] = np.sum(weighted_signal) * dmz
#             line_spectrum = (spectrum.intensities @ kernels) * dmz
#             self.line_spectra[idx, :] = line_spectrum
        
#         assert hasattr(self, 'kernel_params'), 'calculate kernels with set_kernels'
        
#         indices_spectra = self.indices
#         N_spectra = len(indices_spectra)  # number of spectra in mcf file
#         N_peaks = len(self.peaks)  # number of identified peaks
#         self.line_spectra = np.zeros((N_spectra, N_peaks))  # result array
        
#         # precompute bigaussians
#         dmz = self.mzs[1] - self.mzs[0]
#         kernels = np.zeros((N_peaks, len(self.mzs)))
#         if self.kernel_shape == 'bigaussian':
#             for idx_peak in range(N_peaks):
#                 # x_c, H, sigma_l, sigma_r
#                 sigma_l = self.kernel_params[idx_peak, 2]
#                 sigma_r = self.kernel_params[idx_peak, 3]
#                 H = np.sqrt(2 / np.pi) / (sigma_l + sigma_r)  # normalization constant
#                 kernels[idx_peak] = self.bigaussian(
#                     self.mzs, 
#                     x_c=self.kernel_params[idx_peak, 0], 
#                     H=H,  # normalized kernels
#                     sigma_l=sigma_l, 
#                     sigma_r=sigma_r
#                 )
#             kernels = kernels.T
#         elif self.kernel_shape == 'gaussian':
#             for idx_peak in range(N_peaks):
#                 # x_c, H, sigma
#                 sigma = self.kernel_params[idx_peak, 2]
#                 H = 1 / (np.sqrt(2 * np.pi) * sigma)  # normalization constant
#                 kernels[idx_peak] = self.gaussian(
#                     self.mzs, 
#                     x_c=self.kernel_params[idx_peak, 0], 
#                     H=H,  # normalized kernels
#                     sigma=sigma
#                 )
#             kernels = kernels.T
        
#         # iterate over spectra and bin according to kernels
#         print(f'binning {N_spectra} spectra into {N_peaks} bins ...')
#         time0 = time.time()
#         for it, idx_spectrum in enumerate(indices_spectra):
#             spectrum = reader.get_spectrum(int(idx_spectrum)) 
#             _bin_spectrum(spectrum, it)
#             if it % 10 ** (np.around(np.log10(N_spectra), 0) - 2) == 0:
#                 time_now = time.time()
#                 time_elapsed = time_now - time0
#                 predict = time_elapsed * N_spectra / (it + 1)
#                 print(f'estimated time left: {(predict - time_elapsed):.1f} s')
#         print('done binning spectra')
        
#     def binned_spectra_to_df(self, reader: ReadBrukerMCF):
#         """Turn the line_spectra into the familiar df with R, x, y columns."""
#         if hasattr(self, 'feature_table'):
#             return self.feature_table
#         assert hasattr(self, 'line_spectra'), 'create line spectra with bin_spectra'
#         if not hasattr(reader, 'spots'):
#             reader.create_spots()
        
#         df = pd.DataFrame(
#             data=self.line_spectra.copy(), 
#             columns=np.around(self.kernel_params[:, 0], 4).astype(str)
#         )
#         # add R, x, y columns
#         names = reader.spots.names
#         str_prefix = r'R(\d+)X'
#         str_x = r'R\d+X(.*?)Y'
#         str_y = r'Y(.*?)$'
        
#         def rxy(name):
#             r = int(re.findall(str_prefix, name)[0])
#             x = int(re.findall(str_x, name)[0])
#             y = int(re.findall(str_y, name)[0])
#             return [r, x, y]
#         RXYs = np.array([rxy(name) for name in names])
#         if self.indices.shape != reader.indices.shape:
#             mask = np.array(
#                 [np.argwhere(idx == reader.indices)[0][0] for idx in self.indices]
#             )
#         else:
#             mask = np.ones_like(self.indices, dtype=bool)
#         df['R'] = RXYs[mask, 0]
#         df['x'] = RXYs[mask, 1]
#         df['y'] = RXYs[mask, 2]
#         self.feature_table = df
#         return self.feature_table
    
#     def get_kernel_params_df(self):
#         assert hasattr(self, 'kernel_params'), 'call set_kernels'
#         if self.kernel_shape == 'bigaussian':
#             columns = ['mz', 'H', 'sigma_l', 'sigma_r']
#         elif self.kernel_shape == 'gaussian':
#             columns = ['mz', 'H', 'sigma']
#         df = pd.DataFrame(data=self.kernel_params, columns=columns)
#         return df
        
#     def save(self):
#         """Save object to d-folder."""
#         dict_backup = self.__dict__.copy()
#         # dont save feature table AND line spectra
#         if hasattr(self, 'feature_table') and hasattr(self, 'line_spectra'):
#             self.__delattr__('line_spectra')
#         keep_attributes = set(self.__dict__.keys()) & class_to_attributes(self)
#         existent_attributes = list(self.__dict__.keys())
#         for attribute in existent_attributes:
#             if attribute not in keep_attributes:
#                 self.__delattr__(attribute)
        
#         file = self.path_d_folder + '/' + 'spectra_object.pickle'
#         with open(file, 'wb') as inp:
#             pickle.dump(self, inp, pickle.HIGHEST_PROTOCOL)
#         self.__dict__ = dict_backup
        
#     def load(self):
#         """Load object from d-folder."""
#         file = self.path_d_folder + '/' + 'spectra_object.pickle'
#         with open(file, 'rb') as inp:
#             self.__dict__ = pickle.load(inp).__dict__
        

if __name__ == '__main__':
    pass
    
    
            
        

