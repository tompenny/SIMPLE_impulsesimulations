import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisig
import scipy.optimize as opt
from numba import njit, jit
import scipy.io as sio

def Linewidth(x, a,  x0, gamma): #with noise floor
    return a*(gamma)/((x0**2 - x**2)**2+(x*gamma)**2)  

def Linewidth2(x, a,  x0, gamma, c): #with noise floor
    return a*(gamma)/((x0**2 - x**2)**2+(x*gamma)**2) + c

def Gaussian(x, A, x0, sigma):
    return A*np.exp(-(x-x0)**2/(2*sigma**2))

from scipy.signal import butter, filtfilt, lfilter

def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a


def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, (low, high), btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def bandpass_peak_find(filename, cf, BW, fs, order):
    data = sio.loadmat(filename)
    x = data['x'][0]
    x = butter_bandpass_filter(x, cf-BW, cf+BW, fs = fs, order =order)
    max = np.max(x)
    return max

def bandpass_peak_find_noise(filename, cf, BW, fs, order):
    data = sio.loadmat(filename)
    x = data['x'][0]
    x = butter_bandpass_filter(x, cf-BW, cf+BW, fs = fs, order = order)
    m = int(np.random.uniform(0, len(x)))
    max = x[m]
    return max

def histogram_and_fit(amp_max, bin_num, count_amp, fit = True, plot = True):
    hist3, bins3 = np.histogram(amp_max, bin_num)
    bin_c = bins3[1:]-(bins3[1]-bins3[0])/2
    mean = np.mean(amp_max)
    std = np.std(amp_max)
    if fit == True:
        fit3, cov3 = opt.curve_fit(Gaussian, bin_c, hist3, p0 = [count_amp, mean, std])
        x_hist3 = np.linspace(bins3[0], bins3[-1], 100)
        fitted3 = Gaussian(x_hist3, *fit3)
    if plot == True:
        plt.stairs(hist3, bins3)
        plt.plot(x_hist3, fitted3)

    if fit == True:
        return hist3, bins3, fit3, x_hist3, fitted3
    else: 
        return hist3, bins3