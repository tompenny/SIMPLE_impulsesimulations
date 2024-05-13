import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisig
import scipy.optimize as opt
from numba import njit, jit
import scipy.io as sio

def transfer_function(w, w0, y0, yfb):
    # Exact transfer function of particle
    A = np.sqrt(4*kb*T*y0*M)
    chi = (M*(w0**2-w**2-1j*w*(y0+yfb)))**(-1)
    return A*chi

def transfer_function2(w, w0, y0, yfb, rnd, rnd2):
    # Transfer function of particle with noise for finite time measurement
    A = np.sqrt(4*kb*T*y0*M)
    chi = (M*(w0**2-w**2-1j*w*(y0+yfb)))**(-1)
    return A*chi*(rnd+1j*rnd2)


def impulse_resp(time, t0, A, y, w0):
    output = np.zeros(len(time))
    y1 = y/2 # factor two required by definition
    w1 = np.sqrt(w0**2 - y1**2)
    for n, t in enumerate(time):
        if t < t0:
            output[n] = 0
        if t > t0:
            output[n] =  A*np.sin(w1*(t-t0))*np.exp(-y1*(t-t0))
    return output

def generate_displacement(w, w0, y0, yfb, rnd, rnd2, rnd3, t0, A):
    """
    Generates the frequency domain response of a particle then reverse fourier transforms to the time domain
    Adds an impulse response and imprecision noise in the time domain
    Returns time displacement and the independent impulse response
    w: frequency bins for frequency domain response
    w0: natural frequency of particle
    y0: intrinsic damping of particle (gas and/or laser)
    yfb: Additional damping from cold feedback mechanism
    rnd and rnd2: Noise for fourier domain - should each have 0 mean and 1/sqrt(2) width I think
    rnd3: imprecision noise - should be 0 mean with width of Snn/2/dtn where Snn of single-sided PSD noise value in m^2/Hz and dtn is timestep in s (2/(max frequency in Hz))
    t0: impulse time in s - recommend doing half way through the trace
    A: response amplitude - in m
    """
    numbins = len(w)

    # Generate impulse response
    time = np.linspace(0, numbins/5/10**5/2, numbins)
    ir = impulse_resp(time, t0, A, y0+yfb, w0)

    # Generate frequency response of particle
    thermal_response = transfer_function2(w, w0, y0, yfb, rnd, rnd2)
    x = np.fft.irfft(thermal_response)[int(numbins/2-1):-int(numbins/2-1)] # Reverse fourier transform to time domain
    x = x*np.sqrt(np.trapz(np.abs(thermal_response**2), w/2/np.pi)/np.var(x)) # Scale to correct units
    x += ir # Add impulse response
    x += rnd3 # Add measurement noise

    return time, x, ir