import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisig
import scipy.optimize as opt
from numba import njit, jit
import scipy.io as sio

def transfer_function(w, w0, y0, yfb):
    """
    Exact transfer function of particle
    w: frequency bins for frequency domain response
    w0: natural frequency of particle
    y0: intrinsic damping of particle (gas and/or laser)
    yfb: Additional damping from cold feedback mechanism
    """
    A = np.sqrt(4*kb*T*y0*M)
    chi = (M*(w0**2-w**2-1j*w*(y0+yfb)))**(-1)
    return A*chi

def transfer_function2(w, w0, y0, yfb, rnd, rnd2):
    """
    Transfer function of particle with noise for finite time measurement
    w: frequency bins for frequency domain response
    w0: natural frequency of particle
    y0: intrinsic damping of particle (gas and/or laser)
    yfb: Additional damping from cold feedback mechanism
    rnd and rnd2: Noise for fourier domain - should each have 0 mean and 1/sqrt(2) width I think
    """
    A = np.sqrt(4*kb*T*y0*M)
    chi = (M*(w0**2-w**2-1j*w*(y0+yfb)))**(-1)
    return A*chi*(rnd+1j*rnd2)


def impulse_resp(time, t0, A, y0, yfb, w0):
    """
    Generates impulse response for particle
    t0: impulse time in s - recommend doing half way through the trace
    A: response amplitude - in m
    w0: natural frequency of particle
    y0: intrinsic damping of particle (gas and/or laser)
    yfb: Additional damping from cold feedback mechanism
    """
    output = np.zeros(len(time))
    y1 = (y0+yfb)/2 # factor two required by definition
    w1 = np.sqrt(w0**2 - y1**2)
    for n, t in enumerate(time):
        if t < t0:
            output[n] = 0
        if t > t0:
            output[n] =  A*np.sin(w1*(t-t0))*np.exp(-y1*(t-t0))
    return output

def generate_displacement(w, w0, y0, yfb, rnd, rnd2, rnd3, ir):
    """
    Generates the frequency domain response of a particle then reverse fourier transforms to the time domain
    Adds an impulse response and imprecision noise in the time domain
    Returns displacement
    w: frequency bins for frequency domain response
    w0: natural frequency of particle
    y0: intrinsic damping of particle (gas and/or laser)
    yfb: Additional damping from cold feedback mechanism
    rnd and rnd2: Noise for fourier domain - should each have 0 mean and 1/sqrt(2) width I think
    rnd3: imprecision noise - should be 0 mean with width of Snn/2/dtn where Snn of single-sided PSD noise value in m^2/Hz and dtn is timestep in s (1/(2*(max frequency in Hz)))
    ir: impulse response - must have same number of points as 2*len(w)
    """

    numbins = len(w)
    # Generate frequency response of particle
    thermal_response = transfer_function2(w, w0, y0, yfb, rnd, rnd2)
    x = np.fft.irfft(thermal_response)[int(numbins/2-1):-int(numbins/2-1)] # Reverse fourier transform to time domain and throw away start and end to get rid of finite time effects
    x = x*np.sqrt(np.trapz(np.abs(thermal_response**2), w/2/np.pi)/np.var(x)) # Scale to correct units
    x += ir # Add impulse response
    x += rnd3 # Add measurement noise

    return x

def generate_random_numbers(seeds, Snn, numbins, maxw):
    """
    Generates a set of random numbers of correct wdith and mean for generate displacement functions
    seeds: seeds for random numbers - should all be different
    Snn: Value of single-sided PSD of noise in m^2/Hz
    numbins: number of frequency bins
    maxw: maximum frequency in frequency domain in rad/s
    """
    np.random.seed(seeds[0])
    randomlist = np.random.normal(0, 0.5*np.sqrt(2), numbins)
    np.random.seed(seeds[1])
    randomlist2 = np.random.normal(0, 0.5*np.sqrt(2), numbins)
    np.random.seed(seeds[2])
    randomlist3 = np.random.normal(0, np.sqrt(Snn*maxw), numbins)
    return randomlist, randomlist2, randomlist3