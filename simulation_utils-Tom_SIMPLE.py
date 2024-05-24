import numpy as np
import h5py

def transfer_function(w, w0, y0, yfb, M, T):
    """
    Exact transfer function of particle
    w: frequency bins for frequency domain response
    w0: natural frequency of particle
    y0: intrinsic damping of particle (gas and/or laser)
    yfb: Additional damping from cold feedback mechanism
    M: mass of oscillator
    T: environment temperature
    """
    kb = 1.38*10**(-23) # Boltzmann constant
    A = np.sqrt(4*kb*T*y0*M)
    chi = (M*(w0**2-w**2-1j*w*(y0+yfb)))**(-1)
    return A*chi

def transfer_function2(w, w0, y0, yfb, rnd, rnd2, M, T):
    """
    Transfer function of particle with noise for finite time measurement
    w: frequency bins for frequency domain response
    w0: natural frequency of particle
    y0: intrinsic damping of particle (gas and/or laser)
    yfb: Additional damping from cold feedback mechanism
    rnd and rnd2: Noise for fourier domain - should each have 0 mean and 1/sqrt(2) width I think
    M: mass of oscillator
    T: environment temperature
    """
    kb = 1.38*10**(-23) # Boltzmann constant
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

def generate_displacement(w, w0, y0, yfb, M, T, rnd, rnd2, rnd3, ir):
    """
    Generates the frequency domain response of a particle then reverse fourier transforms to the time domain
    Adds an impulse response and imprecision noise in the time domain
    Returns displacement
    w: frequency bins for frequency domain response
    w0: natural frequency of particle
    y0: intrinsic damping of particle (gas and/or laser)
    yfb: Additional damping from cold feedback mechanism
    M: mass of oscillator
    T: environment temperature
    rnd and rnd2: Noise for fourier domain - should each have 0 mean and 1/sqrt(2) width I think
    rnd3: imprecision noise - should be 0 mean with width of Snn/2/dtn where Snn of single-sided PSD noise value in m^2/Hz and dtn is timestep in s (1/(2*(max frequency in Hz)))
    ir: impulse response - must have same number of points as 2*len(w)
    """

    numbins = len(w)
    # Generate frequency response of particle
    thermal_response = transfer_function2(w, w0, y0, yfb, rnd, rnd2, M, T)
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

    np.random.seed(int(seeds[0]))
    randomlist = np.random.normal(0, 0.5*np.sqrt(2), numbins)
    np.random.seed(int(seeds[1]))
    randomlist2 = np.random.normal(0, 0.5*np.sqrt(2), numbins)
    np.random.seed(int(seeds[2]))
    randomlist3 = np.random.normal(0, np.sqrt(Snn*maxw), numbins)
    return randomlist, randomlist2, randomlist3

def frequency_modulation(x, time, time2):
    """
    Mimics frequency modulation of an oscillator by modulating the time between points of a fixed frequency oscillator then interpolating back to a fixed time interval.
    Note: I think this will only work for smooth and small changes to the frequency since the interpolation back to fixed time is linear. I have not tested the limits of this.
    x: displacement to be modulated
    time: time base of data
    time2: modulated time base
    """

    x_mod = np.interp(time, time2, x)
    return x_mod


def generate_sawtooth_frequency_modulation(time, iter, phase):
    """
    Mimics frequency modulation of an oscillator by modulating the time between points of a fixed frequency oscillator.
    The modulation is a single cycle symmetric sawtooth function with an adjustable phase.
    Note: I think this will only work for smooth and small changes to the frequency since the interpolation back to fixed time is linear. I have not tested the limits of this.
    time: time base of data
    iter: modulation depth. Frequency will be modulated by +-iter*w0 
    phase: starting phase of the sawtooth modulation expressed in normalised period. i.e. 0 will start at minimum frequency and go positive, 0.5 will start at max frequency and go negative 
    """

    ht = int(len(time)/2)
    tmod1 = (1-iter+iter*time[:ht]/time[ht])*time[:ht]
    tmod2 = (1+3*iter-iter*(time[ht:])/time[ht])*(time[ht:])-0.2*time[ht]
    time2 = np.concatenate((tmod1, tmod2))
    t_pos = int(phase*len(time))
    if 0 < t_pos < len(time):
        time2 = np.concatenate((time2[t_pos:], time2[-1]+time2[:t_pos]))
        time2 -= time2[0]
    return time2

def generate_sawtooth_frequency_modulation_impulse(time, iter, phase):
    """
    Creates a sawtooth frequency modulations for the impulse response. This directly modulates the frequency in the time domain.
    time: time base of data
    iter: modulation depth. Frequency will be modulated by +-iter*w0 
    phase: starting phase of the sawtooth modulation expressed in normalised period. i.e. 0 will start at minimum frequency and go positive, 0.5 will start at max frequency and go negative 
    """
    ht = int(len(time)/2)
    mod1 = 1-iter+2*iter*time[:ht]/time[ht]
    mod2 = 1+3*iter-2*iter*(time[ht:])/time[ht]
    mod = np.concatenate((mod1, mod2))
    t_pos = int(phase*len(time))
    if 0 < t_pos < len(time):
        mod = np.concatenate((mod[t_pos:], mod[:t_pos]))
        #mod -= mod[0]-0.9
    return mod


def impulse_resp_fm(time, t0, A, y0, yfb, w0, fm):
    """
    Generates impulse response for particle that is frequency modulated
    t0: impulse time in s - recommend doing half way through the trace
    A: response amplitude - in m
    w0: natural frequency of particle
    y0: intrinsic damping of particle (gas and/or laser)
    yfb: Additional damping from cold feedback mechanism
    fm: an array describing the frequency modulation
    """
    output = np.zeros(len(time))
    y1 = (y0+yfb)/2 # factor two required by definition
    w1 = np.sqrt((w0*fm)**2 - y1**2)
    for n, t in enumerate(time):
        if t < t0:
            output[n] = 0
        if t > t0:
            output[n] =  A*np.sin(w1[n]*(t-t0))*np.exp(-y1*(t-t0))
    return output

def generate_displacement_fm(w, w0, y0, yfb, M, T, rnd, rnd2, rnd3, ir, iter, phase):
    """
    Generates the frequency domain response of a particle then reverse fourier transforms to the time domain
    Frequency modulates the oscillator by modulating the timestep
    Adds an impulse response (that must also be frequency modulated) and imprecision noise in the time domain
    Returns displacement
    w: frequency bins for frequency domain response
    w0: natural frequency of particle
    y0: intrinsic damping of particle (gas and/or laser)
    yfb: Additional damping from cold feedback mechanism
    M: mass of oscillator
    T: environment temperature
    rnd and rnd2: Noise for fourier domain - should each have 0 mean and 1/sqrt(2) width I think
    rnd3: imprecision noise - should be 0 mean with width of Snn/2/dtn where Snn of single-sided PSD noise value in m^2/Hz and dtn is timestep in s (1/(2*(max frequency in Hz)))
    ir: impulse response - must have same number of points as 2*len(w)
    iter: modulation depth of frequency modulation
    phase: phase of sawtooth for frequency modulation
    """

    numbins = len(w)
    # Generate frequency response of particle
    thermal_response = transfer_function2(w, w0, y0, yfb, rnd, rnd2, M, T)
    x = np.fft.irfft(thermal_response)[int(numbins/2-1):-int(numbins/2-1)] # Reverse fourier transform to time domain and throw away start and end to get rid of finite time effects
    x = x*np.sqrt(np.trapz(np.abs(thermal_response**2), w/2/np.pi)/np.var(x)) # Scale to correct units
    time = np.linspace(0, np.pi/w[-1]*numbins, numbins)
    time_mod = generate_sawtooth_frequency_modulation(time, iter, phase) # modulate time step intervals
    x = frequency_modulation(x, time, time_mod) # interpolate back to fixed time base
    x += ir # Add impulse response
    x += rnd3 # Add measurement noise

    return x

def save_data_hd5f(filename, data):
    """
    Saves data in HDF5. Does it in a simple way by looping through data and datasetnames
    filename: Filename of file you want to save
    data: the data you want to save as a dictionary
    """
    keys = list(data.keys())
    with h5py.File(filename, "w") as f:
        for m, j in enumerate(data):
            f.create_dataset(keys[m], data=j)
        f.close()
    return 0

def load_data_hd5f(filename):
    """
    Loads data in HDF5. Doesn't load metadata. Outputs as dictionary.
    filename: Filename of file you want to load
    """
    f = h5py.File(filename, "r")
    keys = list(f.keys())
    mdict = {}
    for key in keys:
        mdict[key] = f[key]
    f.close()
    return mdict
    

