'''
Created on May 29, 2024

@author: F.Knorr
'''

from scipy.constants import c as c0
from string import ascii_letters
from random import seed, choice, randint
import numpy as np
import pylab as pl
import plotly

'''
Initializing the random number generator
fixed seed for reproducible results
'''     
LETTERS = ascii_letters
RNDSEED = 'b332d15a-7368-3da1-8b1e-7283541146da'
seed(a=RNDSEED, version=2)


class Signal(np.ndarray):
    '''
    Simple signal model
    '''
    def __new__(cls, array, *args, **kwargs):
        obj = np.asarray(array).view(cls)
        return obj
    
    def convolve(self, convolution):
        '''
        convolve current signal with convolution function
        '''
        return Signal(np.convolve(self, convolution, mode='same'))


class Component:   
    '''
    abstract component class
    base class for all optical components
    '''
    def input(self, signal:Signal):
        self.input = signal
        self.output = self.modify_signal()
        
    def set_filter(self, filter):
        self.filter = filter
        
    def modify_signal(self, *args, **kwargs):
        '''
        abstract class of signal modification
        by default it's the unity operator
        '''
        return 1*self.input

               
class Transmitter(Component):
    '''
    Encoding a random letter
    into binary function
    Applying gaussian filter function on top
    '''
    def __init__(self, carrier_wavelength=940e-9, 
                 time_window_fs=100, 
                 resolution=994):
               
        '''
        sinusoidal carrier signal
        using f = c/wl @ 940 nm wavelength as default
        cram at least 8 periods (for 8 bit) into the window
        --> about 20 fs window 
        Choosing a decent resolution of 1k sample points
        :param carrier_wavelength
        :param time_window
        :param resolution
        add a default gaussian noise
        '''
        self.f = c0 / carrier_wavelength
        self.window = Signal(np.linspace(0, time_window_fs * 1e-15, resolution))
        self.carrier = np.cos(2*np.pi*self.f*self.window)
        self.filter = gaussian_filter(0, 0.1)(np.linspace(-1, 1,len(self.window)))
   
    def set_filter(self, filter):
        '''
        set the filter / noise function
        '''
        self.filter = filter(np.linspace(-1, 1,len(self.window)))
    
    def get_random_signal(self):
        '''
        Choose a random letter from the 128 bit ASCII alphabet
        (a-z and A-Z), 2*26 letters in total
        Altenatively: Create a random int in [0, 127] and pick from array
        
        convolve with a gaussian filter 
        
        '''
        rand_number = randint(0,2*26-1)
        self.letter = str(LETTERS)[rand_number]
        
        # # convenient alternative:
        # self.letter = choice(LETTERS)
        
        self.encoding = self.imprint_binary_waveform(self.letter, self.carrier)
        self.encoding = self.encoding.convolve(self.filter)
        signal = self.carrier * self.encoding
        self.output = signal
        return signal
    
    def imprint_binary_waveform(self, literal, signal):
        '''
        Convert a 7-bit ascii-string to
        rect-function in range [-1 , 1]
        for an array with same no of sampling
        points as reference array 'signal'
        :param s: 7-bit string ascii-literal
        :param signal: reference signal window
        '''
        if len(literal) > 1:
            literal = literal[0]
        binary_repr = '{0:07b}'.format(ord(literal))      
        self.binary_repr = binary_repr
        
        encoding = np.zeros(len(signal))
        patch_length = int(len(signal) / len(binary_repr))
        
        for idx, bit in enumerate(binary_repr):
            encoding[idx*patch_length:(idx+1)*patch_length] = -(-1)**int(bit)
        
        return Signal(encoding)


class Fiber(Component): 
    '''
    Simple representation of an optical fiber 
    '''  
    
    def __init__(self, *args, **kwargs):
        '''
        Set fiber losses:
        :param l_fiber: length of the fiber
        :param a_att: attenuation per meter fiber length: 
        dissipative loss
        :param scattering_att: Attenuation due to scattering. Typical
        inomogeneities should be < 0.1*wl so Rayleigh-regime can be assumed.
        Simplified forward scattered intensity is about (wl/l_fiber)^2 * 1e-7,
        hence it could be assumed that the loss is about 
        d_particle/d_fiber * n_particles. 
        :param inc_att: attenuation of the signal due to incoupling loss
        incoupling loss: mainly imperfect alignment between coupling mates
        :param inc_att: attenuation of the signal due to outcoupling loss
        outcoupling loss: mainly imperfect alignment between coupling mates
        :param bending_att: Losses introduced by fiber bending if local surface
        tilt exceeds critical angle arcsin(n_clad /n_core) 
        
        additional potential loss channels:
        chirp losses: Not relevant because model uses single-wavelength only.
        important for e.g. short pulse packages --> go for hollow core fiber 
        Modal dispersion: Assume using a perfect SM-fiber here
        
        
        
        '''
        self.l_fiber = kwargs.get("fiber_length")
        self.a_att = kwargs.get("attenuation_per_meter", 1)
        self.inc_att = kwargs.get("incoupling_loss", 1)
        self.outc_att = kwargs.get("outcoupling_loss", 1)   
        self.bending_att = kwargs.get("bending_att", 1)             
    
    def modify_signal(self, *args, **kwargs):
        signal = self.input * (1-self.inc_att) 
        signal = signal * self.l_fiber * (1-self.a_att)
        signal = signal * (1-self.outc_att)
        return signal


class Receiver(Component):
    '''
    Models a simple direct receiver
    It captures amplitude only
    In coherent capture (heterodyne) one could additionally
    evaluate phase information. For that we need basically
    two channels with known/calibrated path difference and 
    loss characteristics or two calibrated reference carrier 
    signal generators
    
    carrier1 -----BPSK-----LOSSES----/---interference detector
                                     |
                                     |
                                     |
                                 carrier2
                                 

    
    '''
    def __init__(self, *args, **kwargs):
        self.wl = kwargs.get("wl", 940e-9)
        self.qe = kwargs.get("qe", 1)
        
        self.f = c0 / self.wl
        
    def _demodulate(self, signal):
        '''
        get length of the signal
        create an appropriate carrier frequency
        multiply with said frequency to extract AM
        '''
        resolution = len(signal)
        
        carrier = Signal(np.linspace(0, 100 * 1e-15, resolution))
        carrier = np.cos(2*np.pi*self.f*carrier)
        
        return signal*carrier
    
    def get_binary(self):
        return ord(self.output)
            
    def modify_signal(self, *args, **kwargs):
        signal = self.input * self.qe
        signal = self._demodulate(signal)
        self.output = signal
        return signal
    
    def get_message(self):
        fft_signal = np.fft.fft(self.output)
        # low pass filter: cut off frequencies of carrier and higher
        fft_signal = fft_signal[:2**6]
        signal = np.fft.ifft(fft_signal)
        # lazy binarization
        signal = np.where(signal > 0, 1, 0)
        self.fft_filtered = signal
        result = [signal[int(int(2**6 / 7)*(i+0.5))] for i in range(7)]
        result_str = ""
        for i in result:
            result_str = result_str + str(i) 
        result = int(result_str, 2)

 
        return chr(result)
        

def gaussian_filter(center, fwhm):
    '''
    simple gaussian filter / noise function
    '''
    sigma = fwhm / (2 * np.sqrt(2*np.log(2)))
    filter = lambda arr: 1 / \
        (np.sqrt(2*np.pi) * sigma) * \
        np.exp(- np.power(arr-center, 2) / (np.sqrt(2)*np.power(sigma, 2)))
    return filter


def compare_bitwise(a,b):
    matches = 0
    for idx, i in enumerate(a):
        if i == b[idx]:
            matches += 1
    return matches
        
        

'''
Create transmitter object, taking up the input signal
Try a couple of times with different losses
'''
sent = []
received = []
matches = []

for i in range(15):
    i += 1 # skip the zero

    t = Transmitter()
    t.set_filter(gaussian_filter(0, (0.3*i)*0.2))
    
    f = Fiber(fiber_length=10,
              attenuation_per_meter = i*12e-9,
              incoupling_loss=i*10e-4,
              outcoupling_loss=i*10e-4
              )
    r = Receiver(qe=0.99**i)
    
    
    signal = t.get_random_signal()
    print (f"signal sent is '{t.letter}'")
    
    f.input(signal)
    signal = f.output
    
    r.input(signal)
    signal_received = r.output

    '''
    store sent, received and count no of matching bits
    '''   
    sent.append(t.binary_repr)
    received.append('{0:07b}'.format(ord(r.get_message())))
    matches.append(compare_bitwise(sent[-1], received[-1]))
   
    print (f"signal received is: '{r.get_message()}'")

print(sent)
print(received)
print(matches)
pl.figure(figsize=(12,8))
pl.subplot(131)
pl.title("last signal sent")
pl.plot(t.window, t.encoding)
pl.subplot(132)
pl.title("last signal received")
pl.plot(signal_received)
pl.subplot(133)
pl.title("bits correctly received")
pl.scatter(np.arange(15), matches)
pl.grid()
pl.tight_layout()
pl.show()

print ("--- done ---")
