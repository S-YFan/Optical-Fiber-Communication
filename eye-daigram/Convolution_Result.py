import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import Output
import LFSR
import RaisedCosinePulse as rcp

# This is a completed version from how to upsampled the PRBS and utilizing Pulse to shaping the signal.
# I provide time domain, spectrum, eye-diagram graph in this code. 2025 10 31

def eye_diagram(signal, sps, span = 2, shift=0):
    """
    signal : circular convolution
    sps    : samples per symbol
    span   : how many symbol we will take (we choose 2 in general)

    We slice the every Bit Pulse in the list, and we can get a list, which is include all slice about the origin signal.
    """
    n = len(signal) // sps
    eye = []
    for i in range(n-span):
        seg = signal[(i)*sps+shift : (i+span)*sps+shift]
        eye.append(seg)
    return np.array(eye)

def p_sg(t,T,m):
    return np.exp(-np.log(2)*(t/T)**(2*m))
#注意到FWHM要正確

taps = [1,0,0,1,0,1,0,0,0,0,0,1] #x^12+x^6+x^4+x+1
seed=[1]*12
output = LFSR.lfsr_generate(taps,seed)
samples_per_bit =50

raised_bit =1
upsample_output = np.concatenate([[index_for_upsample] + [0]*(samples_per_bit-1) for index_for_upsample in output])

b=0.3
t = np.linspace(-1,1,samples_per_bit+1)
prc03_for50RZ = rcp.rcp(t,0.5,b)
t_NRZ = np.linspace(-2,2,2*samples_per_bit+1)
prc03_forNRZ = rcp.rcp(t_NRZ,1,b)

convolution50RZ = np.convolve(upsample_output, prc03_for50RZ,mode='full')
convolutionNRZ = np.convolve(upsample_output, prc03_forNRZ,mode='full')

n_axis = np.linspace(0,4095,len(convolution50RZ))
plt.figure()
plt.plot(n_axis, convolution50RZ,'orange')
n_axis = np.linspace(0,4095,len(convolutionNRZ[samples_per_bit//2:-samples_per_bit//2]))

plt.plot(n_axis, convolutionNRZ[samples_per_bit//2:-samples_per_bit//2],'blue')


ticks = np.linspace(0, 20, 21)
plt.xticks(ticks)
ax = plt.gca()
labels = [str(int(t)) if t in np.linspace(0,20,5) else "" for t in ticks]
ax.set_xticklabels(labels)
plt.xlim(0,20)

plt.grid(True)

plt.figure()
fft_50RZ = np.fft.fft(convolution50RZ[samples_per_bit//2:-samples_per_bit//2])
freq_50RZ = np.fft.fftfreq(len(convolution50RZ[samples_per_bit//2:-samples_per_bit//2]))
fft_NRZ = np.fft.fft(convolutionNRZ[samples_per_bit:-samples_per_bit])
freq_NRZ = np.fft.fftfreq(len(convolutionNRZ[samples_per_bit:-samples_per_bit]))
plt.plot(freq_50RZ, 20*np.log10(abs(fft_50RZ)/np.max(abs(fft_50RZ))),'orange')
plt.plot(freq_NRZ, 20*np.log10(abs(fft_NRZ)/np.max(abs(fft_NRZ))),'blue')
plt.ylim(-80,0)
plt.xlim(-0.05,0.05)

eye_NRZ = eye_diagram(convolutionNRZ, samples_per_bit)

plt.figure()
t = np.linspace(-1,1,len(eye_NRZ[0]))
for eye in eye_NRZ:
    plt.plot(t, eye, 'b')
plt.grid(True)

eye_50RZ = eye_diagram(convolution50RZ, samples_per_bit)

plt.figure()
t = np.linspace(-1,1,len(eye_50RZ[0]))
for eye in eye_50RZ:
    plt.plot(t, eye, 'orange')
plt.grid(True)

plt.show()
