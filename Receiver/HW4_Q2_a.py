import matplotlib.pyplot as plt
import numpy as np
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

taps = [1,0,0,1,0,1,0,0,0,0,0,1] #x^12+x^6+x^4+x+1
seed=[1]*12

output = LFSR.lfsr_generate(taps,seed)
samples_per_bit =50

upsample_output = np.concatenate([[index_for_upsample] + [0]*(samples_per_bit-1) for index_for_upsample in output])

b=0.5
t_NRZ = np.linspace(-2,2,2*samples_per_bit+1)
prc03_forNRZ = rcp.rcp(t_NRZ,1,b)
convolutionNRZ = np.convolve(upsample_output, prc03_forNRZ,mode='full')
fft_NRZ = np.fft.fft(convolutionNRZ[samples_per_bit:-samples_per_bit])

tau = 75 * 1e-12
omega = 2 * np.pi *np.fft.fftfreq(len(convolutionNRZ[samples_per_bit:-samples_per_bit]),d=1/(10*1e9*samples_per_bit))
H_75 = 1/(1+1j*omega*tau)
affect_convolutionNRZ_FT_75 = H_75 * fft_NRZ

tau = 150 * 1e-12
omega = 2 * np.pi * np.fft.fftfreq(len(convolutionNRZ[samples_per_bit:-samples_per_bit]),d=1/(10*1e9*samples_per_bit)) #2 * np.pi 不用加
H_150 = 1/(1+1j*omega*tau)
affect_convolutionNRZ_FT_150 = H_150 * fft_NRZ

affect_convolutionNRZ_75 = np.abs(np.fft.ifft(affect_convolutionNRZ_FT_75))
affect_convolutionNRZ_150 = np.abs(np.fft.ifft(affect_convolutionNRZ_FT_150))

eye_NRZ = eye_diagram(affect_convolutionNRZ_75[samples_per_bit:], samples_per_bit, shift=samples_per_bit//4+10)
plt.figure(figsize=(10,7))
t = np.linspace(-100,100,len(eye_NRZ[0]))
for eye in eye_NRZ:
    plt.plot(t, eye)
plt.grid(True)
plt.xlim(-100,100)
plt.xlabel("Time (ps)")
plt.ylabel("Ampltiude")
plt.ylim(bottom=0)
plt.title(r"10 GHz with $\tau_{RC}=75$ ps")

eye_NRZ = eye_diagram(affect_convolutionNRZ_150[samples_per_bit:], samples_per_bit, shift=samples_per_bit//4+10)
plt.figure(figsize=(10,7))
t = np.linspace(-100,100,len(eye_NRZ[0]))
for eye in eye_NRZ:
    plt.plot(t, eye)
plt.grid(True)
plt.xlim(-100,100)
plt.xlabel("Time (ps)")
plt.ylabel("Ampltiude")
plt.ylim(bottom=0)
plt.title(r"10 GHz with $\tau_{RC}=150$ ps")
plt.show()
