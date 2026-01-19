import matplotlib.pyplot as plt
import numpy as np
import LFSR
import RaisedCosinePulse as rcp

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
freq_NRZ = np.fft.fftfreq(len(convolutionNRZ[samples_per_bit:-samples_per_bit]))

plt.figure()
plt.plot(freq_NRZ,20*np.log10(np.abs(fft_NRZ)/np.abs(np.max(fft_NRZ))))



E_0 = 1
V_pi = 1  #這個factor影響了除以convolution NRZ的結果，因此再想想，決定這個參數的原因有哪些。

eye1 = eye_diagram(convolutionNRZ, samples_per_bit)
plt.figure()
t = np.linspace(-1,1,2*samples_per_bit)
for eye_i in eye1:
    plt.plot(t, eye_i)
plt.title("origin")

E_output = E_0*(np.cos((convolutionNRZ[samples_per_bit:-samples_per_bit]/V_pi)*np.pi/2+np.pi/2)) #我透過+pi/2解決掉0變成1, 1變成0的問題



eye1 = eye_diagram(np.abs(E_output**2), samples_per_bit)
plt.figure()
t = np.linspace(-1,1,2*samples_per_bit)
for eye_i in eye1:
    plt.plot(t, eye_i)
plt.title("Modulated eye diagram")

#上面第a題
#下面第b題

T_b = 1
omega_cl = 2 * np.pi / T_b
t_carver = np.linspace(0,4095,len(output)*samples_per_bit)
A_50_RZ = np.cos((np.pi*np.cos(omega_cl*t_carver+np.pi))/4+np.pi/4) * E_output
A_33_RZ = np.cos(np.pi*np.cos((omega_cl*t_carver+np.pi )/2)/2) * E_output
A_67_RZ = np.cos(np.pi*np.cos(omega_cl*t_carver/2)/2+np.pi/2) * E_output

plt.figure()
A_RZ_Spectrum = np.fft.fft(A_50_RZ)
A_RZ_freq = np.fft.fftfreq(len(A_RZ_Spectrum), d=1 / 10e9)
fft_NRZ = np.fft.fft(E_output)
plt.plot(A_RZ_freq/1e9,20*np.log10(np.abs(A_RZ_Spectrum)/np.max(np.abs(A_RZ_Spectrum))), 'orange')
plt.plot(A_RZ_freq/1e9,20*np.log10(np.abs(fft_NRZ)/np.max(np.abs(fft_NRZ))), 'blue')
plt.title("50RZ Spectrum")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.ylim(-90,0)
plt.xlim(-1,1)
plt.legend(['50%RZ','NRZ'])
plt.grid(True)

plt.figure()
A_RZ_Spectrum = np.fft.fft(A_33_RZ)
A_RZ_freq = np.fft.fftfreq(len(A_RZ_Spectrum), d=1 / 10e9)
plt.plot(A_RZ_freq/1e9,20*np.log10(np.abs(A_RZ_Spectrum)/np.max(np.abs(A_RZ_Spectrum))), 'green')
plt.plot(A_RZ_freq/1e9,20*np.log10(np.abs(fft_NRZ)/np.max(np.abs(fft_NRZ))), 'blue')
plt.title("33%RZ Spectrum")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.ylim(-90,0)
plt.xlim(-1,1)
plt.legend(['33%RZ','NRZ'])
plt.grid(True)

plt.figure()
A_RZ_Spectrum = np.fft.fft(A_67_RZ)
A_RZ_freq = np.fft.fftfreq(len(A_RZ_Spectrum), d=1 / 10e9)
plt.plot(A_RZ_freq/1e9,20*np.log10(np.abs(A_RZ_Spectrum)/np.max(np.abs(A_RZ_Spectrum))), 'red')
plt.plot(A_RZ_freq/1e9,20*np.log10(np.abs(fft_NRZ)/np.max(np.abs(fft_NRZ))), 'blue')
plt.title("67%RZ Spectrum")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.ylim(-90,0)
plt.xlim(-1,1)
plt.grid(True)
plt.legend(['67%RZ','NRZ'])
plt.show()

eye1 = eye_diagram(np.abs(A_50_RZ**2), samples_per_bit,shift=samples_per_bit//2)
plt.figure()
t = np.linspace(-0.1,0.1, 2*samples_per_bit)
for eye_i in eye1:
    plt.plot(t, eye_i)
plt.title("x-cut Modulated eye diagram, 50%RZ")
plt.xlabel("Time(ns)")
plt.xlim(-0.1,0.1)
plt.ylabel("Power")
plt.grid(True)

eye1 = eye_diagram(np.abs(A_33_RZ**2), samples_per_bit,shift=samples_per_bit//2)
plt.figure()
t = np.linspace(-0.1,0.1, 2*samples_per_bit)
for eye_i in eye1:
    plt.plot(t, eye_i)
plt.title("x-cut Modulated eye diagram, 33%RZ")
plt.xlabel("Time(ns)")
plt.xlim(-0.1,0.1)
plt.ylabel("Power")
plt.grid(True)

eye1 = eye_diagram(np.abs(A_67_RZ**2), samples_per_bit,shift=samples_per_bit//2)
plt.figure()
t = np.linspace(-0.1,0.1, 2*samples_per_bit)
for eye_i in eye1:
    plt.plot(t, eye_i)
plt.title("x-cut Modulated eye diagram, 67%RZ")
plt.xlabel("Time(ns)")
plt.xlim(-0.1,0.1)
plt.ylabel("Power")
plt.grid(True)
plt.show()
