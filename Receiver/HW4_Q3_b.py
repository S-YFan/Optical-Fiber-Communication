import matplotlib.pyplot as plt
import numpy as np
import LFSR
import RaisedCosinePulse as rcp

taps = [1,0,0,1,0,1,0,0,0,0,0,1] #x^12+x^6+x^4+x+1 s1
taps_s2 = [0,1,1,0,0,0,0,0,1,0,0,1] #x^12+x^9+x^3+x^2+1

seed=[1]*12

output_s1 = LFSR.lfsr_generate(taps,seed)
output_s2 = LFSR.lfsr_generate(taps_s2,seed)
output_s1 = np.array(output_s1)
output_s2 = np.array(output_s2)

delay_s_PAM4 = output_s1 + 2 * output_s2

Spectra_PAM4 = np.fft.fft(delay_s_PAM4)
Freq_PAM4 = np.fft.fftfreq(len(Spectra_PAM4))

plt.figure()
plt.plot(Freq_PAM4, 20*np.log10(np.abs(Spectra_PAM4)/np.max(np.abs(Spectra_PAM4))))
plt.xlabel("Arbitary Frequency Axis")
plt.ylabel("Amplitude (dB)")
plt.title("Specturm independent PRBS PAM-4")
plt.show()