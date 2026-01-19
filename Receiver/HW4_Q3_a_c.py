import matplotlib.pyplot as plt
import numpy as np
import LFSR
import RaisedCosinePulse as rcp


def eye_diagram(signal, sps, span = 2, shift=0):
    """
    sps(spb)    : samples per symbol(bit)
    span   : how many symbol we will take (we choose 2 in general)
    shift: Please directly slice it
    We slice the every Bit Pulse in the list, and we can get a list, which is include all slice about the origin signal.
    """
    n = len(signal) // sps
    eye = []
    for i in range(n-span):
        seg = signal[(i)*sps+shift : (i+span)*sps+shift]
        eye.append(seg)
    return np.array(eye)



taps = [1,0,0,1,0,1,0,0,0,0,0,1] #x^12+x^6+x^4+x+1 s1
seed=[1]*12

output_s1 = LFSR.lfsr_generate(taps,seed)
delay_s_PAM4_1 = output_s1 + 2 * np.pad(output_s1, (1, 0), mode='constant')[:-1]
delay_s_PAM4_7 = output_s1 + 2 * np.pad(output_s1, (7, 0), mode='constant')[:-7]
delay_s_PAM4_13 = output_s1 + 2 * np.pad(output_s1, (13, 0), mode='constant')[:-13]

Spectra_PAM4_1 = np.fft.fft(delay_s_PAM4_1)
Spectra_PAM4_7 = np.fft.fft(delay_s_PAM4_7)
Spectra_PAM4_13 = np.fft.fft(delay_s_PAM4_13)
Freq_PAM4 = np.fft.fftfreq(len(Spectra_PAM4_1))

#a

plt.figure(figsize=(10,7))
plt.plot(Freq_PAM4, 20*np.log10(np.abs(Spectra_PAM4_1)/np.max(np.abs(Spectra_PAM4_1))))
plt.ylabel("Amplitude (dB)")
plt.xlabel("Arbitary Frequency Axis")
plt.title(r"Spectrum delay 1 bit")
plt.figure(figsize=(10,7))
plt.plot(Freq_PAM4, 20*np.log10(np.abs(Spectra_PAM4_7)/np.max(np.abs(Spectra_PAM4_7))))
plt.ylabel("Amplitude (dB)")
plt.xlabel("Arbitary Frequency Axis")
plt.title(r"Spectrum delay 7 bit")
plt.figure(figsize=(10,7))
plt.plot(Freq_PAM4, 20*np.log10(np.abs(Spectra_PAM4_13)/np.max(np.abs(Spectra_PAM4_13))))
plt.ylabel("Amplitude (dB)")
plt.xlabel("Arbitary Frequency Axis")
plt.title(r"Spectrum delay 13 bit")
plt.show()

# c

#這是先Rasied Cosine Pulse再去做delay

samples_per_bit = 200
b=0.3
t_NRZ = np.linspace(-2,2,2*samples_per_bit+1)
prc03_forNRZ = rcp.rcp(t_NRZ,1,b)
upsample_output_s1 = np.concatenate([[index_for_upsample] + [0]*(samples_per_bit-1) for index_for_upsample in output_s1])
convolutionNRZ_s1 = np.convolve(upsample_output_s1, prc03_forNRZ,mode='full')

convolutionNRZ_delay1 = convolutionNRZ_s1 + 2 * np.pad(convolutionNRZ_s1, (1*samples_per_bit, 0), mode='constant')[:-1*samples_per_bit]
convolutionNRZ_delay7 = convolutionNRZ_s1 + 2 * np.pad(convolutionNRZ_s1, (7*samples_per_bit, 0), mode='constant')[:-7*samples_per_bit]
convolutionNRZ_delay13 = convolutionNRZ_s1 + 2 * np.pad(convolutionNRZ_s1, (13*samples_per_bit, 0), mode='constant')[:-13*samples_per_bit]


Spectra_PAM4_1_RCP = np.fft.fft(convolutionNRZ_delay1[samples_per_bit:-samples_per_bit])
Spectra_PAM4_7_RCP = np.fft.fft(convolutionNRZ_delay7[samples_per_bit:-samples_per_bit])
Spectra_PAM4_13_RCP = np.fft.fft(convolutionNRZ_delay13[samples_per_bit:-samples_per_bit])
GHz20 = np.fft.fftfreq(len(convolutionNRZ_delay1[samples_per_bit:-samples_per_bit]), d = 1/(20*1e9*samples_per_bit))

eye_delay1 = eye_diagram(convolutionNRZ_delay1,samples_per_bit)


plt.figure(figsize=(10,7))
t = np.linspace(-50,50,len(eye_delay1[0]))
for eye in eye_delay1:
    plt.plot(t, eye)
plt.grid(True)
plt.xlim(-50,50)
plt.xlabel("Time (ps)")
plt.ylabel("Ampltiude")
plt.ylim(bottom=0)
plt.title(r"20 GHz delay 1 bit")

eye_delay7 = eye_diagram(convolutionNRZ_delay7,samples_per_bit)
plt.figure(figsize=(10,7))
t = np.linspace(-50,50,len(eye_delay7[0]))
for eye in eye_delay7:
    plt.plot(t, eye)
plt.grid(True)
plt.xlim(-50,50)
plt.xlabel("Time (ps)")
plt.ylabel("Ampltiude")
plt.ylim(bottom=0)
plt.title(r"20 GHz delay 7 bit")



eye_delay13 = eye_diagram(convolutionNRZ_delay13,samples_per_bit)
plt.figure(figsize=(10,7))
t = np.linspace(-50,50,len(eye_delay13[0]))
for eye in eye_delay13:
    plt.plot(t, eye)
plt.grid(True)
plt.xlim(-50,50)
plt.xlabel("Time (ps)")
plt.ylabel("Ampltiude")
plt.ylim(bottom=0)
plt.title(r"20 GHz delay 13 bit")
plt.show()


plt.figure(figsize=(10,7))
plt.plot(GHz20 / 1e9, 20 * np.log10((np.abs(Spectra_PAM4_1_RCP)) / np.max(np.abs(Spectra_PAM4_1_RCP))), color='r')
plt.xlim(-40,40)
plt.xlabel("GHz")
plt.ylabel("Ampltiude (dB)")
plt.title(r"Spectrum 20 GHz delay 1 bit")
plt.ylim(-100,0)
plt.figure(figsize=(10,7))
plt.plot(GHz20 / 1e9, 20 * np.log10((np.abs(Spectra_PAM4_7_RCP)) / np.max(np.abs(Spectra_PAM4_7_RCP))), color='g')
plt.xlim(-40,40)
plt.xlabel("GHz")
plt.ylabel("Ampltiude (dB)")
plt.ylim(-100,0)
plt.title(r"Spectrum 20 GHz delay 7 bit")
plt.figure(figsize=(10,7))
plt.plot(GHz20 / 1e9, 20 * np.log10((np.abs(Spectra_PAM4_13_RCP)) / np.max(np.abs(Spectra_PAM4_13_RCP))), color='b')
plt.xlim(-40,40)
plt.xlabel("GHz")
plt.ylabel("Ampltiude (dB)")
plt.title(r"Spectrum 20 GHz delay 13 bit")
plt.ylim(-100,0)
plt.show()


"""
#這是延伸a去做，先delay再Raised Cosine Pulse，以眼圖來看這個統計結果不太對，我推測是和Raised Cosine Pulse的設定有關，它們convolve的結果會被影響。

samples_per_bit =50
upsample_output_delay1 = np.concatenate([[index_for_upsample] + [0]*(samples_per_bit-1) for index_for_upsample in delay_s_PAM4_1])
upsample_output_delay7 = np.concatenate([[index_for_upsample] + [0]*(samples_per_bit-1) for index_for_upsample in delay_s_PAM4_7])
upsample_output_delay13 = np.concatenate([[index_for_upsample] + [0]*(samples_per_bit-1) for index_for_upsample in delay_s_PAM4_13])

b=0.3
t_NRZ = np.linspace(-2,2,2*samples_per_bit+1)
prc03_forNRZ = rcp.rcp(t_NRZ,1,b)
convolutionNRZ_delay1 = np.convolve(upsample_output_delay1, prc03_forNRZ,mode='full')
convolutionNRZ_delay7 = np.convolve(upsample_output_delay7, prc03_forNRZ,mode='full')
convolutionNRZ_delay13 = np.convolve(upsample_output_delay13, prc03_forNRZ,mode='full')

Spectra_PAM4_1_RCP = np.fft.fft(convolutionNRZ_delay1[samples_per_bit:-samples_per_bit])
GHz20 = np.fft.fftfreq(len(convolutionNRZ_delay1[samples_per_bit:-samples_per_bit]), d = 1/(20*1e9))
Spectra_PAM4_7_RCP = np.fft.fft(convolutionNRZ_delay7[samples_per_bit:-samples_per_bit])
Spectra_PAM4_13_RCP = np.fft.fft(convolutionNRZ_delay13[samples_per_bit:-samples_per_bit])

plt.figure()
plt.plot(GHz20 / 1e9, 20 * np.log10((np.abs(Spectra_PAM4_1_RCP)) / np.max(np.abs(Spectra_PAM4_1_RCP))), color='r')
plt.xlim(-2.5,2.5)
plt.figure()
plt.plot(GHz20 / 1e9, 20 * np.log10((np.abs(Spectra_PAM4_7_RCP)) / np.max(np.abs(Spectra_PAM4_7_RCP))), color='g')
plt.xlim(-2.5,2.5)
plt.figure()
plt.plot(GHz20 / 1e9, 20 * np.log10((np.abs(Spectra_PAM4_13_RCP)) / np.max(np.abs(Spectra_PAM4_13_RCP))), color='b')
plt.xlim(-2.5,2.5)



eye_delay1 = eye_diagram(convolutionNRZ_delay1,samples_per_bit)
plt.figure(figsize=(10,7))
t = np.linspace(-50,50,len(eye_delay1[0]))
for eye in eye_delay1:
    plt.plot(t, eye)
plt.grid(True)
plt.xlim(-50,50)
plt.xlabel("Time (ps)")
plt.ylabel("Ampltiude")
plt.ylim(bottom=0)
plt.title(r"20 GHz delay 1 bit")

eye_delay7 = eye_diagram(convolutionNRZ_delay7,samples_per_bit)
plt.figure(figsize=(10,7))
t = np.linspace(-50,50,len(eye_delay1[0]))
for eye in eye_delay1:
    plt.plot(t, eye)
plt.grid(True)
plt.xlim(-50,50)
plt.xlabel("Time (ps)")
plt.ylabel("Ampltiude")
plt.ylim(bottom=0)
plt.title(r"20 GHz delay 7 bit")

eye_delay13 = eye_diagram(convolutionNRZ_delay13,samples_per_bit)
plt.figure(figsize=(10,7))
t = np.linspace(-50,50,len(eye_delay1[0]))
for eye in eye_delay1:
    plt.plot(t, eye)
plt.grid(True)
plt.xlim(-50,50)
plt.xlabel("Time (ps)")
plt.ylabel("Ampltiude")
plt.ylim(bottom=0)
plt.title(r"20 GHz delay 13 bit")
plt.show()
"""