import matplotlib.pyplot as plt
import numpy as np
import LFSR
import RaisedCosinePulse as rcp
import scipy.integrate
from scipy.constants import h, c

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

one_over_four = len(output)//4+1
samples_per_bit_200 = 200
upsample_output = np.concatenate([[index_for_upsample] + [0]*(samples_per_bit_200-1) for index_for_upsample in output[0:one_over_four]])
t_NRZ = np.linspace(-2,2,2*samples_per_bit_200+1)
prc03_forNRZ = rcp.rcp(t_NRZ,1,b)
PRBS_25 = np.convolve(upsample_output, prc03_forNRZ,mode='full')
fft_PRBS_25 = np.fft.fft(PRBS_25[samples_per_bit_200+25:-samples_per_bit_200-25])

#Laser相關的參數設定，都轉成m s單位制方便計算
Gamma = 0.3         # confinement factor
sigma_g = 2.5e-20   # differential gain coefficient (m^2)
v_g = 8.57e7        # group velocity (m/s)
alpha_int = 4000.0  # internal absorption coefficient (m^-1)
N_tr = 1.0e24       # transparent carrier density (m^-3)
A_e = 1.0e8         # leakage coefficient (s^-1)
B_e = 1.0e-16       # spontaneous recombination coefficient (m^3/s)
C_e = 3.0e-41       # Auger coefficient (m^6/s)
epsilon = 6.0e-23   # Gain suppression coefficient (m^3)

# 幾何參數與物理常數
L = 250.0e-6        # Length of active region
W = 2.0e-6          #  width of active region
thickness = 0.2e-6  #  thick ness of active region
R_m1 = 0.3          # Mirror 1 reflection [cite: 1318]
R_m2 = 0.98         # Mirror 2 reflection [cite: 1321]
q = 1.602e-19       # Electron charge (C)
V_active = L * W * thickness
alpha_mir1 = (1 / (2 * L)) * np.log(1 / R_m1)
alpha_mir2 = (1 / (2 * L)) * np.log(1 / R_m2)
# 光子生命週期 (s) p.59
tau_p = 1 / (v_g * (alpha_int + alpha_mir1 + alpha_mir2))
# 增益相關常數 (m^3/s) p.61
G = Gamma * v_g * sigma_g
# 閾值載子濃度 (m^-3) p.62
N_th = N_tr + 1 / (G * tau_p)
# 閾值時的載子生命週期 (s) p.53
tau_eth_inv = A_e + B_e * N_th + C_e * N_th**2
tau_eth = 1 / tau_eth_inv
# 閾值電流 (Amps) p.62 設eta_i =1
I_th = q * V_active * N_th / tau_eth
print(I_th)
def S_B(I_B):
    if I_B <= I_th:
        return 0
    #  p. 62 最下式
    S_B = (Gamma * tau_p / (q * V_active)) * (I_B - I_th)
    return S_B

def Constant_numerator_frqeuecny_response(S_B):
    #計算 H(w)分子項
    numerator = (Gamma * v_g * sigma_g * S_B) / (q * V_active)
    denominator = 1 + epsilon * S_B
    return numerator / denominator

def Constant_firstOrder_frqeuecny_response(S_B):
    #計算 H(w) 的 jw 項
    tau_p_inv = 1 / tau_p
    term1 = (v_g * sigma_g * S_B + epsilon * S_B * tau_p_inv) / (1 + epsilon * S_B)
    damping = term1 + tau_eth_inv
    return damping

def Constant_denominator_frqeuecny_response(S_B):
    #計算 H(w) 的分母常數項項
    tau_p_inv = 1 / tau_p
    numerator = tau_p_inv * (v_g * sigma_g * S_B + tau_eth_inv * epsilon * S_B)
    denominator = 1 + epsilon * S_B
    return numerator / denominator

omega_10 = 2 * np.pi * np.fft.fftfreq(len(convolutionNRZ[samples_per_bit:-samples_per_bit]),d=1 / 500e9 )

S_B_val = S_B(2.0 * I_th )
Num = Constant_numerator_frqeuecny_response(S_B_val)
FirstOrder = Constant_firstOrder_frqeuecny_response(S_B_val)
Denom = Constant_denominator_frqeuecny_response(S_B_val)
H_omega_2_10 = Num / (-(omega_10)**2 + 1j*omega_10*FirstOrder + Denom)

S_B_val_3 = S_B(3.0 * I_th )
Num_3 = Constant_numerator_frqeuecny_response(S_B_val_3)
FirstOrder_3 = Constant_firstOrder_frqeuecny_response(S_B_val_3)
Denom_3 = Constant_denominator_frqeuecny_response(S_B_val_3)
H_omega_3_10 = Num_3 / (-(omega_10)**2 + 1j*omega_10*FirstOrder_3 + Denom_3)

Spectrum_2_10 = H_omega_2_10 * fft_NRZ
Spectrum_2_2_5 = H_omega_2_10 * fft_PRBS_25

Spectrum_3_10 = H_omega_3_10 * fft_NRZ
Spectrum_3_2_5 = H_omega_3_10 * fft_PRBS_25

plt.figure()
frequency_hz = omega_10 / (2 * np.pi)
plt.semilogx(frequency_hz/1e9,20*np.log10(abs(H_omega_2_10)/(abs(H_omega_2_10[0]))),'orange')
plt.semilogx(frequency_hz/1e9,20*np.log10(abs(H_omega_3_10)/(abs(H_omega_3_10[0]))),'green')

plt.title("H Frequency Response (Bode Plot)")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(2e-1,4e1)
plt.ylim(-30,30)
plt.grid(True, which="both", ls="--") # "which='both'" 在 log 尺度下很有用

IndirectSignal_2_10 = np.real(np.fft.ifft(Spectrum_2_10))
IndirectSignal_2_2_5 = np.real(np.fft.ifft(Spectrum_2_2_5))
IndirectSignal_3_10 = np.real(np.fft.ifft(Spectrum_3_10))
IndirectSignal_3_2_5 = np.real(np.fft.ifft(Spectrum_3_2_5))

wavelength = 1550e-9  # 1550 nm
photon_energy = (h * c) / wavelength
K_power = v_g * alpha_mir1 * photon_energy * V_active

plt.figure()
plt.plot(IndirectSignal_2_10)
plt.title('in time domain')

eye_NRZ = eye_diagram(IndirectSignal_2_10[samples_per_bit:]*K_power, samples_per_bit, shift=samples_per_bit//2)

plt.figure()
t = np.linspace(-0.1,0.1,len(eye_NRZ[0]))
for eye in eye_NRZ:
    plt.plot(t, eye)
plt.grid(True)
plt.title('2$I_{th}$, 10Gbps')
plt.ylabel('Power (mW)')
plt.xlabel('Time (ns)')
plt.xlim(-0.1,0.1)
plt.ylim(bottom=0)


eye_NRZ = eye_diagram(IndirectSignal_2_2_5[samples_per_bit_200:]*K_power, samples_per_bit_200)

plt.figure()
t = np.linspace(-0.4,0.4,len(eye_NRZ[0]))
for eye in eye_NRZ:
    plt.plot(t, eye)
plt.grid(True)
plt.ylim(bottom=0)
plt.xlim(-0.4,0.4)
plt.ylabel('Power (mW)')
plt.xlabel('Time (ns)')
plt.title('2$I_{th}$, 2.5Gbps')

eye_NRZ = eye_diagram(IndirectSignal_3_10[samples_per_bit:]*K_power, samples_per_bit, shift=samples_per_bit//2)

plt.figure()
t = np.linspace(-0.1,0.1,len(eye_NRZ[0]))
for eye in eye_NRZ:
    plt.plot(t, eye)
plt.grid(True)
plt.ylim(bottom=0)
plt.xlim(-0.1,0.1)
plt.ylabel('Power (mW)')
plt.xlabel('Time (ns)')
plt.title('3$I_{th}$, 10Gbps')

eye_NRZ = eye_diagram(IndirectSignal_3_2_5[samples_per_bit_200:]*K_power, samples_per_bit_200)

plt.figure()
t = np.linspace(-0.4,0.4,len(eye_NRZ[0]))
for eye in eye_NRZ:
    plt.plot(t, eye)
plt.grid(True)
plt.title('3$I_{th}$, 2.5Gbps')
plt.ylabel('Power (mW)')
plt.xlabel('Time (ns)')
plt.xlim(-0.4,0.4)
plt.ylim(bottom=0)
plt.show()


alpha_H = 3.0
kappa = epsilon / tau_p
d=2e-12 # (1/R_b)/sampler_per_bit

def delta_nu(Input, d, alpha_H, kappa, limit_Hz=50e9):
    dS_dt = np.gradient(Input, d)
    Input = Input + np.abs(np.min(Input))
    S_t_safe = Input + 1e-20
    term_transient = (1.0 / S_t_safe) * dS_dt
    term_adiabatic = kappa * Input
    Delta_nu = (alpha_H / (4 * np.pi)) * (term_transient + term_adiabatic)
    Delta_nu = np.clip(Delta_nu, -limit_Hz, limit_Hz)
    mask_too_large = np.abs(Delta_nu) > limit_Hz
    Delta_nu[mask_too_large] = 0


    phi_t = 2 * np.pi * scipy.integrate.cumulative_trapezoid(Delta_nu, dx=d, initial=0)
    print(phi_t)
    E_chirped = np.sqrt(S_t_safe) * np.exp(1j * phi_t)
    Spec_NoChirp = np.fft.fft(np.sqrt(S_t_safe))
    Spec_WithChirp = np.fft.fft(E_chirped)
    return Spec_NoChirp, Spec_WithChirp

Spec_NoChirp, Spec_WithChirp = delta_nu(IndirectSignal_2_10, d, alpha_H, kappa)

freq_axis = np.fft.fftfreq(len(IndirectSignal_2_10), d=d)
plt.figure(figsize=(10, 6))
plt.plot(freq_axis/1e9, 20 * np.log10(np.abs(Spec_WithChirp) / np.max(np.abs(Spec_WithChirp)) + 1e-20), color='orange', linestyle='--', label='2$I_{th}$')

Spec_NoChirp, Spec_WithChirp = delta_nu(IndirectSignal_3_10, d, alpha_H, kappa)

freq_axis = np.fft.fftfreq(len(IndirectSignal_3_10), d=d)
plt.plot(freq_axis/1e9, 20 * np.log10(np.abs(Spec_WithChirp) / np.max(np.abs(Spec_WithChirp)) + 1e-20), 'b--', label='3$I_{th}$')

plt.title("Impact of Chirp on Optical Spectrum 10GHz with 2 and 3$I_{th}$")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Normalized Power (dB)")
plt.legend()
plt.grid(True)

plt.xlim(-100,100)
plt.ylim(-80,0)

Spec_NoChirp, Spec_WithChirp = delta_nu(IndirectSignal_2_2_5, d, alpha_H, kappa)

freq_axis = np.fft.fftfreq(len(IndirectSignal_2_2_5), d=d)
plt.figure(figsize=(10, 6))
plt.plot(freq_axis/1e9, 20 * np.log10(np.abs(Spec_WithChirp) / np.max(np.abs(Spec_WithChirp)) + 1e-20), color='orange', linestyle='--', label='2$I_{th}$')

Spec_NoChirp, Spec_WithChirp = delta_nu(IndirectSignal_3_2_5, d, alpha_H, kappa)

freq_axis = np.fft.fftfreq(len(IndirectSignal_3_2_5), d=d)
plt.plot(freq_axis/1e9, 20 * np.log10(np.abs(Spec_WithChirp) / np.max(np.abs(Spec_WithChirp)) + 1e-20), 'b--', label='3$I_{th}$')

plt.title("Impact of Chirp on Optical Spectrum 2.5GHz with 2 and 3$I_{th}$")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Normalized Power (dB)")
plt.legend()
plt.grid(True)

plt.xlim(-100,100)
plt.ylim(-80,0)

plt.show()

'''
# Chirp 在時域的樣子

    plt.figure(figsize=(10, 4))
    plt.plot(Delta_nu / 1e9)
    plt.title("Chirp in Time Domain")
    plt.ylabel("Frequency Shift (GHz)")
    plt.xlabel("Time Index")
    plt.grid(True)
    plt.show()
'''