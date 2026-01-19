import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, hbar, m_e, pi, electron_volt
import scipy

constant_c = 1 / (2 * np.pi ** 2) * (2 * 0.067 * m_e * k * 300 / (hbar ** 2)) ** (3 / 2)
constant_v = 1 / (2 * np.pi ** 2) * (2 * 0.48 * m_e * k * 300 / (hbar ** 2)) ** (3 / 2)
exp_constanct_c = (0.0263*electron_volt)/(k*300)
exp_constanct_v = (0.0037*electron_volt)/(k*300)

def calculate_integral(eta):
    def integral_kernel(x):
        return np.sqrt(x) / (1 + np.exp(x - eta))
    result, error = scipy.integrate.quad(integral_kernel, 0, np.inf)
    return result

def N_c(eta_c):
    integral_value = constant_c * calculate_integral(eta_c)
    return integral_value

def function_for_hole_root(eta_v_guess, N_c_target):
    N_v_calculated = constant_v * calculate_integral(eta_v_guess)
    return N_v_calculated - N_c_target

def gain(x,y):
    return 1/(1+np.exp(exp_constanct_c-x))-1/(1+np.exp(-exp_constanct_v+y))

eta_c = np.linspace(-5,30,1001)
N_c_list = [1]*len(eta_c)
eta_v_list = [1]*len(eta_c)
gain_list = [1]*len(eta_c)

for index, value in enumerate(eta_c):
    N_c_target = N_c(value)
    N_c_list[index] = N_c_target
    eta_v_solution = scipy.optimize.fsolve(function_for_hole_root, 1, args=(N_c_target,))
    eta_v_list[index] = eta_v_solution[0]
    gain_list[index] = gain(eta_c[index],eta_v_list[index])

N_c_array = np.array(N_c_list)
gain_array = np.array(gain_list)
N_tr_m3 = 0
N_tr_cm3 = 0

fit_indices = np.where((gain_array > -0.05) & (gain_array < 0.05))
x_fit_data = N_c_array[fit_indices]
y_fit_data = gain_array[fit_indices]
if len(x_fit_data) > 2:
    slope, intercept = np.polyfit(x_fit_data, y_fit_data, 1)
    N_tr_m3 = -intercept / slope
    N_tr_cm3 = N_tr_m3 / 1e6  # 轉為 cm^-3
    tangent_line_y = slope * N_c_array + intercept

print(f"--- Transparency Carrier Density (N_tr) at g_p = 0 ---")
print(f"  N_tr = {N_tr_m3:.3e} m^-3")
print(f"  N_tr = {N_tr_cm3:.3e} cm^-3")


plt.figure(figsize=(10, 6))
plt.plot(N_c_array, gain_array, 'b-', label='g_p vs. $N_c$ (Simulation)')  # 原始曲線
if 'tangent_line_y' in locals():
    plt.plot(N_c_array, tangent_line_y, 'r--',
             label=f'Tangent Line')
plt.plot(N_c_array, gain_array, 'b-')
plt.plot(N_tr_m3, 0, 'ko', markersize=8) # 'ko' = Black Circle
plt.title("Gain ($g_p$) vs. Electron Concentration ($N_c$)")
plt.xlabel("Electron Concentration $N_c$ ($m^{-3}$)")
plt.ylabel("Gain $g_p$ (a.u.)")
plt.grid(True)
plt.legend()
plt.ylim(-0.2,0.2)
plt.xlim(1e24,2e24)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7) # 加上 g_p=0 的輔助線
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # 讓 X 軸更易讀
plt.savefig("Nc-g_p_with_Ntr_point.png")
plt.show()