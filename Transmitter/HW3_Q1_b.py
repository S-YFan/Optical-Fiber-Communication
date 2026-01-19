import numpy as np
from scipy.constants import k, hbar, m_e, pi
import scipy

def calculate_integral(eta):
    def integral_kernel(x):
        return np.sqrt(x) / (1 + np.exp(x - eta))
    result, error = scipy.integrate.quad(integral_kernel, 0, np.inf)
    return result

def function_for_electron(eta_c):
    N_c = 2.2e24  # cm-3 to m-3 Another 2.2e18 cm-3 to 2.2e24 m-3
    constant_c = 1 / (2 * np.pi ** 2) * (2 * 0.067 * m_e * k * 300 / (hbar ** 2)) ** (3 / 2)
    integral_value = calculate_integral(eta_c)
    return N_c - constant_c * integral_value

def function_for_hole(eta_v):
    N_v = 2.2e24  # Another 2.2e18 cm -3 to 2.2e24 m-3
    constant_v = 1 / (2 * np.pi ** 2) * (2 * 0.48 * m_e * k * 300 / (hbar ** 2)) ** (3 / 2)
    integral_value = calculate_integral(eta_v)
    return N_v - constant_v * integral_value

Efc_c_solution = scipy.optimize.fsolve(function_for_electron, 1)
Efv_v_solution = scipy.optimize.fsolve(function_for_hole, 1)
print(Efc_c_solution*k*300*6.242e18, Efv_v_solution*k*300*6.242e18, 'eV')