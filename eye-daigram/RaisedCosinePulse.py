import numpy as np
import matplotlib.pyplot as plt

def rcp(t, T, b):
    # notice that T is half width, so it is DIFFERENT from the slide in chap. 1 ('FWHM is 2*T')
    FWHM = 2*T
    rcp_output = []
    for i in range(len(t)):
        if np.abs(t[i]) < (1-b)*T:
            rcp_output.append(1)
        elif np.abs(t[i]) >= (1+b)*T:
            rcp_output.append(0)
        elif  (1+b)*T > np.abs(t[i]) >= (1-b)*T:
            rcp_output.append((0.5) * (1 + np.cos((np.pi * (np.abs(t[i]) - (1 - b) * FWHM / 2)) / (b * FWHM))))
    return np.array(rcp_output)

if __name__ == '__main__':
    t = np.linspace(-15, 15, 201)
    s_0_rcp = rcp(t, 5, 0)
    s_03_rcp = rcp(t, 5, 0.3)
    s_06_rcp = rcp(t, 5, 0.6)
    s_1_rcp = rcp(t, 5, 1)
    plt.figure()
    plt.plot(t, s_0_rcp, 'b')
    plt.plot(t, s_03_rcp, 'orange')
    plt.plot(t, s_06_rcp, 'g')
    plt.plot(t, s_1_rcp, 'r')
    plt.xlim(-15, 15)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.legend([r'$\beta = 0$', r'$\beta = 0.3$', r'$\beta = 0.6$', r'$\beta = 1$'])
    plt.grid(True)
    plt.show()