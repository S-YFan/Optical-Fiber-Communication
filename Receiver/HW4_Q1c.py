import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc


p_1 = np.linspace(1e-3,1-1e-3,1000)
p_0 = 1-p_1
sigma_1 = 2/15
sigma_0 = 1/30
I_0 = 0
I_1 = 1

Q = lambda x: 0.5 * erfc(x / 2**0.5)

plt.figure(figsize=(10,7))

A_coeff = (1/(2*sigma_0**2)-1/(2*sigma_1**2))
B_coeff = ((I_1)/(sigma_1**2)-(I_0)/(sigma_0**2))
C_coeff =((I_1**2)/(2*sigma_1**2)+(I_0**2)/(2*sigma_0**2)+np.log(sigma_1/sigma_0))
I_precise_4 = (-B_coeff+np.sqrt(B_coeff**2+4*A_coeff*(C_coeff+np.log(p_0/p_1))))/(2*A_coeff)
BER_4 = p_0*Q((I_precise_4-I_0)/sigma_0)+p_1*Q((I_1-I_precise_4)/sigma_1)
#畢竟I_d會取正的，因為Id顯然會大於0，因此負根不取。

sigma_1 = 1/12
sigma_0 = 1/12

B_coeff = ((I_1)/(sigma_1**2)-(I_0)/(sigma_0**2))
C_coeff =((I_1**2)/(2*sigma_1**2)+(I_0**2)/(2*sigma_0**2)+np.log(sigma_1/sigma_0))
I_precise_1 = (C_coeff+np.log(p_0/p_1))/B_coeff
BER_1 = p_0*Q((I_precise_1-I_0)/sigma_0)+p_1*Q((I_1-I_precise_1)/sigma_1)

sigma_1 = 1/18
sigma_0 = 1/9

A_coeff = (1/(2*sigma_0**2)-1/(2*sigma_1**2))
B_coeff = ((I_1)/(sigma_1**2)-(I_0)/(sigma_0**2))
C_coeff =((I_1**2)/(2*sigma_1**2)+(I_0**2)/(2*sigma_0**2)+np.log(sigma_1/sigma_0))
I_precise_05 = (-B_coeff+np.sqrt(B_coeff**2+4*A_coeff*(C_coeff+np.log(p_0/p_1))))/(2*A_coeff)
BER_05 = p_0*Q((I_precise_05-I_0)/sigma_0)+p_1*Q((I_1-I_precise_05)/sigma_1)

plt.plot(p_1,I_precise_4, label = r"$\sigma_1/\sigma_0=4$")
plt.plot(p_1,I_precise_1, label = r"$\sigma_1/\sigma_0=1$")
plt.plot(p_1,I_precise_05, label = r"$\sigma_1/\sigma_0=\frac{1}{2}$")
plt.xlabel(r"$p_1$")
plt.ylabel(r"$I_d$")
plt.title(r"$p$ v.s. optimized threshold(Precisely $I_d$)")
plt.legend()
plt.grid()
plt.xlim(0,1)
plt.ylim(bottom=0)
plt.locator_params(nbins=10)

plt.figure(figsize=(10,7))
plt.semilogy(p_1,BER_4, label = r"$\sigma_1/\sigma_0=4$")
plt.semilogy(p_1,BER_1, label = r"$\sigma_1/\sigma_0=1$")
plt.semilogy(p_1,BER_05, label = r"$\sigma_1/\sigma_0=\frac{1}{2}$")
plt.xlim(0,1)
plt.xlabel(r"$p_1$")
plt.ylabel("BER")
plt.grid()
plt.legend()
plt.title(r"$p$ v.s. BER (via optimized threshold as integral boundary(Precisely $I_d$))")

plt.show()