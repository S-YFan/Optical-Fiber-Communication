import matplotlib.pyplot as plt
import numpy as np

sigma_term = np.linspace(0.0001,10,5000)

I_tilde = 1/(1+sigma_term)
I_precise = np.linspace(0,0,5000)
for i, value in enumerate(sigma_term):
    if value == 1:
        I_precise[i]=0.5
    I_precise[i] = (-1+np.sqrt(1+(value**2-1)*(1+2*(1/(6*(1+1/value)))**2*np.log(value))))/((value**2-1))
#畢竟I_d會取正的，因為Id顯然會大於0，因此負根不取。

plt.figure(figsize=(10,7))
plt.plot(I_tilde,sigma_term, label = r"Approximate $I_d$")
plt.plot(I_precise,sigma_term, label = r"Preciesly $I_d$")
plt.ylabel(r"$\frac{\sigma_1}{\sigma_0}$")
plt.xlabel(r"$I_d\;or\; \tilde{I}_d$")
plt.title(r"Approximate $\tilde{I}_d$ v.s. Preciesly $I_d$")
plt.legend()
plt.grid()
plt.xlim(0,1)
plt.ylim(0,10)
plt.locator_params(nbins=10)
plt.show()