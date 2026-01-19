from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc


p_1 = 1/2
p_0 = 1/2
sigma_1 = 2/15
sigma_0 = 1/30
I_0 = 0
I_1 = 1

Q = lambda x: 0.5 * erfc(x / 2**0.5)

plt.figure(figsize=(10,7))

I_D = np.linspace(0,1,1000)
BER_4 = p_0*Q((I_D-I_0)/sigma_0)+p_1*Q((I_1-I_D)/sigma_1)
BER_Q4=Q((I_1-I_0)/(sigma_0+sigma_1))
opt_Id_4 = (sigma_0 * I_1 + sigma_1 * I_0) / (sigma_0 + sigma_1)
plt.semilogy(I_D, BER_4, label=r"$\sigma_1/\sigma_0=4$")

sigma_1 = 1/12
sigma_0 = 1/12
lowest_1 = (I_1-I_0)/(sigma_0+sigma_1)
BER_1 = p_0*Q((I_D-I_0)/sigma_0)+p_1*Q((I_1-I_D)/sigma_1)
BER_Q1=Q((I_1-I_0)/(sigma_0+sigma_1))
opt_Id_1 = (sigma_0 * I_1 + sigma_1 * I_0) / (sigma_0 + sigma_1)
plt.semilogy(I_D, BER_1, label=r"$\sigma_1/\sigma_0=1$")


sigma_1 = 1/18
sigma_0 = 1/9
BER_05 = p_0*Q((I_D-I_0)/sigma_0)+p_1*Q((I_1-I_D)/sigma_1)
BER_Q05=Q((I_1-I_0)/(sigma_0+sigma_1))
opt_Id_05 = (sigma_0 * I_1 + sigma_1 * I_0) / (sigma_0 + sigma_1)
plt.semilogy(I_D, BER_05, label=r"$\sigma_1/\sigma_0=\frac{1}{2}$")


opt_Id = [opt_Id_4,opt_Id_1,opt_Id_05]
min_ber = [BER_Q4,BER_Q1,BER_Q05]
print(min_ber)



plt.plot(opt_Id, min_ber, 'o',
         markersize=10,  # 稍微加大一點
         color="blue",  # 圓圈邊框顏色跟線一樣
         markerfacecolor='white',  # [關鍵] 改成白色填滿，蓋住背後的線
         zorder=10)  # [關鍵] 強制把點畫在最上層，避免被格線蓋住

plt.legend()
plt.grid()
plt.xlim(0,1)
plt.ylabel('Bit Error Rate (BER)')
plt.xlabel('Decision Threshold ($I_D$)')
plt.title('BER vs Threshold (Q=6)')
plt.show()
