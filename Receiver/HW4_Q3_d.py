import matplotlib.pyplot as plt
import numpy as np
import LFSR
import RaisedCosinePulse as rcp



taps = [1,0,0,1,0,1,0,0,0,0,0,1] #x^12+x^6+x^4+x+1 s1
seed=[1]*12

output_s1 = LFSR.lfsr_generate(taps,seed)

samples_per_bit =200

different_beta_Jitter = []
store_timmingjitter = []


blist = np.linspace(0,1,101)

for b in blist:
    t_NRZ = np.linspace(-2, 2, 2 * samples_per_bit + 1)
    prc03_forNRZ = rcp.rcp(t_NRZ, 1, b)
    upsample_output_s1 = np.concatenate(
        [[index_for_upsample] + [0] * (samples_per_bit - 1) for index_for_upsample in output_s1])
    convolutionNRZ_s1 = np.convolve(upsample_output_s1, prc03_forNRZ, mode='full')
    convolutionNRZ_delay13 = convolutionNRZ_s1 + 2 * np.pad(convolutionNRZ_s1, (13 * samples_per_bit, 0),
                                                            mode='constant')[:-13 * samples_per_bit]
    #這樣對於1->2而言，有這個值存在的可能比較大，不然delay1和delay7信號相關性長得比較像。
    store_timmingjitter = []
    threshold = 1.5
    for j in range(len(convolutionNRZ_delay13) - 1):
        y1 = convolutionNRZ_delay13[j]
        y2 = convolutionNRZ_delay13[j + 1]
        if (y1 < threshold and y2 > threshold) or \
                (y1 > threshold and y2 < threshold):
            #
            delta = (threshold - y1) / (y2 - y1)
            t_precise = j + delta
            t_relative = t_precise % samples_per_bit
            store_timmingjitter.append(t_relative)
    # 算jitter
    if store_timmingjitter:
        jitter_pp = max(store_timmingjitter) - min(store_timmingjitter)
        different_beta_Jitter = different_beta_Jitter + [jitter_pp]


different_beta_Jitter = np.array(different_beta_Jitter)
plt.figure()
plt.plot(np.linspace(0,1,101),different_beta_Jitter / samples_per_bit)
plt.xlim(0,1)
plt.ylim(bottom=0)
plt.xlabel(r"$\beta$")
plt.ylabel("peak to peak timing jitter(ps)")
plt.show()