taps = [1,0,1]
seed = [1]*3

def lfsr_generate(taps, seed):
    """
    taps : list of 0/1, defines the LFSR taps. If it is a PRBS (pseudo-random binary sequence), it can generate a spectrally uniform white noise.
    seed : list of 0/1, initial state, usually an all-one list (other values can be used as well; theoretically, after one full cycle of length 2^n - 1, if it is a PRBS, it will return to the original value).
    return: output list, the generated LFSR sequence
    """
    L = len(seed)
    output = []
    state = seed.copy()  # avoid change original seed
    temp = 0
    for KK in range(2**L - 1):
        # 計算 feedback
        for i in range(L):
            if taps[i] == 1:
                temp += state[i]
        temp_0 = temp % 2
        # Right shift, insert feedback at the front        
        output.append(state[-1])
        state.pop()
        state.insert(0, temp_0)
        temp = 0
    # we can verify the taps whethere is PRBS by return state(optional)
    return output

if __name__ == '__main__':
    lfsr_primitity = [0, 1, 1]  # x^3+x^2+1 which is prime
    lfsr_testing = [1, 1, 1]  # initial seed is 111，there are 2^3-1 combination.
    s1, s_s = lfsr_generate(lfsr_primitity, lfsr_testing)

    print(s1, len(s1), s_s)
