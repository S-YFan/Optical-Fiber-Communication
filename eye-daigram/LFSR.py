taps = [1,0,1]
seed = [1]*3

def lfsr_generate(taps, seed):
    """
    taps : list of 0/1, 給的lfsr，若是PRBS可以產生頻譜上均勻的white noise。
    seed : list of 0/1, 最初的all one list (也可以給其他值，理論上經過一周期2^n-1後，若是PRBS它會回到原值)

    return: output list, 生成的 LFSR 序列
    """
    L = len(seed)
    output = []
    state = seed.copy()  # 避免改變原本 seed
    temp = 0
    for KK in range(2**L - 1):
        # 計算 feedback
        for i in range(L):
            if taps[i] == 1:
                temp += state[i]
        temp_0 = temp % 2
        # shift 右移，把 feedback 放到最前面
        output.append(state[-1])
        state.pop()
        state.insert(0, temp_0)
        temp = 0
    # 若好奇state在尾端有無符合原先預期，就把state也一起回傳遍可驗證
    return output

if __name__ == '__main__':
    lfsr_primitity = [0, 1, 1]  # x^3+x^2+1 which is prime
    lfsr_testing = [1, 0, 1]  # 假設初始是111，會有2^3-1的組合
    s1, s_s = lfsr_generate(lfsr_primitity, lfsr_testing)
    print(s1, len(s1), s_s)