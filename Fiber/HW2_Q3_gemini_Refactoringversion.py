"""
光纖通訊系統模擬腳本 (HW1 & HW2)

這個腳本模擬了 10-Gbps 的 NRZ 和 50%-RZ 訊號在不同長度的
單模光纖中傳輸後，因色散 (chromatic dispersion) 造成的波形失真。

主要功能:
1.  (HW1) 產生 PRBS (偽隨機二進位序列) 作為數據源。
2.  (HW1) 產生 NRZ 和 RZ 基礎訊號，並透過 Raised Cosine 濾波器進行脈衝成形。
3.  (HW2) 根據光纖參數計算 GVD 參數 beta2。
4.  (HW2) 使用 Split-Step Fourier Method (SSFM) 的簡化版本（只考慮色散）
    來模擬訊號在光纖中的傳播。
5.  (HW2) 繪製傳輸後訊號的眼圖 (Eye Diagram) 和頻譜 (Spectrum) 以分析訊號品質。
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ==============================================================================
# Section 1: 核心函式庫 (Core Functions)
# ==============================================================================

def lfsr_generate(taps: List[int], seed: List[int]) -> List[int]:
    """
    產生一個 LFSR (線性回饋移位暫存器) 序列。

    Args:
        taps (List[int]): 回饋抽頭 (feedback taps) 的位置列表。
        seed (List[int]): 移位暫存器的初始狀態 (seed)。

    Returns:
        List[int]: 產生的 PRBS 序列。
    """
    L = len(seed)
    num_bits = 2**L - 1
    state = seed.copy()
    output = []

    for _ in range(num_bits):
        # 在 sum() 中使用生成器表達式是計算 feedback 的一種 Pythonic 寫法
        feedback_val = sum(state[i] for i, tap in enumerate(taps) if tap == 1) % 2
        output.append(state[-1])
        state.pop()
        state.insert(0, feedback_val)

    return output

def p_rc(t: np.ndarray, T: float, b: float) -> np.ndarray:
    """
    Args:
        t (np.ndarray): 時間軸陣列。
        T (float): 位元週期 (Bit duration)。
        b (float): 滾降係數 (Roll-off factor)。
    Returns:
        np.ndarray: 產生的脈衝波形。
    """
    # 向量化 (Vectorized) 寫法
    prc_output = np.zeros_like(t, dtype=float)
    abs_t = np.abs(t)

    FWHM = 2*T #20251025的修改

    flat_top_condition = abs_t < (1 - b) * T
    prc_output[flat_top_condition] = 1.0

    roll_off_condition = (abs_t >= (1 - b) * T ) & (abs_t < (1 + b) * T )
    t_roll_off = abs_t[roll_off_condition]
    prc_output[roll_off_condition] = 0.5 * (1 + np.cos(np.pi * (t_roll_off - (1 - b) * FWHM / 2) / (b * FWHM)))

    return prc_output

def generate_shaped_signal(prbs: List[int], samples_per_bit: int, T_bit: float, beta: float, duty_cycle: float) -> np.ndarray:
    """
    將 PRBS 序列轉換為經過 Raised Cosine 脈衝成形的訊號。

    Args:
        prbs (List[int]): PRBS 序列。
        samples_per_bit (int): 每個 bit 的取樣點數。
        T_bit (float): 位元週期。
        beta (float): Raised Cosine 滾降係數。
        duty_cycle (float): 佔空比 (例如，NRZ=1.0, 50%-RZ=0.5)。

    Returns:
        np.ndarray: 最終的時域訊號功率 P(0,t)。
    """
    # 1. 產生基礎的矩形 RZ/NRZ 訊號
    on_samples = int(round(samples_per_bit * duty_cycle))
    one_pattern = np.concatenate([np.ones(on_samples), np.zeros(samples_per_bit - on_samples)])
    zero_pattern = np.zeros(samples_per_bit)
    base_signal = np.concatenate([one_pattern if bit == 1 else zero_pattern for bit in prbs])

    # 2. 建立 Raised Cosine 濾波器
    t_filter = np.linspace(-T_bit, T_bit, samples_per_bit)
    rc_filter = p_rc(t_filter, T_bit/2, beta)
    ##這邊要注意到！！！！20251025

    # 3. 透過 FFT 實現卷積
    N = len(base_signal)
    # 將濾波器補零以匹配訊號長度
    rc_filter_padded = np.zeros(N)
    rc_filter_padded[:len(rc_filter)] = rc_filter

    signal_fft = np.fft.fft(base_signal)
    filter_fft = np.fft.fft(rc_filter_padded)
    shaped_signal_fft = signal_fft * filter_fft

    shaped_signal = np.fft.ifft(shaped_signal_fft).real

    return shaped_signal

def propagate_fiber(P_in: np.ndarray, z_km: float, beta2_si: float, T_bit: float, samples_per_bit: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    模擬訊號功率 P_in 通過長度為 z_km 的光纖後的結果 (只考慮色散)。

    Args:
        P_in (np.ndarray): 輸入訊號功率 P(0,t)。
        z_km (float): 光纖長度 [km]。
        beta2_si (float): GVD 參數 [s²/m]。
        T_bit (float): 位元週期 [s]。
        samples_per_bit (int): 每個 bit 的取樣點數。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 一個包含 (傳輸後的訊號功率 P(z,t), 頻率軸 f [Hz]) 的元組。
    """
    # 1. 準備 input envelope，並處理數值計算可能產生的負功率
    P_in_non_negative = np.maximum(0, P_in)
    A_in = np.sqrt(P_in_non_negative)
    N = len(A_in)

    # 2. 計算頻率軸
    sampling_interval = T_bit / samples_per_bit
    f = np.fft.fftfreq(N, d=sampling_interval)
    omega = 2 * np.pi * f
    omega_shifted = np.fft.fftshift(omega)

    # 3. 執行 FFT
    A_in_fft = np.fft.fft(A_in)
    A_in_fft_shifted = np.fft.fftshift(A_in_fft)

    # 4. 施加 dispersion-induced phase shifts
    z_meters = z_km * 1000
    H = np.exp(-1j * 0.5 * beta2_si * omega_shifted**2 * z_meters)
    A_out_fft_shifted = A_in_fft_shifted * H

    # 5. 執行 IFFT
    A_out_fft = np.fft.ifftshift(A_out_fft_shifted)
    A_out = np.fft.ifft(A_out_fft)

    # 6. 計算輸出功率
    P_out = np.abs(A_out)**2

    return P_out, f

def get_eye_diagram_data(signal: np.ndarray, samples_per_bit: int, span_bits: float = 2.0, num_traces: int = 500) -> np.ndarray:
    """
    從一維時域訊號中擷取用於繪製眼圖的數據。
    """
    window_samples = int(span_bits * samples_per_bit)
    start_sample = 5 * samples_per_bit

    if len(signal) < start_sample + window_samples:
        return np.empty((0, window_samples))

    max_traces = (len(signal) - start_sample - window_samples) // samples_per_bit + 1
    traces_to_gen = min(num_traces, max_traces)

    if traces_to_gen <= 0:
        return np.empty((0, window_samples))

    eye_data = np.zeros((traces_to_gen, window_samples))
    for i in range(traces_to_gen):
        segment_start = start_sample + i * samples_per_bit
        eye_data[i, :] = signal[segment_start : segment_start + window_samples]

    return eye_data

# ==============================================================================
# Section 2: 主程式執行區塊 (Main Script Block)
# ==============================================================================

if __name__ == "__main__":
    # --- 1. 模擬參數 ---
    BIT_RATE = 10e9  # 10 Gbps
    T_BIT = 1 / BIT_RATE
    SAMPLES_PER_BIT = 50
    BETA_RC = 0.5  # Raised Cosine 滾降係數

    # --- 2. 光纖參數與 beta2 計算 ---
    D_PARAMETER = 15  # [ps/(nm*km)]
    LAMBDA_C = 1580e-9 # [m]
    C_SPEED = 2.99792e8 # [m/s]

    # 將 D 轉換為 SI 單位 [s/m^2]
    D_si = D_PARAMETER * 1e-12 / (1e-9 * 1e3)
    # 計算 beta2 (SI 單位: s^2/m)
    beta2_si = -(D_si * LAMBDA_C**2) / (2 * np.pi * C_SPEED)

    print(f"計算出的 GVD 參數 β2 = {beta2_si:.4g} s²/m")

    # --- 3. 定義所有要測試的 PRBS taps ---
    lfsr_taps_1 = [1,0,0,1,0,1,0,0,0,0,0,1] # x^12+x^6+x^4+x+1
    lfsr_taps_2 = [0,1,1,0,0,0,0,0,1,0,0,1] # x^12+x^9+x^3+x^2+1
    lfsr_taps_3 = [0,1,0,0,1,0,0,1,0,0,0,1] # x^12+x^8+x^5+x^2+1

    all_taps = {
        'Seq 1 (x^12+x^6+x^4+x+1)': lfsr_taps_1,
        'Seq 2 (x^12+x^9+x^3+x^2+1)': lfsr_taps_2,
        'Seq 3 (x^12+x^8+x^5+x^2+1)': lfsr_taps_3
    }

    lfsr_seed = [1]*12

    # --- 4. 遍歷每個 PRBS 序列進行完整模擬 ---
    for seq_name, taps in all_taps.items():
        print(f"\n===== 正在處理序列: {seq_name} =====")
        prbs_sequence = lfsr_generate(taps, lfsr_seed)

        # --- 4a. 產生初始訊號 P(0,t) ---
        print("  正在產生初始訊號 P(0,t)...")
        P_nrz_in = generate_shaped_signal(prbs_sequence, SAMPLES_PER_BIT, T_BIT, BETA_RC, duty_cycle=1.0)
        plt.figure()
        t=np.linspace(0,4095,len(P_nrz_in))
        plt.plot(t, P_nrz_in)
        plt.xlim(0, 20)

        P_rz50_in = generate_shaped_signal(prbs_sequence, SAMPLES_PER_BIT, T_BIT, BETA_RC, duty_cycle=0.5)
        plt.figure()
        t = np.linspace(0, 4095, len(P_rz50_in))
        plt.plot(t, P_rz50_in)
        plt.xlim(0, 20)

        plt.show()

        # --- 5. 設定內部模擬迴圈參數 ---
        fiber_lengths_km = [10, 25, 50]
        signals_to_simulate = {'NRZ': P_nrz_in, '50%-RZ': P_rz50_in}

        # --- 6. 為當前序列建立新的繪圖視窗 ---
        fig_eye, axes_eye = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
        fig_spec, axes_spec = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
        fig_eye.suptitle(f'Eye Diagrams for {seq_name}', fontsize=16)
        fig_spec.suptitle(f'Spectra for {seq_name}', fontsize=16)

        # --- 7. 執行模擬與繪圖 ---
        print("  正在對不同訊號與光纖長度進行模擬...")
        for row_idx, (name, P_in) in enumerate(signals_to_simulate.items()):
            max_power = np.max(P_in) # 用於統一 Y 軸

            for col_idx, z_km in enumerate(fiber_lengths_km):
                print(f"    模擬 {name} 在 {z_km} km...")

                # (A) 執行光纖傳播模擬
                P_out, freq_axis = propagate_fiber(P_in, z_km, beta2_si, T_BIT, SAMPLES_PER_BIT)

                # (B) 處理並繪製眼圖
                ax_e = axes_eye[row_idx, col_idx]
                eye_data = get_eye_diagram_data(P_out, SAMPLES_PER_BIT, span_bits=2.0)

                if eye_data.shape[0] > 0:
                    time_axis = np.linspace(-1.0, 1.0, eye_data.shape[1])
                    for trace in eye_data:
                        ax_e.plot(time_axis, trace, color='royalblue', alpha=0.05)

                ax_e.set_title(f'{name} @ {z_km} km')
                ax_e.grid(True, linestyle='--', alpha=0.6)
                ax_e.set_ylim(-0.1 * max_power, 3 * max_power)
                if row_idx == 1: ax_e.set_xlabel('Time (bits)')
                if col_idx == 0: ax_e.set_ylabel('Intensity (Power)')

                # (C) 處理並繪製頻譜
                ax_s = axes_spec[row_idx, col_idx]
                P_out_fft = np.fft.fft(P_out)
                freq_shifted_ghz = np.fft.fftshift(freq_axis) / 1e9
                P_out_fft_shifted_abs = np.abs(np.fft.fftshift(P_out_fft))

                # 將頻譜正規化至 0 dB，並加上極小值以避免 log(0)
                max_val = np.max(P_out_fft_shifted_abs)
                if max_val == 0: max_val = 1 # 避免對全零訊號做除以零的運算

                P_out_fft_db = 20 * np.log10(P_out_fft_shifted_abs / max_val + 1e-9)

                ax_s.plot(freq_shifted_ghz, P_out_fft_db, color='crimson')
                ax_s.set_title(f'{name} @ {z_km} km')
                ax_s.set_xlim(-40, 40)
                ax_s.set_ylim(-80, 5)
                ax_s.grid(True, linestyle='--', alpha=0.6)
                if row_idx == 1: ax_s.set_xlabel('frequecny (GHz)')
                if col_idx == 0: ax_s.set_ylabel('Magnitude (dB)')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # 調整佈局以容納大標題

    plt.show()
    print("全部模擬完成。")

