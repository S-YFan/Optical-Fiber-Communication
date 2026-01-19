import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kv, jvp, kvp
from scipy.optimize import root
import os

#suppose u = pa, w = qa
V = 8.0       # given normalized freq
CORE_RADIUS_A = 1.0


def characteristic_equation(u, l):
    w = np.sqrt(V ** 2 - u ** 2)
    lhs = u * jv(l - 1, u) / jv(l, u)
    rhs = -w * kv(l - 1, w) / kv(l, w)
    return lhs - rhs

eigenvalues = []
u_scan = np.linspace(0.01, V - 0.01, 200)  #掃u,

#l多大自己決定
for l in range(5):  # l=0, 1, 2, 3, 4
    eq_values = np.array([characteristic_equation(val, l) for val in u_scan])
    sign_changes = np.sign(eq_values)
    m = 1
    for i in range(len(sign_changes) - 1):
        if sign_changes[i] != sign_changes[i + 1]:
            initial_guess = (u_scan[i] + u_scan[i + 1]) / 2
            sol = root(characteristic_equation, initial_guess, args=(l,))
            if sol.success:
                u_val = sol.x[0]
                # 檢查是否為重複的根，避免重複加入
                is_duplicate = any(np.isclose(eig['u'], u_val) for eig in eigenvalues if eig['l'] == l)
                if not is_duplicate:
                    eigenvalues.append({'l': l, 'm': m, 'u': u_val})
                    m += 1
eigenvalues.sort(key=lambda x: (x['l'], x['u']))
print("Found Eigenvalues (u values) for V=8:")
for eig in eigenvalues:
    w_val = np.sqrt(V ** 2 - eig['u'] ** 2)
    print(f"LP_{eig['l']}{eig['m']}: u = {eig['u']:.4f}, w = {w_val:.4f}")

def calculate_lp_field(l, u, X, Y, mode_type='cos'):
    w = np.sqrt(V ** 2 - u ** 2)
    R = np.sqrt(X ** 2 + Y ** 2)
    PHI = np.arctan2(Y, X)
    E = np.zeros_like(X, dtype=float)
    core_mask = (R <= CORE_RADIUS_A)
    cladding_mask = (R > CORE_RADIUS_A)
    if mode_type == 'cos':
        angular_part_core = np.cos(l * PHI[core_mask])
        angular_part_cladding = np.cos(l * PHI[cladding_mask])
    elif mode_type == 'sin':
        # l=0 時 sin(0) 為 0，此模態自然為零
        angular_part_core = np.sin(l * PHI[core_mask])
        angular_part_cladding = np.sin(l * PHI[cladding_mask])
    else:
        raise ValueError("mode_type 必須是 'cos' 或 'sin'")
    radial_part_core = jv(l, u * R[core_mask] / CORE_RADIUS_A)
    E[core_mask] = radial_part_core * angular_part_core
    scaling_factor = jv(l, u) / (kv(l, w) + 1e-99)
    radial_part_cladding = scaling_factor * kv(l, w * R[cladding_mask] / CORE_RADIUS_A)
    E[cladding_mask] = radial_part_cladding * angular_part_cladding
    return E

def plot_lp_vector_field(l, m, u, out_dir='output'):
    """
    為給定的 LP_lm 模態，繪製其電場強度和向量場。
    """
    # 建立 2D 網格
    import os
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    grid_size = 201
    plot_range = 2.0 * CORE_RADIUS_A
    x = np.linspace(-plot_range, plot_range, grid_size)
    y = np.linspace(-plot_range, plot_range, grid_size)
    X, Y = np.meshgrid(x, y)
    # 定義要畫的模態類型
    modes_to_plot = ['cos']
    if l > 0:  # 只有 l>0 才有 sin 形式
        modes_to_plot.append('sin')
    for mode_type in modes_to_plot:
        # 我們假設一個 X-polarized 的線性偏振模態
        # Ex 分量為 分量由我們的函式計算, Ey 0
        Ex = calculate_lp_field(l, u, X, Y, mode_type=mode_type)
        Ey = np.zeros_like(Y)

        # --- 開始繪圖 ---
        fig, ax = plt.subplots(figsize=(8, 7))

        # 1. 畫背景的電場強度 (contourf)
        #    使用 'bwr' (Blue-White-Red) 色彩圖，清楚表示正負場
        vmax = np.max(np.abs(Ex))
        contour = ax.contourf(X, Y, Ex, levels=100, cmap='jet', vmin=-vmax, vmax=vmax)
        fig.colorbar(contour, ax=ax, label='$E_y$ Amplitude')

        # 2. 畫向量場 (quiver)
        #    為了圖面清晰，我們只在稀疏的網格點上畫向量箭頭
        skip = 18
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  Ex[::skip, ::skip], Ey[::skip, ::skip],
                  color='black', scale=12, minshaft=2)

        # 3. 畫出纖核邊界
        core_circle = plt.Circle((0, 0), CORE_RADIUS_A, color='lime', fill=False, linestyle='--', linewidth=2)
        ax.add_artist(core_circle)

        # 4. 設定圖表格式
        ax.set_title(f'LP$_{{{l}{m}}}$ Electric Field ({mode_type} mode, X-pol)', fontsize=16)
        ax.set_xlabel('$x/a$')
        ax.set_ylabel('$y/a$')
        ax.set_aspect('equal')
        fig.tight_layout()
        filename = f"{out_dir}/LP{l}{m}_{mode_type}_Xpol.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)


    for mode_type in modes_to_plot:
        # 我們假設一個 Y-polarized 的線性偏振模態
        # Ex 分量為 0, Ey 分量由我們的函式計算
        Ex = np.zeros_like(X)
        Ey = calculate_lp_field(l, u, X, Y, mode_type=mode_type)

        # --- 開始繪圖 ---
        fig, ax = plt.subplots(figsize=(8, 7))

        # 1. 畫背景的電場強度 (contourf)
        #    使用 'bwr' (Blue-White-Red) 色彩圖，清楚表示正負場
        vmax = np.max(np.abs(Ey))
        contour = ax.contourf(X, Y, Ey, levels=100, cmap='plasma', vmin=-vmax, vmax=vmax)
        fig.colorbar(contour, ax=ax, label='$E_y$ Amplitude')

        # 2. 畫向量場 (quiver)
        #    為了圖面清晰，我們只在稀疏的網格點上畫向量箭頭
        skip = 18
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  Ex[::skip, ::skip], Ey[::skip, ::skip],
                  color='black', scale=12, minshaft=2)

        # 3. 畫出纖核邊界
        core_circle = plt.Circle((0, 0), CORE_RADIUS_A, color='lime', fill=False, linestyle='--', linewidth=2)
        ax.add_artist(core_circle)

        # 4. 設定圖表格式
        ax.set_title(f'LP$_{{{l}{m}}}$ Electric Field ({mode_type} mode, Y-pol)', fontsize=16)
        ax.set_xlabel('$x/a$')
        ax.set_ylabel('$y/a$')
        ax.set_aspect('equal')

        '''
        fig.tight_layout()
        filename = f"{out_dir}/LP{l}{m}_{mode_type}_Ypol.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        '''

# --- 5. 執行所有繪圖任務 ---

print("\n正在生成所有模態的電場分佈圖...")
for eig in eigenvalues:
    plot_lp_vector_field(eig['l'], eig['m'], eig['u'])

plt.show()
print("\n所有圖像已生成。")