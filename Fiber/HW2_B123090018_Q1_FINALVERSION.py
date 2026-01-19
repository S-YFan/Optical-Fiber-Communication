import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kv
from scipy.optimize import root

# 參數
n1, n2 = 1.7, 1.4
a, wavelength = 2.0, 1.55
k0 = 2*np.pi / wavelength
V = k0*a*np.sqrt(n1**2 - n2**2)
delta = (n1**2 - n2**2)/(2*n1**2)

pa = np.linspace(0.01, V-1e-5, 1000)
qa = np.sqrt(V**2 - pa**2)

#找根是最complicated！

# 特徵方程差值函數
def diff_TE_m0(pa):
    pa = float(np.atleast_1d(pa)[0]); qa0 = np.sqrt(V**2 - pa**2)
    return -jv(1,pa)/jv(0,pa) - kv(1,qa0)/(qa0*kv(0,qa0))

def diff_TM_m0(pa):
    pa = float(np.atleast_1d(pa)[0]); qa0 = np.sqrt(V**2 - pa**2)
    return -jv(1,pa)/jv(0,pa) - (n2**2/n1**2)*(kv(1,qa0)/(qa0*kv(0,qa0)))

def diff_HE_m(pa, m):
    pa = float(np.atleast_1d(pa)[0]); qa0 = np.sqrt(V**2 - pa**2)
    neff = np.sqrt(n1**2 - (pa/(k0*a))**2)
    J = jv(m-1,pa)/(pa*jv(m,pa))
    term = -(kv(m-1,qa0)/(qa0*kv(m,qa0))) - m/(qa0**2)
    sec = np.sqrt(delta**2*term**2 + (m**2*neff**2/n1**2)*((1/pa**2)+(1/qa0**2))**2)
    RHS = -((n1**2+n2**2)/(2*n1**2))*term + (m/pa**2 - sec)
    return J - RHS

def diff_EH_m(pa, m):
    pa = float(np.atleast_1d(pa)[0]); qa0 = np.sqrt(V**2 - pa**2)
    neff = np.sqrt(n1**2 - (pa/(k0*a))**2)
    J = -jv(m+1,pa)/(pa*jv(m,pa))
    term = -(kv(m-1,qa0)/(qa0*kv(m,qa0))) - m/(qa0**2)
    sec = np.sqrt(delta**2*term**2 + (m**2*neff**2/n1**2)*((1/pa**2)+(1/qa0**2))**2)
    RHS = -((n1**2+n2**2)/(2*n1**2))*term - (m/pa**2 - sec)
    return J - RHS

# 根搜尋函數
def find_roots(func, *args):
    scan = np.linspace(0.01, V-1e-5, 5000)
    vals = np.array([func(x, *args) for x in scan])
    roots = []
    for i in range(len(scan)-1):
        if np.sign(vals[i]) != np.sign(vals[i+1]):
            guess = (scan[i]+scan[i+1])/2
            sol = root(func, guess, args=args)
            if sol.success:
                r = float(sol.x[0])
                if r>0 and not any(np.isclose(r, rr, atol=1e-6) for rr in roots):
                    roots.append(r)
    return roots


def fix_plot_asymptotes(y_values, threshold=20.0):
    y_fixed = y_values.copy()
    jumps = np.abs(np.diff(y_fixed))
    jump_indices = np.where(jumps > threshold)[0]
    # 在「跳躍發生後」的那個點 (i+1) 插入 np.nan
    # Matplotlib 會自動在 nan 處斷開線條
    y_fixed[jump_indices + 1] = np.nan
    return y_fixed

# 先收集所有根
roots = {}

roots = {
    'TE0': find_roots(diff_TE_m0),
    'TM0': find_roots(diff_TM_m0)
}

for m in range(1, 7):
    roots[f'HE{m}'] = find_roots(diff_HE_m, m)
    roots[f'EH{m}'] = find_roots(diff_EH_m, m)

# 列印根值，格式如 TE01, TE02, HE11, HE12, EH11, EH12, ...
for mode, rs in roots.items():
    print(f"{mode} roots:")
    if rs:
        for i, r in enumerate(rs, start=1):
            beta = np.sqrt(n1 ** 2 * k0 ** 2 * a ** 2 - r ** 2) / (a*1e-6)
            # 根序號用兩位數（TE0、TM0），或單位數（HE1、EH1）
            if mode in ('TE0', 'TM0'):
                label = f"{mode}{i:01d}"   # e.g. TE0 -> TE01, TE02
            else:
                label = f"{mode}{i}"       # e.g. HE1 -> HE11, HE12
            print(f"  {label}: pa = {r:.6f}, beta={beta:.6f}")
    else:
        print("  (no roots found)")
print()


# m=0: TE & TM
plt.figure()

LHS_TE_TM_array = -jv(1,pa)/jv(0,pa)
LHS_TE_TM_fixed = fix_plot_asymptotes(LHS_TE_TM_array)



for mode0, diff0, color, marker in [
    ('TE', diff_TE_m0, 'r', 'o'),
    ('TM', diff_TM_m0, 'g', 's')
]:
    roots0 = find_roots(diff0)

    plt.plot(pa, LHS_TE_TM_fixed, 'b-', label='LHS TE/TM')
    if mode0=='TE':
        plt.plot(pa, kv(1,qa)/(qa*kv(0,qa)), 'r--', label='RHS TE')
    else:
        plt.plot(pa, (n2**2/n1**2)*(kv(1,qa)/(qa*kv(0,qa))), 'g--', label='RHS TM')
    for r in roots0:
        # LHS
        y_LHS = -jv(1,r)/jv(0,r)
        plt.plot(r, y_LHS, color+marker, markersize=6)
        # RHS
        qa0 = np.sqrt(V**2 - r**2)
        if mode0=='TE':
            y_RHS = kv(1,qa0)/(qa0*kv(0,qa0))
        else:
            y_RHS = (n2**2/n1**2)*(kv(1,qa0)/(qa0*kv(0,qa0)))
        plt.plot(r, y_RHS, color+marker, markersize=6)
    plt.title(f'm=0: TE/{mode0}')
    plt.xlabel('pa'); plt.ylabel('Characteristic Equation Value')
    plt.xlim(0,V); plt.ylim(-2,2); plt.grid(True); plt.legend()
    plt.tight_layout()

# m=1..6: HE and EH
for m in range(1,7):
    LHS_array_HE = jv(m - 1, pa) / (pa * jv(m, pa))
    LHS_array_EH = -jv(m + 1, pa) / (pa * jv(m, pa))
    LHS_array_HE_fixed = fix_plot_asymptotes(LHS_array_HE)
    LHS_array_EH_fixed = fix_plot_asymptotes(LHS_array_EH)

    roots_HE = find_roots(diff_HE_m, m)
    roots_EH = find_roots(diff_EH_m, m)
    plt.figure()
    # LHS
    plt.plot(pa, LHS_array_HE_fixed, 'm-', label=f'HE{m}(LHS)')
    plt.plot(pa, LHS_array_EH_fixed, 'c--', label=f'EH{m}(LHS)')
    # RHS
    neff_arr = np.sqrt(n1**2 - (pa/(k0*a))**2)
    term = -(kv(m-1,qa)/(qa*kv(m,qa))) - m/qa**2
    sec = np.sqrt(delta**2*term**2 + (m**2*neff_arr**2/n1**2)*((1/pa**2)+(1/qa**2))**2)
    RHS_HE = -((n1**2+n2**2)/(2*n1**2))*term + (m/pa**2 - sec)
    RHS_EH = -((n1**2+n2**2)/(2*n1**2))*term - (m/pa**2 - sec)
    plt.plot(pa, RHS_HE, 'm-.', label=f'HE{m}(RHS)')
    plt.plot(pa, RHS_EH, 'c:', label=f'EH{m}(RHS)')
    # 標根
    for r in roots_HE:
        y1 = jv(m-1,r)/(r*jv(m,r))
        qa0 = np.sqrt(V**2 - r**2)
        neff = np.sqrt(n1**2 - (r/(k0*a))**2)
        term0 = -(kv(m-1,qa0)/(qa0*kv(m,qa0))) - m/(qa0**2)
        sec0 = np.sqrt(delta**2*term0**2 + (m**2*neff**2/n1**2)*((1/r**2)+(1/qa0**2))**2)
        y2 = -((n1**2+n2**2)/(2*n1**2))*term0 + (m/r**2 - sec0)
        plt.plot(r, y1, 'ko', markersize=6); plt.plot(r, y2, 'ko', markersize=6)
    for r in roots_EH:
        y1 = -jv(m+1, r)/(r*jv(m, r))
        qa0 = np.sqrt(V**2 - r**2)
        neff = np.sqrt(n1**2 - (r/(k0*a))**2)
        term0 = -(kv(m-1,qa0)/(qa0*kv(m,qa0))) - m/(qa0**2)
        sec0 = np.sqrt(delta**2*term0**2 + (m**2*neff**2/n1**2)*((1/r**2)+(1/qa0**2))**2)
        y2 = -((n1**2+n2**2)/(2*n1**2))*term0 - (m/r**2 - sec0)
        plt.plot(r, y1, 'ks', markersize=6); plt.plot(r, y2, 'ks', markersize=6)
    plt.title(f'm={m}: HE{m} & EH{m}')
    plt.xlabel('pa'); plt.ylabel('Characteristic Equation Value')
    plt.xlim(0,V);plt.ylim(-2,2); plt.grid(True); plt.legend()
    plt.tight_layout()

plt.show()
