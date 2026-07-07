import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('1d_double_well/errors_rates.txt')
output_label_loss = ['kbt','gamma','l2 loss','l1 loss','linf loss','l2 pinn loss','l1 pinn loss','linf pinn loss','l2 loss pinn','l1 loss pinn','linf loss pinn','l2 pinn loss pinn','l1 pinn loss pinn','linf pinn loss pinn']
output_label_2 = ['kbt','gamma','l2 loss','l2 loss pinn','l2 pinn loss','l2 pinn loss pinn']
output_label_1 = ['kbt','gamma','l1 loss','l1 loss pinn','l1 pinn loss','l1 pinn loss pinn']
output_label_inf = ['kbt','gamma','linf loss','linf loss pinn','linf pinn loss','linf pinn loss pinn']
gamma_inv = [1.0 / gamma for gamma in df['gamma']]
plt.plot(gamma_inv, df['NN rate'], marker='o', label='NN rate')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('1/\gamma')
plt.ylabel('Transition rate')
plt.legend(loc='upper right')
plt.savefig('1d_double_well/figures/rates_vs_gamma_inv.png')

def fmt_sig3_val(v):
    try:
        s = format(float(v), ".2e")
    except Exception:
        s = str(v)
    return s

# 为每列最小值加粗并格式化为三位有效数字
def bold_min_col(col):
    s = col.copy()
    minmask = s == s.min()
    out = []
    for val, ismin in zip(s, minmask):
        txt = fmt_sig3_val(val)
        if ismin:
            # 插入 LaTeX 加粗命令，Styler.to_latex 可能会对内容转义，需 escape=False
            out.append(r"\textbf{" + txt + "}")
        else:
            out.append(txt)
    return out

styler = df[output_label_loss].style.format(fmt_sig3_val)      # 基本格式（可选）
#styler = styler.apply(bold_min_col, axis=0) # 对每列应用加粗最小值

# 导出 LaTeX。注意 escape=False 以保留 \textbf{} 等命令
latex = styler.to_latex(hrules=True)
with open("1d_double_well/table_inf.txt", "w", encoding="utf-8") as f:
    f.write(latex)
print("Saved table.txt")

