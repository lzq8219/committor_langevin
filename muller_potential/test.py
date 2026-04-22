import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('muller_potential/loss_results.csv')
output_label_loss = ['gamma','kbt','rl2_loss','rl1_loss','linf_loss','l2_lq','l1_lq','linf_lq','rl2_loss_pinn','rl1_loss_pinn','linf_loss_pinn','l2_lq_pinn','l1_lq_pinn','linf_lq_pinn']
output_label_2 = ['kbt','gamma', 'l2 loss', 'l2 loss pinn', 'l2 pinn loss', 'l2 pinn loss pinn']
output_label_1 = ['kbt', 'gamma', 'l1 loss', 'l1 loss pinn', 'l1 pinn loss', 'l1 pinn loss pinn']
output_label_inf = ['kbt', 'gamma', 'linf loss', 'linf loss pinn', 'linf pinn loss', 'linf pinn loss pinn']


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
with open("muller_potential/table.txt", "w", encoding="utf-8") as f:
    f.write(latex)
print("Saved table.txt")

