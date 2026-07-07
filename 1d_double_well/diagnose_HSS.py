import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('1d_double_well/errors_rates_t.txt')
print(df)
plt.plot(df['t'], df['l2 loss HSS'],label='HSS')
plt.plot(df['t'], df['l2 loss'],label='CGW')
plt.yscale('log')
plt.xlabel('Iteration steps')
plt.ylabel('L2 Loss')
plt.legend()
plt.show()
plt.savefig('1d_double_well/figures/HSS_L2.png')

plt.plot(df['t'], df['l1 loss HSS'],label='HSS')
plt.plot(df['t'], df['l1 loss'],label='CGW')
plt.yscale('log')
plt.xlabel('Iteration steps')
plt.ylabel('L1 Loss')
plt.legend()
plt.show()
plt.savefig('1d_double_well/figures/HSS_L1.png')