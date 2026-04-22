import numpy as np
from scipy import stats

# -------- 输入数据 --------

data = np.array([0.8,0.5,0.7,0.7,0.45,0.45,0.4,0.35,0.7,0.25,0.2,0.4,0.65,0.4,
0.25,0.8,0.35,0.35,0.6,0.8,0.45,0.05,0.45,0.4,0.5,0.45,0.4,0.35,
0.3,0.55,0.2,0.6,0.5,0.7,0.3,0.9,0.4,0.1,0.25,0.85,0.45,0.4,
0.3,0.1,0.55,0.95,0.5,0.6,0.3,0.6,0.75,0.25,0.15,0.7,0.2,0.3,
0.75,0.45,0.55,0.3,0.45,0.85,0.6,0.3,0.55,0.45,0.2,0.65,0.75,0.5,
0.95,0.35,0.15,0.8,0.2,0.35,0.15,0.55,0.4,0.3,0.6,0.35,0.65,0.15,
0.3,0.3,0.2,0.75,0.6,0.2,0.4,0.5,0.2,0.95,0.3,0.85,0.9,0.4,
0.55,0.6,0.55,0.5,0.5,0.35,0.15,0.2,0.5,0.65,0.65,0.35,0.3,0.0,
0.25,0.9,0.3,0.75,0.25,0.3,0.9,0.3,0.55,0.95,0.3,0.7,0.35,0.5,
0.15,0.45,0.4,0.6,0.55,0.4,0.35,0.3,0.3,0.45,0.2,0.55,0.4,0.6,
0.45,0.4,0.5,0.15,0.55,0.7,0.4,0.5,0.3,0.7,0.8,0.2,0.8,0.85,
0.3,0.7,0.6,0.8,0.45,0.35,0.3,0.4,0.4,0.3,0.2,0.1,0.45,0.55,
0.5,0.7,0.15,0.7,0.7,0.25,0.5,0.3,0.5,0.9,0.45,0.75,0.3,0.8,
0.1,0.4,0.2,0.55,0.85,0.3,0.95,0.1,0.5,0.45,0.45,0.55,0.45,0.7,
0.8,0.1,0.8,0.5,1.0,0.35,0.2,0.5,0.5,0.35,0.0,1.0,0.75,0.55,
0.2,0.6,0.8,0.7,0.5,0.1,0.5,0.15,0.65,0.45,0.9,0.75,0.5,0.7,
0.4,0.4,0.45,0.4,0.4,0.3,0.3,0.45,0.6,0.95,0.25,0.45,0.7,0.4,
0.4,0.65,0.2,0.75,0.55,0.05,0.0,0.15,0.2,0.35,0.15,0.35,0.2,0.6,
0.5,0.4,0.65,0.3,0.25,0.45,0.65,0.35,0.25,0.05,0.45,0.2,0.9,0.3,
0.45,0.6,0.65,0.6,0.75,0.3,0.15,0.35,0.55,0.8,0.8,0.9,0.5,0.4,
0.85,0.35,0.3,0.35,0.75,0.0,0.25,0.95,0.3,0.9,0.2,0.7,0.55,0.15,
0.45,0.15,0.45,0.2,0.35,0.25])



n = data.size

# -------- 参数估计 --------
mu_hat = data.mean()
sigma_hat = data.std(ddof=1)
print(f"n={n}, mu_hat={mu_hat:.6f}, sigma_hat={sigma_hat:.6f}")

# -------- 初始分箱（可调整 k）--------
k = 10
bins = np.linspace(data.min() - 1e-12, data.max() + 1e-12, k+1)  # 包含数据范围
obs_counts, _ = np.histogram(data, bins=bins)
print("initial observed counts:", obs_counts)

# -------- 计算理论概率与期望频数（基于估计的正态分布）--------
# p_i = P(bins[i] <= X < bins[i+1]) under N(mu_hat, sigma_hat^2)
cdf = stats.norm.cdf
p_i = cdf(bins[1:], loc=mu_hat, scale=sigma_hat) - cdf(bins[:-1], loc=mu_hat, scale=sigma_hat)
expected = n * p_i
print("initial expected counts:", expected)

# -------- 若期望频数过小则合并相邻区间（从两端合并）--------
def merge_bins(obs, exp, bins_edges, min_expected=5):
    obs = obs.tolist()
    exp = exp.tolist()
    edges = bins_edges.tolist()
    # merge until all expected >= min_expected and at least 2 bins remain
    i = 0
    while any(e < min_expected for e in exp) and len(exp) > 1:
        # find index of smallest expected
        idx = int(np.argmin(exp))
        if idx == 0:
            # merge with next
            obs[1] += obs[0]
            exp[1] += exp[0]
            obs.pop(0); exp.pop(0); edges.pop(1)  # remove first internal edge
        else:
            # merge with previous
            obs[idx-1] += obs[idx]
            exp[idx-1] += exp[idx]
            obs.pop(idx); exp.pop(idx); edges.pop(idx)
    return np.array(obs), np.array(exp), np.array(edges)

obs_counts_merged, expected_merged, bins_merged = merge_bins(obs_counts, expected, bins)
m = len(obs_counts_merged)
print("merged observed counts:", obs_counts_merged)
print("merged expected counts:", expected_merged)
print("number of bins after merge m =", m)

# -------- 计算卡方统计量与 p 值（自由度 = m-1-r, r=2 estimated params）--------
chi2_stat = ((obs_counts_merged - expected_merged)**2 / expected_merged).sum()
r = 2  # mu and sigma estimated
df = m - 1 - r
if df <= 0:
    raise ValueError("After merging, degrees of freedom <= 0; cannot perform chi-square test. Increase sample or reduce number of estimated parameters.")
p_value = 1 - stats.chi2.cdf(chi2_stat, df)
print(f"Chi-square stat = {chi2_stat:.6f}, df = {df}, p-value = {p_value:.6f}")

# -------- Parametric bootstrap for empirical p-value (based on N(mu_hat,sigma_hat))--------
def parametric_bootstrap_chi2(mu, sigma, bins_edges, obs_counts, B=5000, random_state=123):
    rng = np.random.default_rng(random_state)
    n = obs_counts.sum()
    exp_probs = stats.norm.cdf(bins_edges[1:], loc=mu, scale=sigma) - stats.norm.cdf(bins_edges[:-1], loc=mu, scale=sigma)
    exp_counts = n * exp_probs
    chi2_sim = np.empty(B)
    for i in range(B):
        sim = rng.normal(loc=mu, scale=sigma, size=n)
        sim_counts, _ = np.histogram(sim, bins=bins_edges)
        # If any expected < 1e-8, avoid division by zero by skipping (shouldn't happen with merging)
        chi2_sim[i] = ((sim_counts - exp_counts)**2 / np.where(exp_counts>0, exp_counts, 1e-12)).sum()
    obs_chi2 = ((obs_counts - exp_counts)**2 / np.where(exp_counts>0, exp_counts, 1e-12)).sum()
    p_emp = np.mean(chi2_sim >= obs_chi2)
    return obs_chi2, p_emp

# Use merged bins edges for bootstrap
# reconstruct bins_edges from bins_merged array: it's the remaining edge list
# Note: merge_bins returned edges as original bin edges with some internal edges removed
bins_edges_after = bins_merged
obs_chi2_boot, p_emp = parametric_bootstrap_chi2(mu_hat, sigma_hat, bins_edges_after, obs_counts_merged, B=5000)
print(f"Bootstrap (parametric) chi2 = {obs_chi2_boot:.6f}, empirical p-value = {p_emp:.6f}")