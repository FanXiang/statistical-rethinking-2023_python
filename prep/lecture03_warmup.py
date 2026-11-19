"""
Lecture 03 - Geocentric Models 预习热身脚本
================================================
在看视频之前跑一遍，先对每个概念有“手感”。在仓库根目录运行：

    python prep/lecture03_warmup.py

只依赖 numpy / scipy / pandas / matplotlib；第 5 部分需要 pymc（没装会自动跳过）。
图片保存到 prep/figs/。每部分末尾的 TODO 是留给你的小练习。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)
rng = np.random.default_rng(2023)


# ---------------------------------------------------------------------------
# 1. 为什么是正态分布？—— 很多小波动相加会趋向正态（生成论证）
# ---------------------------------------------------------------------------
def part1_random_walk(n_people=10_000, n_steps=16):
    steps = rng.choice([-1, 1], size=(n_people, n_steps)) * rng.uniform(0, 1, size=(n_people, n_steps))
    pos = steps.sum(axis=1)
    xs = np.linspace(pos.min(), pos.max(), 200)

    plt.figure(figsize=(5, 3.5))
    plt.hist(pos, bins=50, density=True, alpha=0.5, label="simulated positions")
    plt.plot(xs, stats.norm(pos.mean(), pos.std()).pdf(xs), "k", label="Normal(mean, sd)")
    plt.title(f"Sum of {n_steps} random steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "1_random_walk.png", dpi=100)
    plt.close()
    print(f"[1] 位置均值 {pos.mean():.2f}, 标准差 {pos.std():.2f}  -> figs/1_random_walk.png")
    # TODO: 把 n_steps 改成 2、4、8，看看多少步之后就“很像正态”了。
    # TODO: 把相加改成相乘：np.prod(1 + 0.1*steps, axis=1)，结果还是正态吗？取 log 呢？


# ---------------------------------------------------------------------------
# 2. 写一个生成模型：H -> W <- U
#    W_i = beta * H_i + U_i,  U_i ~ Normal(0, sigma),  H_i ~ Uniform(130, 170)
# ---------------------------------------------------------------------------
def sim_weight(H, beta, sigma):
    U = rng.normal(0, sigma, size=len(H))
    return beta * H + U


def part2_generative():
    H = rng.uniform(130, 170, size=200)
    W = sim_weight(H, beta=0.5, sigma=5)
    plt.figure(figsize=(5, 3.5))
    plt.scatter(H, W, facecolors="none", edgecolors="C0")
    plt.xlabel("height H (cm)")
    plt.ylabel("weight W (kg)")
    plt.title("Synthetic data from the generative model")
    plt.tight_layout()
    plt.savefig(FIGS / "2_generative.png", dpi=100)
    plt.close()
    print("[2] 合成数据已生成 -> figs/2_generative.png")
    # TODO: 问自己：为什么老师坚持“先写生成模型，再写估计器”？
    return H, W


# ---------------------------------------------------------------------------
# 3. 先验预测模拟：先验画出来的“线”合不合理？
#    对比 仓库 notebook 的先验 与 课上（中心化身高）的先验
# ---------------------------------------------------------------------------
def part3_prior_predictive(n=50):
    Hseq = np.linspace(130, 170, 30)
    Hbar = 150
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    setups = [
        ("notebook: a~N(0,10), b~U(0,1), raw H",
         lambda: (rng.normal(0, 10), rng.uniform(0, 1), 0.0)),
        ("vague: a~N(60,10), b~N(0,10), H-Hbar",
         lambda: (rng.normal(60, 10), rng.normal(0, 10), Hbar)),
        ("lecture: a~N(60,10), b~LogNormal(0,1), H-Hbar",
         lambda: (rng.normal(60, 10), rng.lognormal(0, 1), Hbar)),
    ]
    for ax, (title, draw) in zip(axs, setups):
        for _ in range(n):
            a, b, c = draw()
            ax.plot(Hseq, a + b * (Hseq - c), color="C0", alpha=0.4)
        ax.axhline(0, ls="--", color="k")
        ax.set_ylim(-100, 250)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("height (cm)")
    axs[0].set_ylabel("E[W|H] (kg)")
    plt.tight_layout()
    plt.savefig(FIGS / "3_prior_predictive.png", dpi=100)
    plt.close()
    print("[3] 先验预测线对比 -> figs/3_prior_predictive.png")
    # TODO: 哪组先验允许“体重为负”或“越高越轻”？哪组符合常识但又不过分自信？


# ---------------------------------------------------------------------------
# 4. 网格近似（Lecture 2 的老朋友）：后验 ∝ 似然 × 先验
#    这里固定 sigma，只在 (alpha, beta) 网格上算后验，观察数据量变化的影响
# ---------------------------------------------------------------------------
def grid_posterior(x, y, a_grid, b_grid, sigma=0.5):
    A, B = np.meshgrid(a_grid, b_grid, indexing="ij")
    log_post = stats.norm(0, 1).logpdf(A) + stats.norm(0, 1).logpdf(B)
    for xi, yi in zip(x, y):
        log_post = log_post + stats.norm(A + B * xi, sigma).logpdf(yi)
    post = np.exp(log_post - log_post.max())
    return A, B, post / post.sum()


def part4_grid():
    x = rng.normal(size=32)
    y = -1 + 0.5 * x + rng.normal(0, 0.5, size=32)
    grid = np.linspace(-2, 2, 150)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, n in zip(axs, [0, 2, 8, 32]):
        A, B, post = grid_posterior(x[:n], y[:n], grid, grid)
        ax.contourf(A, B, post, levels=15, cmap="Blues")
        ax.plot(-1, 0.5, "r+", ms=12, mew=2)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("intercept a")
    axs[0].set_ylabel("slope b")
    plt.tight_layout()
    plt.savefig(FIGS / "4_grid_posterior.png", dpi=100)
    plt.close()
    print("[4] 后验随样本量收缩 -> figs/4_grid_posterior.png（红十字=真实值）")
    # TODO: 再加 sigma 成为第三个参数，网格点数会变成多少？这就是为什么后面要用 quap / MCMC。


# ---------------------------------------------------------------------------
# 5. （可选，需要 pymc）用模拟数据验证估计器，再用于真实 Howell 成人数据
#    使用课上的中心化写法：mu = a + b * (H - Hbar)
# ---------------------------------------------------------------------------
def fit_weight_model(H, W, draws=1000):
    import pymc as pm

    Hbar = H.mean()
    with pm.Model():
        a = pm.Normal("a", 60, 10)
        b = pm.LogNormal("b", 0, 1)
        sigma = pm.Uniform("sigma", 0, 10)
        mu = a + b * (H - Hbar)
        pm.Normal("W", mu, sigma, observed=W)
        idata = pm.sample(draws=draws, tune=1000, chains=2, progressbar=False, random_seed=1)
    post = idata.posterior
    return {k: (float(post[k].mean()), float(post[k].std())) for k in ["a", "b", "sigma"]}, Hbar, idata


def part5_pymc():
    try:
        import pymc  # noqa: F401
    except ImportError:
        print("[5] 未安装 pymc，跳过。安装：pip install pymc")
        return
    # 5a. 用已知参数造数据，看能否把参数“找回来”
    H = rng.uniform(130, 170, size=100)
    W = 70 + 0.5 * (H - H.mean()) + rng.normal(0, 5, size=100)
    est, _, _ = fit_weight_model(H, W)
    print("[5a] 模拟数据（真值 a=70, b=0.5, sigma=5）的后验均值±sd：")
    for k, (m, s) in est.items():
        print(f"      {k:5s} {m:7.3f} ± {s:.3f}")

    # 5b. 真实数据：Howell1 成年人
    d = pd.read_csv(ROOT / "data" / "Howell1.csv", sep=";")
    d = d[d.age >= 18]
    est, Hbar, idata = fit_weight_model(d.height.values, d.weight.values)
    print(f"[5b] Howell 成人 (N={len(d)}, Hbar={Hbar:.1f}) 后验均值±sd：")
    for k, (m, s) in est.items():
        print(f"      {k:5s} {m:7.3f} ± {s:.3f}")

    # 5c. “Obey the law”：不要孤立解读参数，把后验推成预测
    # 把 (chain, draw) 摊平成一维样本；兼容 PyMC 5（InferenceData）和 PyMC 6（DataTree）
    a_s, b_s, s_s = (np.asarray(idata.posterior[k].values).reshape(-1) for k in ["a", "b", "sigma"])
    Hseq = np.linspace(135, 180, 50)
    mu = a_s[None, :] + b_s[None, :] * (Hseq[:, None] - Hbar)
    Wsim = rng.normal(mu, s_s[None, :])
    plt.figure(figsize=(5, 4))
    plt.scatter(d.height, d.weight, facecolors="none", edgecolors="C0", alpha=0.6)
    for i in range(0, mu.shape[1], 100):
        plt.plot(Hseq, mu[:, i], color="k", alpha=0.15)
    lo, hi = np.percentile(Wsim, [5.5, 94.5], axis=1)
    plt.fill_between(Hseq, lo, hi, color="k", alpha=0.1, label="89% predictive interval")
    plt.xlabel("height (cm)")
    plt.ylabel("weight (kg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "5_howell_posterior_predictive.png", dpi=100)
    plt.close()
    print("[5c] 后验线 + 预测区间 -> figs/5_howell_posterior_predictive.png")
    # TODO: a 的含义是什么？（提示：H = Hbar 时的平均体重）如果不中心化，a 又是什么意思？


if __name__ == "__main__":
    part1_random_walk()
    part2_generative()
    part3_prior_predictive()
    part4_grid()
    part5_pymc()
