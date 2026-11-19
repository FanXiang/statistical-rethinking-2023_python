"""
Lecture 02 - The Garden of Forking Data 预习热身脚本
=====================================================
在看视频之前跑一遍。在仓库根目录运行：

    python prep/lecture02_warmup.py

只依赖 numpy / scipy / matplotlib；第 8 部分需要 pymc（没装会自动跳过）。
图片保存到 prep/figs/。每部分末尾的 TODO 是留给你的小练习。

全程只做三件事：数一数、乘一乘、抽一抽。不需要任何微积分。
"""
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)
rng = np.random.default_rng(2023)

DATA = "WLWWWLWLW"  # 课上的地球仪数据：W=水(water)，L=陆地(land)，6 个 W、3 个 L


# ---------------------------------------------------------------------------
# 1. 分岔花园：把“可能性”真的一条一条数出来
#    假设地球仪只有 4 个面，每个面要么是水 W 要么是陆地 L。
#    5 种猜想：水面有 0、1、2、3、4 个。
#    每次抛地球仪 = 随机摸一个面。观察到 "W L W" 的“路径”有几条？
# ---------------------------------------------------------------------------
def part1_count_paths(obs="WLW"):
    print("[1] 观察到", obs, "时，每种猜想能产生这个结果的路径数：")
    for n_water in range(5):
        faces = ["W"] * n_water + ["L"] * (4 - n_water)  # 例如 n_water=1 -> [W, L, L, L]
        # 暴力枚举所有可能的抛掷序列（4 x 4 x 4 = 64 条路径），数出和观察完全一致的路径
        paths = list(product(range(4), repeat=len(obs)))
        n_match = sum(all(faces[i] == o for i, o in zip(path, obs)) for path in paths)
        # 和“乘法”捷径对比：每一步能匹配的面数相乘
        shortcut = np.prod([faces.count(o) for o in obs])
        print(f"    水面={n_water}/4  路径数={n_match:3d}  (乘法捷径 {shortcut})")
    # TODO: 把 obs 改成 "WLWW"，路径数怎么变？是不是正好等于旧路径数 × 新这一步的匹配面数？


# ---------------------------------------------------------------------------
# 2. 贝叶斯更新 = 把旧计数乘上新数据的计数
#    今天的后验就是明天的先验。数据的先后顺序不影响最终结果。
# ---------------------------------------------------------------------------
def ways(n_water, obs, n_sides=4):
    """一个 n_sides 面、有 n_water 个水面的地球仪，产生 obs 序列的路径数。"""
    n_W = obs.count("W")
    n_L = obs.count("L")
    return n_water**n_W * (n_sides - n_water) ** n_L


def part2_updating():
    counts = np.ones(5)  # 一开始什么都没看到：每种猜想都是 1（“无知”的先验）
    print("[2] 逐个加入数据，计数不断相乘：")
    for i, o in enumerate(DATA, start=1):
        counts = counts * np.array([ways(k, o) for k in range(5)])
        print(f"    第{i}次 {o}: {counts.astype(int)}")
    batch = np.array([ways(k, DATA) for k in range(5)])
    shuffled = "".join(rng.permutation(list(DATA)))
    batch2 = np.array([ways(k, shuffled) for k in range(5)])
    print(f"    一次性算: {batch}   打乱顺序 {shuffled}: {batch2}  -> 一样")
    probs = batch / batch.sum()
    print("    归一化成概率（除以总数，让它们加起来等于 1）:", np.round(probs, 3))
    # TODO: 为什么“水面=0”和“水面=4”的计数变成了 0？只要看到一个什么就足以排除它们？


# ---------------------------------------------------------------------------
# 3. 从 4 个面到无穷多个面：柱状图 -> 平滑曲线（Beta 分布）
# ---------------------------------------------------------------------------
def grid_posterior(n_W, n_L, n_points):
    p = np.linspace(0, 1, n_points)  # 候选的“水的比例”
    unnorm = p**n_W * (1 - p) ** n_L  # 每个候选值的“路径数”（相对大小）
    return p, unnorm / unnorm.sum()


def part3_infinite_globe():
    n_W, n_L = DATA.count("W"), DATA.count("L")
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, n in zip(axs[:3], [5, 11, 21]):
        p, post = grid_posterior(n_W, n_L, n)
        ax.bar(p, post, width=0.8 / (n - 1), color="k")
        ax.set_title(f"{n - 1}-sided globe")
        ax.set_xlabel("proportion water p")
    # 无穷多个面：解析解就是 Beta(W+1, L+1)
    p = np.linspace(0, 1, 200)
    axs[3].plot(p, stats.beta(n_W + 1, n_L + 1).pdf(p), "C1", lw=3)
    axs[3].set_title(f"infinite: Beta({n_W + 1}, {n_L + 1})")
    axs[3].set_xlabel("proportion water p")
    plt.tight_layout()
    plt.savefig(FIGS / "L02_1_infinite_globe.png", dpi=100)
    plt.close()

    # 核对：精细网格算出来的均值 ≈ Beta 分布的均值
    p, post = grid_posterior(n_W, n_L, 1001)
    print(f"[3] 网格后验均值 {np.sum(p * post):.4f}  vs  Beta({n_W+1},{n_L+1}) 均值 "
          f"{stats.beta(n_W + 1, n_L + 1).mean():.4f}  -> figs/L02_1_infinite_globe.png")
    # TODO: 柱子越多每根越矮，为什么？（提示：所有柱子加起来必须等于 1）


# ---------------------------------------------------------------------------
# 4. 先测试再估计：用已知的 p 造数据，看估计器能不能把 p 找回来
# ---------------------------------------------------------------------------
def simulate_globe(p, N):
    return "".join(rng.choice(["W", "L"], size=N, p=[p, 1 - p]))


def part4_validate(true_p=0.7):
    p = np.linspace(0, 1, 200)
    plt.figure(figsize=(6, 3.5))
    print(f"[4] 真实 p = {true_p}，后验均值随样本量变化：")
    for N in [5, 20, 100, 1000]:
        obs = simulate_globe(true_p, N)
        w, l = obs.count("W"), obs.count("L")
        post = stats.beta(w + 1, l + 1)
        plt.plot(p, post.pdf(p), label=f"N={N}")
        lo, hi = post.ppf([0.055, 0.945])
        print(f"    N={N:5d}  W={w:4d}  均值={post.mean():.3f}  89%区间=[{lo:.3f}, {hi:.3f}]")
    plt.axvline(true_p, color="k", ls="--", label="true p")
    plt.xlabel("proportion water p")
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "L02_2_validate.png", dpi=100)
    plt.close()
    print("    -> figs/L02_2_validate.png")
    # TODO: 把 true_p 设成 0 或 1（极端情况），估计器还正常吗？


# ---------------------------------------------------------------------------
# 5. 把“积分问题”变成“数数问题”：从后验里抽样，再对样本做总结
# ---------------------------------------------------------------------------
def part5_sampling():
    post = stats.beta(7, 4)  # 6W 3L + 均匀先验 -> Beta(7, 4)
    samples = post.rvs(size=10_000, random_state=1)
    print("[5] 从后验抽 10000 个 p，然后像处理普通数据一样总结：")
    print(f"    均值 {samples.mean():.3f}   中位数 {np.median(samples):.3f}")
    print(f"    89% 区间 {np.percentile(samples, [5.5, 94.5]).round(3)}")
    print(f"    P(p > 0.5) = {np.mean(samples > 0.5):.3f}   （就是“数一数有多少样本大于 0.5”）")
    # TODO: 用 samples 算 P(0.6 < p < 0.8)。不需要任何公式，一行 np.mean 就行。
    return samples


# ---------------------------------------------------------------------------
# 6. 后验预测：如果再抛 9 次，会看到几个 W？
#    关键：每次预测用一个“不同的” p（来自后验），这样参数的不确定性也被带进了预测
# ---------------------------------------------------------------------------
def part6_posterior_predictive(samples):
    pred_post = rng.binomial(n=9, p=samples)  # 每个 p 样本抛 9 次
    pred_fixed = rng.binomial(n=9, p=0.64, size=len(samples))  # 假装 p 已知 = 0.64
    k = np.arange(10)
    plt.figure(figsize=(6, 3.5))
    plt.bar(k - 0.2, np.bincount(pred_fixed, minlength=10), width=0.4, color="k", label="fixed p = 0.64")
    plt.bar(k + 0.2, np.bincount(pred_post, minlength=10), width=0.4, color="C1", label="posterior predictive")
    plt.xticks(k)
    plt.xlabel("number of W in next 9 tosses")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "L02_3_posterior_predictive.png", dpi=100)
    plt.close()
    print(f"[6] 预测分布的标准差：固定 p {pred_fixed.std():.2f}  vs  后验预测 {pred_post.std():.2f}"
          " -> figs/L02_3_posterior_predictive.png")
    # TODO: 为什么后验预测更“宽”？如果你只用一个点估计做预测，会犯什么错？


# ---------------------------------------------------------------------------
# 7. 用到工作中：A/B 测试（和地球仪是同一个模型！）
#    “水/陆地” 换成 “点击/没点击”，“水的比例 p” 换成 “转化率”
# ---------------------------------------------------------------------------
def part7_ab_test():
    # 假想数据：按钮 A 展示 200 次点击 18 次；按钮 B 展示 180 次点击 25 次
    a = stats.beta(18 + 1, 200 - 18 + 1).rvs(20_000, random_state=2)
    b = stats.beta(25 + 1, 180 - 25 + 1).rvs(20_000, random_state=3)
    diff = b - a
    print("[7] A/B 测试（同一个 Beta 后验）:")
    print(f"    A 转化率 ≈ {a.mean():.3f}，B 转化率 ≈ {b.mean():.3f}")
    print(f"    P(B 比 A 好) = {np.mean(diff > 0):.3f}")
    print(f"    B-A 提升的 89% 区间 = {np.percentile(diff, [5.5, 94.5]).round(3)}")
    # TODO: 把 B 的数据改成 展示 18 次点击 3 次（转化率差不多但样本小很多），P(B 比 A 好) 怎么变？区间呢？


# ---------------------------------------------------------------------------
# 8. （可选，需要 pymc）同一个模型用 PyMC 写一遍，提前熟悉语法
# ---------------------------------------------------------------------------
def part8_pymc():
    try:
        import pymc as pm
    except ImportError:
        print("[8] 未安装 pymc，跳过。安装：pip install pymc")
        return
    with pm.Model():
        p = pm.Uniform("p", 0, 1)  # 先验：p 在 0 到 1 之间都一样可能
        pm.Binomial("W", n=9, p=p, observed=6)  # 似然：抛 9 次，看到 6 个水
        idata = pm.sample(1000, tune=1000, chains=2, progressbar=False, random_seed=1)
    s = np.asarray(idata.posterior["p"].values).reshape(-1)
    print(f"[8] PyMC 后验均值 {s.mean():.3f} sd {s.std():.3f}   vs  解析解 Beta(7,4) "
          f"均值 {stats.beta(7, 4).mean():.3f} sd {stats.beta(7, 4).std():.3f}")


if __name__ == "__main__":
    part1_count_paths()
    part2_updating()
    part3_infinite_globe()
    part4_validate()
    s = part5_sampling()
    part6_posterior_predictive(s)
    part7_ab_test()
    part8_pymc()
