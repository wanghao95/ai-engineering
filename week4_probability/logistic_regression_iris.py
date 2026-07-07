"""
Week 4 概率论+信息论 — 动手实战 Part 2：训练逻辑回归（鸢尾花数据集）

逻辑回归 = 线性变换 + Softmax + 交叉熵，是最简单的神经网络。
不调任何深度学习框架，手动算梯度，同时对比 MSE vs 交叉熵的收敛差距。

运行依赖：numpy, matplotlib, scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ============================================================
# 0. 工具函数
# ============================================================


def softmax(x):
    """任意实数向量 → 概率分布"""
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=-1, keepdims=True)


def cross_entropy(y_true, y_pred):
    """交叉熵损失 H(P, Q) = -Σ P(x) × log Q(x)"""
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / len(y_true)


def mse_loss(y_true, y_pred):
    """均方误差 L = mean((ŷ - y)²)"""
    return np.mean((y_pred - y_true) ** 2)


# ============================================================
# 1. 加载数据
# ============================================================
iris = load_iris()
X = iris.data                                         # [150, 4]
y = iris.target                                       # [150]

# 转 one-hot
y_onehot = np.zeros((len(y), 3))
y_onehot[np.arange(len(y)), y] = 1

# 标准化 + 划分
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.3, random_state=42
)

n_features, n_classes = X_train.shape[1], y_train.shape[2]
print(f"训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")
print(f"特征数: {n_features}, 类别数: {n_classes}")

# ============================================================
# 2. 用交叉熵训练
# ============================================================
print(f"\n{'=' * 50}")
print("训练：交叉熵损失")
print("=" * 50)

W_ce = np.random.randn(n_features, n_classes) * 0.01
b_ce = np.zeros(n_classes)
lr = 0.1
epochs = 500
losses_ce = []

for epoch in range(epochs):
    # 前向
    logits = X_train @ W_ce + b_ce               # [105, 3]
    probs = softmax(logits)

    # 损失
    loss = cross_entropy(y_train, probs)
    losses_ce.append(loss)

    # 反向：∂L/∂logits = (probs - y_true) / n   ← 最漂亮的梯度公式
    grad_logits = (probs - y_train) / len(X_train)
    grad_W = X_train.T @ grad_logits             # [4, 3]
    grad_b = grad_logits.sum(axis=0)             # [3]

    # 更新
    W_ce -= lr * grad_W
    b_ce -= lr * grad_b

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"  epoch {epoch:3d}  loss={loss:.6f}")

# 评估
test_logits = X_test @ W_ce + b_ce
test_probs = softmax(test_logits)
test_preds = np.argmax(test_probs, axis=1)
test_true = np.argmax(y_test, axis=1)
acc_ce = (test_preds == test_true).mean()
print(f"\n交叉熵模型 — 测试准确率: {acc_ce:.2%}")

# ============================================================
# 3. 用 MSE 训练（对比用）
# ============================================================
print(f"\n{'=' * 50}")
print("训练：MSE 损失（对比组）")
print("=" * 50)

W_mse = np.random.randn(n_features, n_classes) * 0.01
b_mse = np.zeros(n_classes)
losses_mse = []

for epoch in range(epochs):
    # 前向
    logits = X_train @ W_mse + b_mse
    probs = softmax(logits)

    # MSE 损失
    loss = mse_loss(y_train, probs)
    losses_mse.append(loss)

    # 反向：MSE + Softmax 梯度（含 ŷ(1-ŷ) 饱和项）
    # grad = 2(ŷ - y) ⊙ ŷ(1-ŷ) / n
    error = (probs - y_train)                  # [105, 3]
    softmax_deriv = probs * (1 - probs)        # ← 这就是饱和的根源
    grad_logits = 2 * error * softmax_deriv / len(X_train)
    grad_W = X_train.T @ grad_logits
    grad_b = grad_logits.sum(axis=0)

    W_mse -= lr * grad_W
    b_mse -= lr * grad_b

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"  epoch {epoch:3d}  loss={loss:.6f}")

# 评估
test_logits = X_test @ W_mse + b_mse
test_probs = softmax(test_logits)
test_preds = np.argmax(test_probs, axis=1)
acc_mse = (test_preds == test_true).mean()
print(f"\nMSE 模型 — 测试准确率: {acc_mse:.2%}")

# ============================================================
# 4. 可视化：三张图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 图1：Loss 下降曲线对比 ---
ax = axes[0]
ax.plot(range(epochs), losses_ce, color="steelblue", linewidth=1.5, label="交叉熵")
ax.plot(range(epochs), losses_mse, color="coral", linewidth=1.5, label="MSE")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss 下降曲线：交叉熵 vs MSE")
ax.legend()
ax.grid(True, alpha=0.2)
# 小窗放大前 50 轮
ax_inset = ax.inset_axes([0.55, 0.5, 0.4, 0.4])
ax_inset.plot(range(50), losses_ce[:50], color="steelblue", linewidth=1.5)
ax_inset.plot(range(50), losses_mse[:50], color="coral", linewidth=1.5)
ax_inset.set_title("前 50 轮放大")
ax_inset.grid(True, alpha=0.2)

# --- 图2：最终准确率对比 ---
ax = axes[1]
bars = ax.bar(["交叉熵", "MSE"], [acc_ce, acc_mse], color=["steelblue", "coral"], width=0.4)
for bar, acc in zip(bars, [acc_ce, acc_mse]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{acc:.2%}", ha="center", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.set_ylabel("测试准确率")
ax.set_title("最终准确率对比")
ax.grid(True, alpha=0.2, axis="y")

# --- 图3：决策边界（用前两个特征投影看） ---
ax = axes[2]
# 只用前两个特征重新训练（便于 2D 可视化）
X_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

W_2d = np.random.randn(2, 3) * 0.01
b_2d = np.zeros(3)
for _ in range(500):
    logits = X_2d @ W_2d + b_2d
    probs = softmax(logits)
    grad_logits = (probs - y_train) / len(X_2d)
    W_2d -= 0.1 * X_2d.T @ grad_logits
    b_2d -= 0.1 * grad_logits.sum(axis=0)

# 画决策区域
h = 0.02
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_logits = np.c_[xx.ravel(), yy.ravel()] @ W_2d + b_2d
Z = np.argmax(softmax(grid_logits), axis=1).reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu", levels=4)

# 画测试数据点
for i, label in enumerate(iris.target_names):
    mask = np.argmax(y_test, axis=1) == i
    ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1],
               label=label, edgecolors="k", linewidths=0.5, s=40)
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_title("决策边界（前两个特征）+ 测试数据")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("logistic_regression_iris.png", dpi=150, bbox_inches="tight")
print("\n图片已保存: logistic_regression_iris.png")

try:
    plt.show()
except Exception:
    pass

# ============================================================
# 5. 总结
# ============================================================
print(f"\n{'=' * 50}")
print(f"交叉熵模型 — 准确率: {acc_ce:.2%}  终损: {losses_ce[-1]:.4f}")
print(f"MSE 模型    — 准确率: {acc_mse:.2%}  终损: {losses_mse[-1]:.4f}")
print(f"交叉熵比 MSE 训练效果好 {acc_ce - acc_mse:.1%} 个点")
print(f"{'=' * 50}")
print("\n原因：MSE + Softmax 存在梯度饱和——模型自信地错时，ŷ(1-ŷ)≈0，梯度消失。")
print("交叉熵数学上消掉了饱和项，梯度 = ŷ - y，干净高效。")
