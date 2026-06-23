"""
Week 3 微积分 — 动手实战：手写梯度下降拟合 y = 2x + 3

不调 PyTorch autograd，手动计算 ∂L/∂w 和 ∂L/∂b，
一步步走到谷底。三张可视化图帮你看到整个过程。

运行依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 0. 生成数据：y = 2x + 3 + 噪声
# ============================================================
N = 100
X = np.random.randn(N) * 2          # 输入 x，均值为 0 标准差 2
TRUE_W, TRUE_B = 2.0, 3.0
Y = TRUE_W * X + TRUE_B + np.random.randn(N) * 0.5  # 加噪声

# ============================================================
# 1. 初始化参数
# ============================================================
w, b = 0.0, 0.0
lr = 0.02           # 学习率
epochs = 100

# 记录训练过程
history = {"w": [w], "b": [b], "loss": []}

# ============================================================
# 2. 手动梯度下降（不调任何深度学习框架）
# ============================================================
for epoch in range(epochs):
    # --- 前向传播：算预测和损失 ---
    y_pred = w * X + b
    errors = y_pred - Y
    loss = float(np.mean(errors ** 2))  # MSE
    history["loss"].append(loss)

    # --- 反向传播：手动推导梯度 ---
    # L = (1/n) * Σ(wx_i + b - y_i)²
    # ∂L/∂w = (2/n) * Σ x_i * (wx_i + b - y_i)
    # ∂L/∂b = (2/n) * Σ (wx_i + b - y_i)
    grad_w = float(2 / N * np.sum(X * errors))
    grad_b = float(2 / N * np.sum(errors))

    # --- 参数更新 ---
    w -= lr * grad_w
    b -= lr * grad_b

    history["w"].append(w)
    history["b"].append(b)

    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"[epoch {epoch:3d}] loss={loss:.6f}  w={w:.4f}  b={b:.4f}")

# ============================================================
# 3. 可视化：三张图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 图1：数据 + 拟合线变化 ---
ax = axes[0]
ax.scatter(X, Y, alpha=0.5, s=30, label="带噪声数据", color="gray")

# 画 4 条拟合线：初始 → 中间 → 中间 → 最终
xs = np.linspace(X.min(), X.max(), 100)
for step, color in [(0, "red"), (5, "orange"), (20, "blue"), (epochs, "green")]:
    wi = history["w"][step]
    bi = history["b"][step]
    label = f"epoch={step}" if step == 0 else (f"epoch={step}" if step < epochs else "epoch=100（最终）")
    ax.plot(xs, wi * xs + bi, color=color, alpha=0.7, label=label)

# 真实线
ax.plot(xs, TRUE_W * xs + TRUE_B, "k--", alpha=0.3, label="真实 y=2x+3")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("拟合线变化")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# --- 图2：Loss 下降曲线 ---
ax = axes[1]
ax.plot(range(epochs), history["loss"], color="steelblue")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("损失下降曲线")
ax.grid(True, alpha=0.2)

# --- 图3：损失曲面 + 梯度下降路径 ---
ax = axes[2]
# 创建一个 w-b 网格
w_grid = np.linspace(-1, 4, 80)
b_grid = np.linspace(-1, 6, 80)
W_mesh, B_mesh = np.meshgrid(w_grid, b_grid)

# 算每个 (w, b) 的损失
loss_surface = np.zeros_like(W_mesh)
for i in range(len(w_grid)):
    for j in range(len(b_grid)):
        pred = W_mesh[j, i] * X + B_mesh[j, i]
        loss_surface[j, i] = float(np.mean((pred - Y) ** 2))

# 画等高线
contour = ax.contour(W_mesh, B_mesh, loss_surface, levels=20, colors="gray",
                     alpha=0.4, linewidths=0.5)
ax.contourf(W_mesh, B_mesh, loss_surface, levels=20, cmap="YlOrRd", alpha=0.6)

# 画梯度下降路径
path_w = history["w"]
path_b = history["b"]
ax.plot(path_w, path_b, "b-", alpha=0.6, linewidth=1.5)
ax.plot(path_w[0], path_b[0], "ro", markersize=8, label="起点 (w=0, b=0)")
ax.plot(TRUE_W, TRUE_B, "g*", markersize=12, label=f"终点≈(w={w:.2f}, b={b:.2f})")

ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_title("损失曲面等高线 + 下降路径")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("gradient_descent_demo.png", dpi=150, bbox_inches="tight")
print("图片已保存: gradient_descent_demo.png")

try:
    plt.show()
except Exception:
    pass

# ============================================================
# 4. 总结
# ============================================================
print(f"\n{'='*50}")
print(f"真实参数: w={TRUE_W}, b={TRUE_B}")
print(f"拟合结果: w={w:.4f}, b={b:.4f}")
print(f"最终 Loss: {history['loss'][-1]:.6f}")
print(f"{'='*50}")
