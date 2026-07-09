"""
Week 2 线性代数 — 动手实战 9.2：矩阵变换可视化

用矩阵对同一组随机点施加拉伸、旋转、剪切变换，
直观看到"矩阵 = 空间变换"。

运行依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 1. 生成 50 个随机二维点
# ============================================================
points = np.random.randn(50, 2)

# ============================================================
# 2. 定义三个变换矩阵
# ============================================================
SCALE = np.array([[2, 0],    # x 方向拉伸 2 倍
                  [0, 1]])   # y 方向不变

ROTATE = np.array([[0, -1],  # 逆时针旋转 90°
                   [1,  0]])

SHEAR = np.array([[1, 0.5],  # 剪切变换：x 不动，y 加到 x 上
                  [0, 1.0]])

# ============================================================
# 3. 施加变换
# ============================================================
scaled  = points @ SCALE.T
rotated = points @ ROTATE.T
sheared = points @ SHEAR.T

# ============================================================
# 4. 可视化：四宫格对比
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

datasets = [
    (points,  "原始数据"),
    (scaled,  "拉伸 x×2"),
    (rotated, "旋转 90°"),
    (sheared, "剪切变形"),
]

for ax, (data, title) in zip(axes, datasets):
    ax.scatter(data[:, 0], data[:, 1], alpha=0.7, c="steelblue", s=50, edgecolors="white")
    ax.axhline(y=0, color="gray", ls="--", alpha=0.3)
    ax.axvline(x=0, color="gray", ls="--", alpha=0.3)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("matrix_transform_demo.png", dpi=150, bbox_inches="tight")
print("图片已保存: matrix_transform_demo.png")

try:
    plt.show()
except Exception:
    pass
print("\n要点：神经网络每一层的权重矩阵 W，")
print("就是在对数据做同样的事——拉伸、旋转、剪切——")
print("直到一团乱麻的数据在空间中变得可分。")
