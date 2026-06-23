# AI 工程化 37 周学习计划 — 配套代码

从 0 和 1 到智能涌现，37 周系统学 AI 工程化的配套代码仓库。

每周一篇深度文章 + 可独立运行的 Python 实验，用代码建立直觉。

公众号「AI矩阵涌现」同步更新。

## 目录结构

```
.
├── week2_linear_algebra/        # 线性代数直觉与 NumPy
│   ├── semantic_similarity.py   #   语义相似度 — 词袋向量 + 余弦相似度
│   └── matrix_transform.py      #   矩阵变换可视化 — 拉伸/旋转/剪切
├── week3_calculus/              # 微积分直觉与梯度下降
│   └── gradient_descent.py      #   手写梯度下降拟合 y=2x+3（不调 PyTorch）
└── (每周更新...)
```

## 环境要求

- Python 3.8+
- numpy
- matplotlib

```bash
pip install numpy matplotlib
```

## 运行

每个脚本独立运行：

```bash
# Week 2 — 感受余弦相似度
python week2_linear_algebra/semantic_similarity.py

# Week 2 — 看矩阵变换长什么样
python week2_linear_algebra/matrix_transform.py

# Week 3 — 手写梯度下降全过程
python week3_calculus/gradient_descent.py
```

## 关联内容

每篇文章的深度讲解见公众号「AI矩阵涌现」→「AI 原理与工程实战笔记」合集。

## License

MIT
