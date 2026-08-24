# AI 工程化学习计划 — 配套代码

从 0 和 1 到智能涌现，系统学 AI 工程化的配套代码仓库。

每周一篇深度文章 + 可独立运行的 Python 实验，用代码建立直觉。

公众号「AI矩阵涌现」同步更新。

## 目录结构

```
.
├── week2_linear_algebra/            # 线性代数直觉与 NumPy
│   ├── semantic_similarity.py       #   语义相似度 — 词袋向量 + 余弦相似度
│   └── matrix_transform.py          #   矩阵变换可视化 — 拉伸/旋转/剪切
├── week3_calculus/                  # 微积分直觉与梯度下降
│   └── gradient_descent.py          #   手写梯度下降拟合 y=2x+3（不调 PyTorch）
├── week4_probability/               # 概率论与信息论直觉
│   ├── softmax_demo.py              #   Softmax + 交叉熵基础验证（含温度四态对比）
│   └── logistic_regression_iris.py  #   鸢尾花逻辑回归 + MSE vs 交叉熵对比
├── week4_recap/                     # 数学基础总复盘（四块拼图）
├── week6_vector_db/                 # 向量数据库与 HNSW 索引
│   ├── docker-compose.yml           #   pgvector 容器配置
│   └── vector_search_demo.py        #   pgvector + HNSW 全流程：建表→插入→索引→调参
├── week7_data_engineering/          # 数据工程与ML闭环
│   └── titanic_ml_pipeline.py       #   泰坦尼克号：脏数据vs干净数据→特征工程→调参→混淆矩阵
├── week8_neural_network/            # 神经网络与反向传播（50 行 NumPy 手写）
│   ├── numpy_nn_mnist.py           #   50 行神经网络：前向+反向传播，MNIST 训练 20 epochs
│   ├── quick_verify.py             #   快速验证：5 epochs，跑通即确认代码正确
│   ├── parse_arff.py               #   数据解析工具：mnist_784.arff.gz → mnist.npz
│   ├── mnist.npz                   #   MNIST 数据集（已解析，随仓库提供）
│   └── data/openml/                #   MNIST 原始数据（openml 下载）
└── (持续更新...)
```

## 环境要求

- Python ≥ 3.8
- Week 2-4：numpy, matplotlib, scikit-learn, scipy
- Week 5-7：pandas, scikit-learn, seaborn
- Week 8：仅需 numpy（已含在上方公共依赖）；MNIST 数据 `mnist.npz` 已随仓库提供，无需下载
- Week 6：额外需要 Docker（启动 pgvector 容器）

```bash
# 公共依赖（Week 2-4）
pip install numpy matplotlib scikit-learn scipy

# Week 5-7 额外依赖
pip install pandas scikit-learn seaborn

# Week 6 额外依赖
pip install psycopg2-binary sentence-transformers
```

## 运行

每个脚本独立运行，目录下直接执行：

```bash
# Week 2 — 感受余弦相似度
python week2_linear_algebra/semantic_similarity.py

# Week 2 — 看矩阵变换长什么样
python week2_linear_algebra/matrix_transform.py

# Week 3 — 手写梯度下降全过程
python week3_calculus/gradient_descent.py

# Week 4 — 概率论与信息论
python week4_probability/softmax_demo.py
python week4_probability/logistic_regression_iris.py

# Week 6 — pgvector + HNSW 向量搜索（需先启动 Docker）
cd week6_vector_db
docker compose up -d                      # 启动 pgvector 容器
python vector_search_demo.py              # 跑全流程
docker compose down                       # 用完后关掉

# Week 7 — 泰坦尼克号 ML 全流程
cd ../week7_data_engineering
python titanic_ml_pipeline.py             # 脏数据vs干净数据→特征工程→调参

# Week 8 — 50 行 NumPy 神经网络（数据已随仓库提供，开箱即跑）
cd ../week8_neural_network
python quick_verify.py                    # 快速验证：5 epochs，确认代码正确
python numpy_nn_mnist.py                  # 完整训练：20 epochs
```

## 开发环境建议

用 VS Code 直接打开 `ai-engineering/` 目录：
- 安装 Python 扩展，获得语法高亮和一键运行
- 打开 `.py` 文件，点右上角 ▶ 或按 `Ctrl+F5` 直接运行
- Week 6 的 Docker 操作在 VS Code 内置终端里执行

## Week 4 实战工具箱

| 工具包 | 核心函数 | 一句话用法 |
| --- | --- | --- |
| NumPy | `np.exp()` | 计算指数，用于手写 Softmax |
| SciPy | `scipy.special.softmax()` | 一行代码实现稳定 Softmax |
| Scikit-learn | `sklearn.metrics.log_loss()` | 直接计算交叉熵损失 |
| Scikit-learn | `sklearn.linear_model.LogisticRegression()` | 开箱即用的逻辑回归分类器 |

## Week 4 避坑指南

1. **数值稳定性**：Softmax 计算时，如果 logits 很大（如 `[1000, 900]`），直接计算 `exp(z)` 会导致上溢出。代码里用 `exp(x - max(x))` 将所有值平移到 ≤ 0，exp 的输出就在 `(0, 1]` 内，永不上溢——这是工程标准做法。
2. **KL 散度的不对称性**：`KL(P‖Q)` vs `KL(Q‖P)` 选择不同，效果相反——一个倾向覆盖全面但可能模糊，一个倾向精准锐利但可能遗漏模式。VAE 用 KL(Q‖P) 做正则化，GAN 的生成器隐含倾向 KL(P‖Q)。实际应用中需根据具体目标函数谨慎选择。

## 关联内容

每篇文章的深度讲解见公众号「AI矩阵涌现」→「AI 原理与工程实战笔记」合集。

## License

MIT
