# Week 7 动手实战：泰坦尼克号生存预测

## 环境准备

```bash
pip install pandas scikit-learn seaborn
```

## 运行完整流水线

```bash
python titanic_ml_pipeline.py
```

跑完后你会看到 9 步输出：数据加载 → 数据探索 → 脏数据基线 → 数据清洗+特征工程 → 干净数据基线 → 脏vs干净对比 → GridSearchCV调参 → 混淆矩阵+分类报告 → 特征重要性。

核心结论：同一个逻辑回归模型，脏数据 ~72%，干净数据 + 特征工程 ~83%。

---

## 提交到 Kaggle 排行榜（验证 Top 30%）

### 1. 下载测试集

参加 [Kaggle Titanic 竞赛](https://www.kaggle.com/c/titanic)，下载三个文件放到当前目录：

- `train.csv` — 训练集（891条，带 survived 标签）
- `test.csv` — 测试集（418条，不带标签，用来预测）
- `gender_submission.csv` — 官方示例提交格式

### 2. 生成预测

在 `titanic_ml_pipeline.py` 末尾追加提交脚本（或用单独的 `kaggle_submit.py`）：

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# 加载训练集
train = pd.read_csv("train.csv")
train.columns = train.columns.str.lower()

# 加载测试集
test = pd.read_csv("test.csv")
test.columns = test.columns.str.lower()

# 派生 seaborn 风格列（与训练脚本一致）
for df in [train, test]:
    df["deck"] = df["cabin"].str[0]
    embark_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    df["embark_town"] = df["embarked"].map(embark_map)
    df["alone"] = (df["sibsp"] + df["parch"] == 0)
    df["title"] = df["name"].str.extract(r",\s*([^\.]+)\.")
    title_map = {"Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs",
                 "Lady":"Rare", "Sir":"Rare", "Don":"Rare",
                 "Rev":"Rare", "Dr":"Rare", "Major":"Rare",
                 "Col":"Rare", "Capt":"Rare", "Jonkheer":"Rare"}
    df["title"] = df["title"].replace(title_map)

# 特征工程
def engineer(df):
    df = df.copy()
    df["age"] = df.groupby(["pclass", "sex"])["age"].transform(lambda x: x.fillna(x.median()))
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["fare_bin"] = pd.qcut(df["fare"], 4, labels=False, duplicates="drop")
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["embarked"] = LabelEncoder().fit_transform(df["embarked"].astype(str))
    df["title"] = LabelEncoder().fit_transform(df["title"].astype(str))
    df["deck"] = df["deck"].fillna("U")
    df["deck"] = LabelEncoder().fit_transform(df["deck"].astype(str))
    return df

train_fe = engineer(train)
test_fe = engineer(test)

features = ["pclass", "sex", "age", "fare", "fare_bin",
            "sibsp", "parch", "family_size", "is_alone",
            "embarked", "title", "deck"]
X_train = train_fe[features]
y_train = train["survived"]
X_test = test_fe[features]

# 训练 + 预测
model = LogisticRegression(max_iter=2000, C=1)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# 输出提交文件
sub = pd.DataFrame({"PassengerId": test["passengerid"], "Survived": preds})
sub.to_csv("submission.csv", index=False)
print("submission.csv 已生成，上传到 Kaggle 即可。")
```

### 3. 上传

在 Kaggle Titanic 竞赛页面点击 "Submit Predictions"，上传 `submission.csv`。查看 Public Score 和排名。

预期分数：**0.775 ~ 0.794**，对应排名约 **Top 25-35%**（逻辑回归 + 12 个手工特征的体面成绩）。
