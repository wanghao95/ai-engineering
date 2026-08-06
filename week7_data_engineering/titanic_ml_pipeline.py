"""
Week 7 动手实战：数据工程与ML闭环 — 泰坦尼克号生存预测

依赖安装：
    pip install pandas scikit-learn seaborn

运行：
    python titanic_ml_pipeline.py

数据来源：seaborn 内置 Titanic 数据集（自动下载），也可从 Kaggle 下载 train.csv
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 1. 加载数据
# ============================================================

print("=" * 60)
print("1. 加载泰坦尼克号数据集（seaborn 内置）")
print("=" * 60)

import seaborn as sns

raw = sns.load_dataset("titanic")
print(f"数据维度: {raw.shape}")  # (891, 15)
print(f"\n前5行:")
print(raw.head())
print(f"\n各列类型与缺失值:")
print(raw.info())

# ============================================================
# 2. 数据探索
# ============================================================

print("\n" + "=" * 60)
print("2. 数据探索：缺失值与异常值")
print("=" * 60)

print(f"\n缺失值统计:")
print(raw.isnull().sum()[raw.isnull().sum() > 0])

print(f"\n年龄分布:")
print(raw["age"].describe())

print(f"\n票价分布:")
print(raw["fare"].describe())

print(f"\n生存率概况: {raw['survived'].mean():.1%}")

# ============================================================
# 3. 脏数据基线（最低限度处理）
# ============================================================

print("\n" + "=" * 60)
print("3. 脏数据基线模型")
print("=" * 60)


def make_dirty_model(df):
    """最低限度的数据处理：只填缺失值、删无用列"""
    df = df.copy()

    # 目标变量
    y = df["survived"].astype(int)

    # 只保留数值列 + 简单编码
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    df = df[features].copy()

    # 粗暴填充
    df["age"] = df["age"].fillna(df["age"].mean())
    df["fare"] = df["fare"].fillna(df["fare"].mean())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # 类别编码
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["embarked"] = LabelEncoder().fit_transform(df["embarked"].astype(str))

    return df, y


X_dirty, y_dirty = make_dirty_model(raw)

model_dirty = LogisticRegression(max_iter=1000)
scores_dirty = cross_val_score(model_dirty, X_dirty, y_dirty, cv=5, scoring="accuracy")
print(f"脏数据 5折交叉验证准确率: {scores_dirty.mean():.3f} (+/- {scores_dirty.std():.3f})")
print(f"各折: {[f'{s:.3f}' for s in scores_dirty]}")

# ============================================================
# 4. 数据清洗 + 特征工程
# ============================================================

print("\n" + "=" * 60)
print("4. 数据清洗 + 特征工程")
print("=" * 60)


def make_clean_model(df):
    """完整的数据清洗和特征工程"""
    df = df.copy()

    # --- 4.1 从姓名提取称谓 ---
    df["title"] = df["name"].str.extract(r",\s*([^\.]+)\.")

    # 归并罕见称谓
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Rare", "Sir": "Rare", "Don": "Rare",
        "Rev": "Rare", "Dr": "Rare", "Major": "Rare",
        "Col": "Rare", "Capt": "Rare", "Jonkheer": "Rare",
        "the Countess": "Rare",
    }
    df["title"] = df["title"].replace(title_map)
    print(f"称谓分布:\n{df['title'].value_counts().to_string()}")

    # --- 4.2 按分组填充年龄（Pclass + Sex） ---
    before_age_null = df["age"].isnull().sum()
    df["age"] = df.groupby(["pclass", "sex"])["age"].transform(
        lambda x: x.fillna(x.median())
    )
    print(f"\n年龄缺失值: {before_age_null} → {df['age'].isnull().sum()} (按Pclass+Sex分组填中位数)")

    # --- 4.3 填充 Embarked ---
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # --- 4.4 从 Cabin 提取 deck ---
    df["deck"] = df["deck"].fillna("U")  # seaframe uses 'deck' column

    # --- 4.5 家庭特征 ---
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    print(f"家庭规模分布:\n{df['family_size'].value_counts().sort_index().to_string()}")

    # --- 4.6 票价分箱 ---
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["fare_bin"] = pd.qcut(df["fare"], 4, labels=False, duplicates="drop")

    # --- 4.7 类别编码 ---
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["embarked"] = LabelEncoder().fit_transform(df["embarked"].astype(str))
    df["title"] = LabelEncoder().fit_transform(df["title"].astype(str))
    df["deck"] = LabelEncoder().fit_transform(df["deck"].astype(str))
    df["embark_town"] = LabelEncoder().fit_transform(df["embark_town"].astype(str))
    df["alone"] = df["alone"].astype(int)  # seaframe uses 'alone' column

    # --- 4.8 选择特征 ---
    feature_cols = [
        "pclass", "sex", "age", "fare", "fare_bin",
        "sibsp", "parch", "family_size", "is_alone",
        "embarked", "title", "deck",
    ]

    y = df["survived"].astype(int)
    X = df[feature_cols].copy()

    return X, y, df


X_clean, y_clean, df_fe = make_clean_model(raw)

# ============================================================
# 5. 干净数据基线
# ============================================================

print("\n" + "=" * 60)
print("5. 干净数据 + 特征工程 基线模型")
print("=" * 60)

model_clean = LogisticRegression(max_iter=1000)
scores_clean = cross_val_score(model_clean, X_clean, y_clean, cv=5, scoring="accuracy")
print(f"干净数据 5折交叉验证准确率: {scores_clean.mean():.3f} (+/- {scores_clean.std():.3f})")
print(f"各折: {[f'{s:.3f}' for s in scores_clean]}")

# ============================================================
# 6. 对比总结
# ============================================================

print("\n" + "=" * 60)
print("6. 脏数据 vs 干净数据 对比")
print("=" * 60)

print(f"""
{'指标':<25} {'脏数据':>10} {'干净数据':>10} {'提升':>10}
{'-' * 55}
{'平均准确率':<25} {scores_dirty.mean():>10.3f} {scores_clean.mean():>10.3f} {scores_clean.mean() - scores_dirty.mean():>+10.3f}
{'标准差':<25} {scores_dirty.std():>10.3f} {scores_clean.std():>10.3f}
{'特征数量':<25} {X_dirty.shape[1]:>10} {X_clean.shape[1]:>10}
""")

# ============================================================
# 7. 调参
# ============================================================

print("=" * 60)
print("7. 调参：GridSearchCV 搜索最佳正则化强度 C")
print("=" * 60)

param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    param_grid,
    cv=5,
    scoring="accuracy",
    return_train_score=True,
)
grid.fit(X_clean, y_clean)

print(f"最佳 C: {grid.best_params_['C']}")
print(f"最佳交叉验证准确率: {grid.best_score_:.3f}")

print(f"\n各C值的表现:")
for i, c in enumerate(param_grid["C"]):
    train = grid.cv_results_["mean_train_score"][i]
    test = grid.cv_results_["mean_test_score"][i]
    marker = " ←" if c == grid.best_params_["C"] else ""
    print(f"  C={c:<7} 训练={train:.4f}  验证={test:.4f}{marker}")

# ============================================================
# 8. 混淆矩阵（用最佳模型单次划分评估）
# ============================================================

print("\n" + "=" * 60)
print("8. 混淆矩阵与分类报告")
print("=" * 60)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

best_model = LogisticRegression(max_iter=2000, C=grid.best_params_["C"])
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(f"\n混淆矩阵:")
print(f"              预测遇难  预测生还")
print(f"  实际遇难       {cm[0][0]:>4}      {cm[0][1]:>4}")
print(f"  实际生还       {cm[1][0]:>4}      {cm[1][1]:>4}")

print(f"\n分类报告:")
print(classification_report(
    y_test, y_pred,
    target_names=["遇难", "生还"],
    zero_division=0,
))

# ============================================================
# 9. 特征重要性（逻辑回归系数）
# ============================================================

print("=" * 60)
print("9. 特征重要性（逻辑回归系数绝对值）")
print("=" * 60)

coef_df = pd.DataFrame({
    "特征": X_clean.columns,
    "系数": best_model.coef_[0],
    "绝对值": np.abs(best_model.coef_[0]),
}).sort_values("绝对值", ascending=False)

print(coef_df.to_string(index=False))

print("\n解读: 绝对值越大，该特征对'生还概率'的影响越大。正值为正向影响（女性/高票价→生还），负值为负向影响（男性/三等舱→遇难）。")
