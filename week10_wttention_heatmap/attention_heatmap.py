import os
# 必须在导入 transformers 之前设置环境变量，指向国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 2. 加载模型（现在会自动走 hf-mirror.com，速度飞快）
model_name = "bert-base-chinese"
print(f"正在从镜像站下载/加载模型: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
print("✅ 模型加载成功！")

# 3. 准备目标句子
sentence = "小明告诉小李他考试通过了"
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 4. 前向传播，获取注意力权重
with torch.no_grad():
    outputs = model(**inputs)

# 取最后一层，并对所有注意力头求平均
last_layer_attentions = outputs.attentions[-1]
avg_attention = last_layer_attentions.mean(dim=0).mean(dim=0).numpy()

# 5. 数据清洗与可视化
tokens_to_plot = tokens[1:-1] # 去掉 [CLS] 和 [SEP]
attention_to_plot = avg_attention[1:-1, 1:-1]

plt.figure(figsize=(8, 6), dpi=150)
sns.heatmap(
    attention_to_plot,
    annot=True, fmt=".2f", cmap="Blues",
    xticklabels=tokens_to_plot, yticklabels=tokens_to_plot,
    cbar_kws={'label': 'Attention Weight'}
)

plt.title(f'Self-Attention Weights: "{sentence}"', fontsize=14, pad=15)
plt.xlabel('Key (被关注的词)', fontsize=12)
plt.ylabel('Query (发起关注的词)', fontsize=12)

# 高亮 "他" 这一行
try:
    ta_index = tokens_to_plot.index("他")
    plt.gca().add_patch(plt.Rectangle((0, ta_index), len(tokens_to_plot), 1, fill=False, edgecolor='red', lw=3))
except ValueError:
    pass

plt.tight_layout()
plt.savefig("fig3_real_attention_heatmap.png", bbox_inches='tight')
print("✅ 热力图已成功保存为 fig3_real_attention_heatmap.png，快去文件夹看看吧！")