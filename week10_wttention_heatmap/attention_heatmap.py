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

# 1. 加载模型
model_name = "bert-base-chinese"
print(f"正在从镜像站下载/加载模型: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
print("✅ 模型加载成功！")

# 2. 含指代的中文句子（真歧义：「他」可指小明，也可指小李）
sentence = "小明告诉小李他考试通过了"
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(f"分词结果: {tokens}")

# bert-base-chinese 是字级模型，正文每个 token = 一个汉字
body = tokens[1:-1]
assert len(body) == len(sentence) and all(a == b for a, b in zip(body, sentence)), \
    f"分词与句子不对齐: {body}"

def span(word):
    """词 -> 它在句子里的字对应 token 下标"""
    start = sentence.index(word)
    return list(range(start + 1, start + 1 + len(word)))

words = ["小明", "告诉", "小李", "他", "考试", "通过", "了"]
word_to_token_indices = {w: span(w) for w in words}
n_words = len(words)
ta_index = words.index("他")
name_keys = ["小明", "小李"]
content_keys = [w for w in words if w not in name_keys]
ta_token = word_to_token_indices["他"][0]  # 他 是单字 token

# 3. 获取所有层的注意力权重
with torch.no_grad():
    outputs = model(**inputs)

# 4. 选头指标：不是「两个名字质量总和最大」（那会让一个名字独吞、另一个饿死），
#    而是「两个候选都 ≥0.2、漏到别的词上的质量 ≤0.15」时，选两个候选里较小那个最大的头
print("正在遍历 144 个注意力头...")
best = None
for layer_idx in range(12):
    for head_idx in range(12):
        row = outputs.attentions[layer_idx][0, head_idx, ta_token].numpy()  # 他 这一行
        xm = row[word_to_token_indices["小明"]].sum()
        xl = row[word_to_token_indices["小李"]].sum()
        other = sum(row[t] for w in content_keys for t in word_to_token_indices[w])
        if xm >= 0.20 and xl >= 0.20 and other <= 0.15:
            cand = (min(xm, xl), xm + xl, layer_idx, head_idx, xm, xl, other)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand

if best is None:
    raise SystemExit("没有找到符合条件（两候选都≥0.2 且 无关词≤0.15）的头，请换句子或放宽阈值")

_, _, best_layer, best_head, b_xm, b_xl, b_other = best
print(f"✅ 最佳头：Layer {best_layer}, Head {best_head}")
print(f"   「他」行 → 小明 {b_xm:.2f} | 小李 {b_xl:.2f} | 其余词合计 {b_other:.2f}")

# 5. 用最佳头把整句画成 word×word 热力图（query 维度 mean、key 维度 sum，与正文口径一致）
best_attention = outputs.attentions[best_layer][0, best_head].numpy()
best_word_attention = np.zeros((n_words, n_words))
for i, w_i in enumerate(words):
    for j, w_j in enumerate(words):
        sub_matrix = best_attention[np.ix_(word_to_token_indices[w_i], word_to_token_indices[w_j])]
        best_word_attention[i, j] = np.mean(np.sum(sub_matrix, axis=1))

plt.figure(figsize=(8, 7), dpi=150)
sns.heatmap(
    best_word_attention,
    annot=True, fmt=".2f", cmap="Blues",
    xticklabels=words, yticklabels=words,
    cbar_kws={'label': 'Attention Weight'}
)
plt.title(f'Word-level Attention (Layer {best_layer}, Head {best_head}):\n"{sentence}"', fontsize=13, pad=12)
plt.xlabel('Key（被关注的词）', fontsize=12)
plt.ylabel('Query（发起关注的词）', fontsize=12)

# 高亮「他」这一行
plt.gca().add_patch(plt.Rectangle((0, ta_index), n_words, 1, fill=False, edgecolor='red', lw=3))

plt.tight_layout()

# 6. 直接输出到文章目录，覆盖旧的 fig3_real_attention_heatmap.png
save_path = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "articles", "week10", "images", "fig3_real_attention_heatmap.png"))
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, bbox_inches='tight')
print(f"✅ 热力图已保存: {save_path}")
