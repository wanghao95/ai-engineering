# sampling_demo.py
# 手动实现大模型采样：Temperature / Top-K / Top-P 对比实验
# 用途：给 Week 11 文章第五节"亲手跑一遍"提供真实数据
#
# 环境准备：
#   pip install torch transformers
# 首次运行会自动下载 GPT-2 模型（约 500MB，下载一次，之后走本地缓存）

# ---------- 0. 国内网络：走 HuggingFace 镜像 ----------
# 直连 huggingface.co 在国内会超时（WinError 10060），换成国内镜像。
# 注意：这行必须在 import transformers 之前！huggingface_hub 在"导入时"就读取
# 这个环境变量，导入之后再设就晚了。
# setdefault 的含义：外面要是已经设过 HF_ENDPOINT，就用外面的；没设才用镜像。
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ---------- 1. 加载模型和分词器 ----------
# GPT-2 是英文模型，不认识中文，所以实验的开头（prompt）用英文。
# 它只有 1.24 亿参数，CPU 也能跑，几秒出结果，不用 GPU。
MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()  # 推理模式，关掉 dropout 等训练时才有的随机行为


# ---------- 2. 手动采样函数 ----------
# 这是全篇的核心：把"采样"这件事，从 logits 开始一步步手写出来。
# 理解每个参数在动哪一步，比直接调 generate() 强一百倍。
def sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.0):
    """
    logits: 一维向量 [词表大小]，是"下一个词每个候选的原始分数"
    返回：采样出来的 token id
    """
    # --- 第 0 步：T=0 是特判，不是真的除以 0 ---
    # 数学上除以 0 无定义，代码里直接降级成"只选分数最高的那个"（贪心）。
    if temperature == 0:
        return torch.argmax(logits).item()

    # --- 第 1 步：temperature，在 softmax 之前除 T ---
    # T<1 分差被拉大（分布变尖），T>1 分差被压平（分布变平）。
    logits = logits / temperature

    # --- 第 2 步：top_k，只保留概率最高的 K 个，其余置为 -inf ---
    # -inf 过 softmax 之后会变成 0，等于"彻底没戏"。
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        threshold = values[-1]          # 第 K 名的分数，当门槛
        logits = torch.where(
            logits < threshold,
            torch.tensor(float("-inf")),
            logits,
        )

    # --- 第 3 步：softmax，把分数压成"加起来等于 1"的百分比 ---
    probs = torch.softmax(logits, dim=-1)

    # --- 第 4 步：top_p（核采样），按累计把握截断 ---
    # 按概率从高到低累加，加到 >= p 就停，其余砍掉，再重新归一化。
    if top_p > 0.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # 标记"累计概率超过 p"的位置
        to_remove = cumulative > top_p
        # 向右移一位：保证至少保留一个候选，避免全被砍光
        to_remove[1:] = to_remove[:-1].clone()
        to_remove[0] = False
        sorted_probs[to_remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()          # 重新归一化
        # 把过滤后的概率按原来的下标放回去
        probs = torch.zeros_like(probs).scatter(0, sorted_indices, sorted_probs)

    # --- 第 5 步：按概率抽签 ---
    # multinomial 就是"按权重抽签"，概率大的更容易被抽中，但不是必然。
    return torch.multinomial(probs, num_samples=1).item()


# ---------- 3. 生成函数：一个字一个字往外接 ----------
def generate(prompt, max_new_tokens=50, temperature=1.0, top_k=0, top_p=0.0):
    # 把 prompt 编码成 token id 序列
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            # 只取"最后一个位置"对下一个词的预测分数
            logits = outputs.logits[0, -1, :]

        next_id = sample_next_token(logits, temperature, top_k, top_p)

        # 把新 token 接到末尾，下一步会"重读"这一整串
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=-1)

        # 遇到结束符就停
        if next_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# ---------- 4. 对比实验 ----------
# 同一个 prompt，不同参数。每组跑两次，观察"同一个参数下答案稳不稳"。
PROMPT = "The future of artificial intelligence is"

CONFIGS = [
    ("贪心 T=0",              dict(temperature=0)),
    ("T=0.7（默认档）",       dict(temperature=0.7)),
    ("T=1.5（放飞）",         dict(temperature=1.5)),
    ("T=0.7 + top_p=0.9",     dict(temperature=0.7, top_p=0.9)),
    ("T=0.7 + top_k=40",      dict(temperature=0.7, top_k=40)),
]

for name, cfg in CONFIGS:
    print(f"\n===== {name} =====")
    for i in range(2):  # 每组跑两次
        text = generate(PROMPT, **cfg)
        print(f"  [{i + 1}] {text}")
