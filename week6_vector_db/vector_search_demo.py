"""
Week 6 动手实战：pgvector + HNSW 向量搜索全流程

依赖安装：
    pip install psycopg2-binary sentence-transformers

环境启动：
    docker compose up -d

环境销毁：
    docker compose down -v
"""

import time
import psycopg2
from sentence_transformers import SentenceTransformer

# ————————————————————————————————————————————————————————————
# 1. 加载 Embedding 模型 + 连接 pgvector
# ————————————————————————————————————————————————————————————

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="postgres",
    password="postgres",
    dbname="postgres",
)
cur = conn.cursor()

# ————————————————————————————————————————————————————————————
# 2. 建表
# ————————————————————————————————————————————————————————————

cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("DROP TABLE IF EXISTS documents")

cur.execute("""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        title TEXT,
        content TEXT,
        embedding vector(384)
    )
""")

# ————————————————————————————————————————————————————————————
# 3. 插入 1000 条文本向量
# ————————————————————————————————————————————————————————————

print("Generating 1000 embeddings...")
documents = [
    {"title": f"文档{i}", "content": f"这是第{i}篇关于AI和向量数据库的文档内容"}
    for i in range(1000)
]

texts = [doc["content"] for doc in documents]
embeddings = model.encode(texts)

for doc, emb in zip(documents, embeddings):
    cur.execute(
        "INSERT INTO documents (title, content, embedding) VALUES (%s, %s, %s)",
        (doc["title"], doc["content"], emb.tolist()),
    )
conn.commit()
print(f"  → {len(documents)} 条插入完成")

# ————————————————————————————————————————————————————————————
# 4. 暴力搜索（无索引）
# ————————————————————————————————————————————————————————————

query_text = "什么是向量数据库"
query_emb = model.encode([query_text])[0].tolist()

start = time.time()
cur.execute(
    """
    SELECT id, title,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_emb, query_emb),
)
results = cur.fetchall()
elapsed = (time.time() - start) * 1000

print(f"\n暴力搜索（无索引）耗时: {elapsed:.1f}ms")
for r in results:
    print(f"  ID={r[0]}, title={r[1]}, similarity={r[2]:.4f}")

# ————————————————————————————————————————————————————————————
# 5. 创建 HNSW 索引
# ————————————————————————————————————————————————————————————

print("\nCreating HNSW index...")
start = time.time()
cur.execute(
    """
    CREATE INDEX documents_embedding_hnsw
        ON documents
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """
)
conn.commit()
idx_elapsed = time.time() - start
print(f"  → 索引创建完成，耗时 {idx_elapsed:.1f}s")

# ————————————————————————————————————————————————————————————
# 6. ef_search 参数对比
# ————————————————————————————————————————————————————————————

print(f"\n{'ef_search':>10}  {'耗时(ms)':>10}  {'Top1相似度':>12}")
print("-" * 38)


def search_with_ef(ef: int, query_emb: list, limit: int = 5):
    cur.execute(f"SET hnsw.ef_search = {ef}")
    start = time.time()
    cur.execute(
        """
        SELECT id, title,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_emb, query_emb, limit),
    )
    results = cur.fetchall()
    elapsed = (time.time() - start) * 1000
    return results, elapsed


for ef in [20, 40, 100, 200]:
    results, ms = search_with_ef(ef, query_emb)
    print(f"{ef:>10}  {ms:>8.1f}ms  {results[0][2]:>12.4f}")

print("\n注意: 仅 1000 条数据，HNSW 索引优势不明显。")
print("数据量到 10 万+级别后，ef_search 对召回率/延迟的权衡才拉开差距。")

# ————————————————————————————————————————————————————————————
# 7. 清理
# ————————————————————————————————————————————————————————————

cur.execute("DROP TABLE IF EXISTS documents")
conn.commit()
cur.close()
conn.close()
print("\nDone.")
