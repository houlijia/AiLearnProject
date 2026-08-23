# -*- coding: utf-8 -*-
"""
文本张量表示完整示例
涵盖知识点：
1. one-hot 编码：原理、numpy 实现、优缺点（稀疏 / 维度高 / 无语义）
2. Word2Vec：CBOW 与 Skip-gram 两种模型（gensim 训练）
3. 词嵌入 nn.Embedding：PyTorch 实现，Skip-gram 训练得到稠密低维向量
4. 三种表示方式的对比总结

核心思想：把文本（词/句子）转成计算机能计算的"张量"。
文本张量表示就是给每个词一个数值向量，让模型能对文本做数学运算。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec


# ============================================================
# 一、one-hot 编码
# ============================================================

def demo_one_hot():
    # 1. 构建词表：收集语料里所有不重复的词，排序后给每个词一个下标
    corpus = ["我 喜欢 猫".split(), "我 喜欢 狗".split(), "猫 吃 鱼".split()]
    vocab = sorted({w for s in corpus for w in s})
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    print("词表 vocab：", vocab)
    print("word2idx ：", word2idx)
    print("词表大小 vocab_size =", len(vocab))

    # 2. 单个词的 one-hot 向量：长度 = vocab_size，只有该词下标处为 1，其余为 0
    def one_hot(word):
        vec = np.zeros(len(vocab), dtype=np.float32)
        vec[word2idx[word]] = 1.0
        return vec

    print("\n【单个词的 one-hot 表示】")
    for w in ["猫", "喜欢", "鱼"]:
        print(f"  '{w}' -> {one_hot(w)}")

    # 3. 一句话 -> 张量：把每个词的 one-hot 叠起来，形状 (句子长度, 词表大小)
    sentence = corpus[0]
    tensor = np.stack([one_hot(w) for w in sentence])
    print(f"\n【句子张量】句子: {sentence}")
    print(f"  张量形状: {tensor.shape}  =  (句子长度={len(sentence)}, 词表大小={len(vocab)})")
    print(tensor)

    # 4. one-hot 可以无损还原回原词（可逆）
    print("\n【还原】由 one-hot 向量反查原词：")
    for row in tensor:
        print(f"  {row} -> '{idx2word[int(np.argmax(row))]}'")

    # 5. 缺点演示：任意两个不同词的 one-hot 向量点积都为 0（正交）
    #    所以无法表达"猫"和"狗"比"猫"和"鱼"更相似这种语义关系
    print("\n【缺点 1：无语义】两两向量点积：")
    for a, b in [("猫", "狗"), ("猫", "鱼"), ("我", "喜欢")]:
        print(f"  <{a}, {b}> 点积 = {float(one_hot(a) @ one_hot(b))}  （恒为 0，正交）")

    # 缺点 2：维度高（词表多大向量就有多长）且极度稀疏（每行只有 1 个 1）
    print(f"\n【缺点 2：维度高、稀疏】")
    print(f"  词表大小 {len(vocab)}，每个 one-hot 向量就有 {len(vocab)} 维，且只有 1 个位置非 0")
    print("  真实语料词表可达几十万 -> 向量几十万维，几乎全 0，浪费计算和内存")


# ============================================================
# 二、Word2Vec（gensim 训练 CBOW / Skip-gram）
# ============================================================

def demo_word2vec():
    # 一个小语料：让"猫/狗""国王/女王""跑步/游泳"分别出现在相似上下文中
    corpus = [
        ["我", "喜欢", "猫"], ["我", "喜欢", "狗"],
        ["猫", "吃", "鱼"], ["狗", "吃", "骨头"],
        ["猫", "喜欢", "鱼"], ["狗", "喜欢", "骨头"],
        ["猫", "是", "动物"], ["狗", "是", "动物"],
        ["国王", "统治", "王国"], ["女王", "统治", "王国"],
        ["国王", "住在", "宫殿"], ["女王", "住在", "宫殿"],
        ["国王", "是", "统治者"], ["女王", "是", "统治者"],
        ["我", "喜欢", "跑步"], ["我", "喜欢", "游泳"],
        ["他", "喜欢", "跑步"], ["他", "喜欢", "游泳"],
        ["跑步", "需要", "运动鞋"], ["游泳", "需要", "泳衣"],
    ]

    # ---- 1. CBOW（Continuous Bag-of-Words）：用上下文词预测中心词 ----
    #    sg=0 表示 CBOW，sg=1 表示 Skip-gram
    print("【CBOW】上下文 -> 中心词（sg=0）")
    model_cbow = Word2Vec(
        sentences=corpus,      # 已分好词的句子列表
        vector_size=50,        # 每个词向量维度（稠密、低维，远小于词表）
        window=2,              # 上下文窗口大小
        min_count=1,           # 词频低于该值的词被丢弃
        sg=0,                  # 0=CBOW, 1=Skip-gram
        epochs=100,            # 训练轮数
    )
    print(f"  '猫' 的 50 维词向量：{model_cbow.wv['猫'][:10]}...")
    print("  与 '猫' 最相似的词：", model_cbow.wv.most_similar("猫", topn=3))
    print("  与 '国王' 最相似的词：", model_cbow.wv.most_similar("国王", topn=3))

    # ---- 2. Skip-gram：用中心词预测上下文词 ----
    print("\n【Skip-gram】中心词 -> 上下文（sg=1）")
    model_sg = Word2Vec(
        sentences=corpus, vector_size=50, window=2,
        min_count=1, sg=1, epochs=100,
    )
    print("  与 '猫' 最相似的词：", model_sg.wv.most_similar("猫", topn=3))

    # ---- 3. 词向量之间的语义运算（需要大语料才明显，这里仅演示 API）----
    #    经典例子：vec("国王") - vec("男人") + vec("女人") ≈ vec("女王")
    print("\n【向量运算】most_similar 支持向量加减：")
    result = model_cbow.wv.most_similar(
        positive=["国王", "女王"], negative=["猫"], topn=1
    )
    print("  国王 + 女王 - 猫 ≈", result)

    print("\n说明：上面的小语料只是为了演示流程，相似度结果只是近似。"
          "真实场景需要百万级语料才能得到稳定、有语义的词向量。")


# ============================================================
# 三、词嵌入 nn.Embedding（PyTorch 实现 Skip-gram）
# ============================================================

def build_pairs(corpus, window=1):
    """构造 (中心词, 上下文词) 训练样本对：窗口内每对词互相预测"""
    pairs = []
    for sent in corpus:
        for i in range(len(sent)):
            for j in range(len(sent)):
                if i != j and abs(i - j) <= window:
                    pairs.append((sent[i], sent[j]))  # (中心词, 上下文词)
    return pairs


def cosine(a, b):
    """两个向量的余弦相似度，值越接近 1 越相似"""
    return float((a @ b) / (a.norm() * b.norm()))


def demo_embedding():
    # 1. 和 one-hot 一样先建词表，但这里用"下标"代替高维稀疏向量
    corpus = [
        ["我", "喜欢", "猫"], ["我", "喜欢", "狗"],
        ["猫", "吃", "鱼"], ["狗", "吃", "骨头"],
        ["猫", "喜欢", "鱼"], ["狗", "喜欢", "骨头"],
        ["猫", "是", "动物"], ["狗", "是", "动物"],
        ["国王", "统治", "王国"], ["女王", "统治", "王国"],
        ["国王", "住在", "宫殿"], ["女王", "住在", "宫殿"],
    ]
    vocab = sorted({w for s in corpus for w in s})
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(vocab)
    print("词表：", vocab, f"（共 {vocab_size} 个词）")

    # 2. 构造 (中心词, 上下文词) 样本
    pairs = build_pairs(corpus)
    centers = torch.tensor([word2idx[c] for c, _ in pairs])
    contexts = torch.tensor([word2idx[t] for _, t in pairs])
    print(f"训练样本对数量：{len(pairs)}（Skip-gram：中心词 -> 上下文词）")

    # 3. 定义模型：Embedding 层（查表得到稠密向量）+ 一个线性层做多分类
    EMBED_DIM = 8   # 词向量维度：远小于词表大小，且是稠密的

    class EmbeddingModel(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)  # (词表大小, 向量维度)
            self.fc = nn.Linear(embed_dim, vocab_size)        # 预测上下文词

        def forward(self, x):
            # x: (B,) 中心词下标 -> embed: (B, embed_dim) -> fc: (B, vocab_size)
            return self.fc(self.embed(x))

    torch.manual_seed(42)
    model = EmbeddingModel(vocab_size, EMBED_DIM)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # 4. 训练：让模型学会"用中心词预测上下文词"，Embedding 层在这个过程中被优化
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        logits = model(centers)                    # (N, vocab_size)
        loss = loss_fn(logits, contexts)           # 目标：上下文词下标
        loss.backward()
        optimizer.step()

    print(f"\n训练完成，loss = {loss.item():.4f}")

    # 5. 取出训练好的词向量（embedding 矩阵的每一行就是一个词的向量）
    vectors = model.embed.weight.detach()
    print(f"Embedding 矩阵形状：{vectors.shape} = (词表大小={vocab_size}, 向量维度={EMBED_DIM})")

    def vec(w):
        return vectors[word2idx[w]]

    print("\n【学到的词向量】上下文相似的词，向量也接近：")
    print(f"  '猫' 的向量：{vec('猫').numpy()}")
    print(f"  '狗' 的向量：{vec('狗').numpy()}")

    print("\n【余弦相似度对比】")
    print(f"  sim(猫, 狗)   = {cosine(vec('猫'), vec('狗')):.4f}   <- 上下文相似，应偏高")
    print(f"  sim(国王, 女王) = {cosine(vec('国王'), vec('女王')):.4f}   <- 应偏高")
    print(f"  sim(猫, 国王) = {cosine(vec('猫'), vec('国王')):.4f}   <- 无关词，应偏低")

    # 6. 对比：one-hot 维度 = 词表大小（高维稀疏），embedding 维度 = EMBED_DIM（低维稠密）
    print(f"\n【维度对比】")
    print(f"  one-hot 向量维度：{vocab_size}（高维、稀疏）")
    print(f"  embedding 向量维度：{EMBED_DIM}（低维、稠密）")


# ============================================================
# 四、三种表示方式对比总结
# ============================================================

def demo_summary():
    print("=" * 70)
    print("三种文本张量表示方式对比")
    print("=" * 70)
    rows = [
        ("方式",        "维度",      "稀疏性", "语义信息", "如何得到"),
        ("one-hot",    "= 词表大小", "极稀疏", "无",       "规则生成，无需训练"),
        ("Word2Vec",   "固定低维",   "稠密",   "有",       "无监督预训练（CBOW/Skip-gram）"),
        ("Embedding",  "可自定义",   "稠密",   "有",       "随机初始化 + 随下游任务训练"),
    ]
    for r in rows:
        print("{:<12}{:<14}{:<10}{:<10}{}".format(*r))
    print("=" * 70)
    print("结论：one-hot 最简单但无语义且维度爆炸；"
          "Word2Vec/Embedding 用低维稠密向量表达词义，是现代 NLP 的主流做法。")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("一、one-hot 编码")
    print("=" * 60)
    demo_one_hot()

    print("\n" + "=" * 60)
    print("二、Word2Vec（CBOW / Skip-gram）")
    print("=" * 60)
    demo_word2vec()

    print("\n" + "=" * 60)
    print("三、词嵌入 nn.Embedding（Skip-gram 训练）")
    print("=" * 60)
    demo_embedding()

    print("\n")
    demo_summary()
