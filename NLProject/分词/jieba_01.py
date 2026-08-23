# -*- coding: utf-8 -*-
"""
jieba 分词完整示例
涵盖知识点：
1. 三种分词模式：精确模式、全模式、搜索引擎模式
2. HMM 新词发现开关
3. 自定义词典：load_userdict / add_word / del_word / suggest_freq
4. 词性标注（jieba.posseg）
5. 关键词提取：TF-IDF 与 TextRank
6. Tokenize：返回词语在原文中的起止位置
7. 并行分词
8. 延迟加载机制
9. 词频统计与停用词过滤
10. 词典导出
"""

import jieba
import jieba.posseg as pseg
import jieba.analyse
from collections import Counter


# ============================================================
# 一、三种分词模式
# ============================================================

def demo_three_modes():
    text = "我来到北京清华大学"

    print("【1】精确模式（默认）—— 最准确，适合文本分析")
    print(list(jieba.cut(text)))                      # 默认 cut_all=False, HMM=True
    # ['我', '来到', '北京', '清华大学']

    print("\n【2】全模式 —— 把所有可能词都切出来，速度最快但存在歧义")
    print(list(jieba.cut(text, cut_all=True)))
    # ['我', '来到', '北京', '清华', '清华大学', '大学']

    print("\n【3】搜索引擎模式 —— 在精确模式基础上再切长词，提高召回率")
    print(jieba.cut_for_search(text))                 # 返回 generator
    print(list(jieba.cut_for_search(text)))
    # ['清华', '清华大学', '我', '来到', '北京']

    # lcut 系列直接返回 list，与 cut / cut_for_search 对应
    print("\nlcut 直接返回 list：", jieba.lcut(text))


# ============================================================
# 二、HMM 新词发现
# ============================================================

def demo_hmm():
    text = "他来到了网易杭研大厦"   # "杭研" 不在词典中

    print("【HMM=False】不使用 HMM，新词无法识别：")
    print(jieba.lcut(text, HMM=False))
    # ['他', '来到', '了', '网易', '杭', '研', '大厦']

    print("\n【HMM=True（默认）】使用 HMM 模型识别新词：")
    print(jieba.lcut(text, HMM=True))
    # ['他', '来到', '了', '网易', '杭研', '大厦']


# ============================================================
# 三、自定义词典
# ============================================================

def demo_custom_dict():
    text = "李小福是创新办主任也是云计算方面的专家"

    print("未加载自定义词典：")
    print(jieba.lcut(text))
    # ['李小福', '是', '创新', '办', '主任', '也', '是', '云计算', '方面', '的', '专家']

    # ---- 方式 1：动态添加单个词 ----
    # add_word(word, freq=None, tag=None)
    jieba.add_word("创新办")              # 只加词
    jieba.add_word("云计算", freq=100, tag="n")

    # ---- 方式 2：删除某个词 ----
    jieba.del_word("创新办")

    # ---- 方式 3：加载外部词典文件 ----
    # 文件格式：词语 词频(可省) 词性(可省)，用空格分隔，每行一个
    # jieba.load_userdict("userdict.txt")

    # ---- 方式 4：调整词频，强制让某个分词结果胜出 ----
    # suggest_freq 可对单个词或切分片段设置频率，强制改变切分结果
    print("\nsuggest_freq 强制不切分 '创新办'：")
    jieba.suggest_freq(("创新", "办"), tune=True)   # 让 "创新 办" 胜出 => 不切 "创新办"
    print(jieba.lcut(text))


# ============================================================
# 四、词性标注（jieba.posseg）
# ============================================================

def demo_posseg():
    text = "我爱北京天安门"

    words = pseg.cut(text)                # 返回 generator，元素为 pair 对象
    print("词性标注结果：")
    for word, flag in words:
        print(f"  {word} / {flag}")
    # 我 / r        代词
    # 爱 / v        动词
    # 北京 / ns     地名
    # 天安门 / ns   地名

    # lcut 直接返回 list[(word, flag)]
    print("\nposseg.lcut：", pseg.lcut(text))


# ============================================================
# 五、关键词提取
# ============================================================

def demo_keywords():
    text = (
        "自然语言处理是人工智能领域中的一个重要方向。"
        "它研究能实现人与计算机之间用自然语言进行有效通信的理论和方法。"
        "自然语言处理是一门融语言学、计算机科学、数学于一体的科学。"
    )

    # ---- 1. 基于 TF-IDF 的关键词提取 ----
    print("【TF-IDF】extract_tags：")
    print(jieba.analyse.extract_tags(text, topK=5, withWeight=True, allowPOS=()))
    # withWeight=True 返回 (word, weight)；allowPOS 限定词性

    # 自定义 IDF 频率库与停用词库
    # jieba.analyse.set_idf_path("../extra_dict/idf.txt.big")
    # jieba.analyse.set_stop_words("../extra_dict/stop_words.txt")

    # ---- 2. 基于 TextRank 的关键词提取 ----
    print("\n【TextRank】textrank：")
    print(jieba.analyse.textrank(text, topK=5, withWeight=True))


# ============================================================
# 六、Tokenize：返回词语在原文中的起止位置
# ============================================================

def demo_tokenize():
    text = "永和服装饰品有限公司"
    query = "永和"

    print("tokenize 默认模式：")
    for tk in jieba.tokenize(text):
        print(f"  word={tk}  start={tk.start}  end={tk.end}")

    print("\ntokenize 全模式：")
    for tk in jieba.tokenize(text, mode="search"):
        print(f"  word={tk}  start={tk.start}  end={tk.end}")

    # 常用于搜索高亮：根据 start/end 截取原文
    print("\n根据位置截取原文：")
    for tk in jieba.tokenize(text):
        print(f"  原文切片: {text[tk.start:tk.end]}")


# ============================================================
# 七、并行分词（多进程加速大规模文本）
# ============================================================

def demo_parallel():
    # 参数为进程数；仅在 0 < n <= os.cpu_count() 时有效
    jieba.enable_parallel(4)

    big_text = " ".join(["自然语言处理"] * 100000)
    print("并行分词结果长度：", len(list(jieba.cut(big_text))))

    jieba.disable_parallel()   # 关闭并行模式


# ============================================================
# 八、延迟加载机制
# ============================================================

def demo_lazy_load():
    # 默认 jieba 在第一次调用时才加载词典（约 1MB），可手动控制
    jieba.initialize()          # 手动初始化（可选）

    # 也可以在导入时就指定延迟加载
    # jieba.set_dictionary("path/to/dict.txt")   # 切换主词典


# ============================================================
# 九、词频统计 + 停用词过滤（实际项目常见用法）
# ============================================================

STOPWORDS = set("的 了 是 在 我 也 有 和 与 及 等 中".split())


def demo_word_freq():
    text = (
        "自然语言处理是人工智能的重要方向，"
        "自然语言处理融合了语言学与计算机科学。"
    )

    # 1. 分词 -> 2. 过滤停用词与非中文字符 -> 3. 统计词频
    words = [w for w in jieba.lcut(text) if w.strip() and w not in STOPWORDS]

    freq = Counter(words)
    print("词频统计 Top 5：")
    for word, cnt in freq.most_common(5):
        print(f"  {word}: {cnt}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("一、三种分词模式")
    print("=" * 60)
    demo_three_modes()

    print("\n" + "=" * 60)
    print("二、HMM 新词发现")
    print("=" * 60)
    demo_hmm()

    print("\n" + "=" * 60)
    print("三、自定义词典")
    print("=" * 60)
    demo_custom_dict()

    print("\n" + "=" * 60)
    print("四、词性标注")
    print("=" * 60)
    demo_posseg()

    print("\n" + "=" * 60)
    print("五、关键词提取")
    print("=" * 60)
    demo_keywords()

    print("\n" + "=" * 60)
    print("六、Tokenize 返回位置")
    print("=" * 60)
    demo_tokenize()

    print("\n" + "=" * 60)
    print("七、并行分词")
    print("=" * 60)
    demo_parallel()

    print("\n" + "=" * 60)
    print("八、延迟加载")
    print("=" * 60)
    demo_lazy_load()

    print("\n" + "=" * 60)
    print("九、词频统计 + 停用词")
    print("=" * 60)
    demo_word_freq()
