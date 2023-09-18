import fileinput
from gensim.models import Word2Vec


def string_to_tripeptide(file):
    data = []
    for line in fileinput.input(file):
        line = line.strip()
        if line.startswith(">"):
            continue
        else:
            letters = []
            for i in range(len(line) - 2):
                letters.append(line[i:i + 3])
            data.append(letters)

    return data


fileName = "seven_n3_word_noAll_sample.txt"

word_dataSet = string_to_tripeptide(fileName)

model = Word2Vec(
    word_dataSet,
    vector_size=100,  # 词向量的维度
    window=5,  # 句子中当前单词和预测单词之间的最大距离
    min_count=1,  # 忽略总频率低于此的所有单词 出现的频率小于 min_count 不用作词向量
    seed=1,     # 设置随机种子
    workers=1,  # 使用这些工作线程来训练模型（使用多核机器进行更快的训练）
    sg=1,  # 训练方法 1：skip-gram 0；CBOW。
    hs=0,  # 1: 采用hierarchical softmax训练模型; 0: 使用负采样
    negative=5,  # 指定使用负采样的数目，设置多个负采样(通常在5-20之间)
    alpha=1e-2,  # 初始学习率
    min_alpha=1e-5,  # 最小学习率，随着训练的进行，学习率线性下降到min_alpha
    sample=1e-5,  # 高频词汇的随机降采样的配置阈值
    epochs=200  # 语料库上的迭代次数y
)

# 词向量保存
model.wv.save_word2vec_format('seven_n3_word_noAll_sample_workers1.vector', binary=False)

# 模型保存
model.save('test.model')
