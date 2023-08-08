# -*- encoding: utf-8 -*-
import math
import time


class ShortTokenizer:
    def __init__(self):
        self.word_freq = {}     # 词频字典
        self.word_num = 0       # 词数

    # 根据训练语料统计词频
    def train(self, filepath):
        # filepath (string): 训练语料文件路径
        # 统计词频
        print("正在训练模型……")
        stime = time.thread_time()

        # 正式训练
        with open(filepath, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip().split()
                self.word_num += len(line)  # 累加每行的词数目
                self.word_freq.update(
                    {i: self.word_freq.get(i, 0) + 1
                     for i in line})        # 更新词频

        etime = time.thread_time()
        print("训练完成，耗时{}s".format(etime - stime))

    # 计算word的词频 -log(P(w)) = log(num) - log(k_w)
    def __weight(self, word):
        # word (string): 切分的词语，切分图上的一条边
        freq = self.word_freq.get(word, 0)
        # 词典中存在该词则返回 -log(P)，否则返回0
        if freq:
            return math.log(self.word_num) - math.log(freq)
        else:
            return 0

    # 结合统计信息的最短路分词函数（最大概率分词）
    def Token(self, sentence):
        # sentence (string): 待切分的句子
        # 返回一个list: 切分的词语构成的 list

        # 句子长度
        length = len(sentence)
        # 构造句子的切分图
        graph = {}
        for i in range(length):
            graph[i] = []
            for j in range(i):
                # 最短是两个字的边
                freq = self.__weight(sentence[j:i + 1])
                if freq:
                    graph[i].append((j, freq))
        # 动态规划求解最优路径 ( arg min[-log(P)] )
        # 初始化DP矩阵 为每个字单作词
        dp = [(i, self.__weight(sentence[i])) for i in range(length)]
        dp.insert(0, (-1, 0))       # 右移dp矩阵权重
        # 状态转移函数：dp[i] = min{dp[j-1] + weight(sentence[j:i])}
        # i：为当前词的词尾；j: 为当前词的词头
        for i in range(2, len(dp)):
            index = dp[i][0]
            cost = dp[i][1] + dp[i - 1][1]     # 默认代价是与上一个字的dp相加
            for j, freq in graph[i - 1]:    # 此处为到词尾索引为i的边
                if freq + dp[j][1] < cost:
                    cost = freq + dp[j][1]
                    index = j
            dp[i] = (index, cost)
        # 回溯最优路径
        token_result = []
        end = length
        while end > 0:
            token_result.append(sentence[dp[end][0]:end])
            end = dp[end][0]
        # 将分得词逆转
        token_result.reverse()
        return token_result


if __name__ == "__main__":
    Tokenizer = ShortTokenizer()
    Tokenizer.train('../data/人民日报语料（UTF8）.txt')
    result = Tokenizer.Token('改革春风吹满地，中国人民真争气')
    print(result)
