# ShortTokenizer-and-HmmPosTag
自然语言处理初步的课程实验：对于人民日报语料库的最大概率模型分词与使用隐马尔可夫模型进行词性标注
<a name="okwZd"></a>

# 一、实验目的

自行实现分词算法和词性标注，不直接或间接调用现有工具包（包括但不限于：HanLP，CoreNLP等）中提供的分词接口。了解分词算法和词性标注背后的原理。
<a name="a0XxU"></a>

# 二、实验内容

● 实现统计分词方法；<br />● 对分词结果进行词性标注，也可以在分词的同时进行词性标注；<br />● 对分词及词性标注结果进行评价，包括4个指标：正确率、召回率、F1值和效率。
<a name="fGTYb"></a>

# 三、实验原理

<a name="SuOsP"></a>

## 1.最大概率模型分词

源代码路径：./models/ShortTokenizer.py<br />算法设计：<br />此算法是基于最短路分词模型的，最短路分词模型的主要思想是将句子中的所有字符当作节点，根据字典找出句子中所有的词语，将词语两端的字符连接起来，构成从词首指向词尾的一条边。通过找出所有的候选词，构建出一个有向无环图（DAG）。找到从句首字符到句尾字符的最短路径，即可作为句子的分词结果。最短路径分词方法采用的规则使切分出来的词数最少，符合汉语自身的规律。<br />最短路分词算法，由以下几个步骤实现：<br />① 构造句子的切分图，如果句子 $sentence$的子串 $w[i:j]$在词典中，则添加边 $V(i,j)$，得到句子的有向无环图 DAG<br />② 采用Dijkstra 算法动态规划地求解最短路径， $dp[i]$表示DAG中句首到第 $i$个字符的路径长度<br />③ 状态转移函数如下:  $dp[i]=mindp[j-1]+1$；其中： $i$为当前边的起点， $j$为当前边的终点。<br />④ 回溯最优路径<br />现在考虑成词的概率，通过极大似然估计，以词频表示成词概率，为DAG的每条边赋予权重，优化分词结果。通过Dijkstra算法求得的带权最短路径即为所有分词结果中概率最大的分词方法。该分词方法本质上是使用了1-gram文法的最大概率分词模型。
<a name="xm50d"></a>

## 2.隐马尔可夫模型进行词性标注

源代码路径：./models/HmmPosTag.py<br />算法设计：<br />词性标注是序列标注问题，可采用Hmm模型的解码问题的解决方法。将词性序列作为隐藏序列，将词语序列作为观测序列，同过Viterbi算法预测最优的词性序列。<br />使用BMES标注方法，将分词任务转换为字标注的问题，通过对每个字进行标注得到词语的划分。具体来说，BMES标注方法是用“B、M、E、S”四种标签对词语中不同位置的字符进行标注，B表示一个词的词首位置，M表示一个词的中间位置，E表示一个词的末尾位置，S表示一个单独的字。<br />字标注的问题可视为隐马尔可夫模型中的解码问题。句子的BMES标注序列作为隐藏状态序列，句子的字符序列作为可观测序列，通过以下两个步骤实现词性标注：<br />① 学习模型参数<br />对预<br />对语料进行统计，获得隐藏状态的转移概率矩阵trans、发射概率矩阵emit 、初始状态矩阵start

1. 观测序列 $O$ ：句子的字符序列 $[w_0, w_1,\dots, w_n]$
2. 隐藏序列 $S$：BMES标注序列 $[p_0, p_1,\dots, p_n]$
3. 初始概率 $\pi$： $start(i)=P_{(p_0=i)}=count(p_0=i)/count(sentence)\quad i\in\{B、M、E、D\}$
4. 转移概率 $trans$： $trans(i,j)=P(j│i)=count(p_k=i ,p_{k+1}=j)/count(i) i,j \in\{B、M、E、D\}$
5. 发射概率 $emit$： $emit(i,w)=P(w│i)=count(state(w)=i)/count(i) \quad i\in\{B、M、E、D\}$

② 使用 Viterbi 算法预测<br />Viterbi算法是用动态规划的方法求解最优的标注序列。每个标注序列视为从句首到句尾的一个路径，通过Viterbi算法获取概率最大的路径，在主要由以下几步实现：

1. 状态 $dp[i][j]$：表示第 $i$个字符，标签为$j$的所有路径中的最大概率。
2. 记录路径 $path[i][j]$：表示 $path[i][j]$为最大概率时，第 $i-1$个字符的标签
3. 状态初始化： $dp[0][j] =start(j) emit(j,w_0)$
4. 递推（状态转移方程）： $dp[i][j]= max_{k\in \{pos\}}⁡(dp[i-1][k]×trans[k,j]) × emit[j,w_i]$
5. 记录路径： $path[i][j]=arg⁡max_{k∈\{pos\}}⁡(dp[i-1][k]×trans[k,j])$
6. 回溯最优路径： $p_i=path[i+1][p_(i+1) ] \quad i=n-1,n-2,……1,0$
7. 输出最优路径： $[p_1,p_2……p_n]$
   <a name="TXdKK"></a>

# 四、实验步骤

<a name="pecJV"></a>

## （一）最大概率模型分词

1.使用最大概率模型分词，创建对象时需要有词频字典和总词数

```python
class ShortTokenizer:
    def __init__(self):
        self.word_freq = {}     # 词频字典
		self.word_num = 0       # 词数
```

2.根据训练语料统计词频，主要是获得词频字典和总词数

```python
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
```

3.计算word的词频 -log(P) = log(总词频) - log(该词词频)

```python
    def __weight(self, word):
        # word (string): 切分的词语，切分图上的一条边
        freq = self.word_freq.get(word, 0)
        # 词典中存在该词则返回 -log(P)，否则返回0
        if freq:
            return math.log(self.word_num) - math.log(freq)
        else:
            return 0
```

4.结合统计信息的最短路分词函数（最大概率分词）

```python
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
```

5.测试分词模型

```python
if __name__ == "__main__":
    Tokenizer = ShortTokenizer()
    Tokenizer.train('../data/人民日报语料（UTF8）.txt')
    result = Tokenizer.Token('改革春风吹满地，中国人民真争气')
    print(result)
```

分词结果：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/25419362/1671114397835-2e316dbc-26a2-4be3-b1bc-99cf8107ab34.png#averageHue=%232d2d2d&clientId=u65cd8d6f-7e04-4&from=paste&height=109&id=u210b29c4&originHeight=136&originWidth=1091&originalType=binary&ratio=1&rotation=0&showTitle=false&size=6780&status=done&style=none&taskId=u8c3b48a3-65a1-4a5c-bead-4a98dd5277a&title=&width=872.8)
<a name="SnBvd"></a>

## （二）隐马尔可夫模型进行词性标注

1.词性标注语料库_人民日报词性标注版.txt_介绍<br />（1）语料库中有 26 个基本词类标记<br />      形容词a、区别词b、连词c、副词d、叹词e、方位词f、语素g、前接成分h、成语i、简称j、后接成分k、习惯用语l、数词m、名词n、拟声词o、介词p、量词q、代词r、处所词s、时间词t、助词u、动词v、标点符号w、非语素字x、语气词y、状态词z。<br />（2）语料库中还有 74 个扩充标记：对于语素，具体区分为 Ag Bg Dg Mg Ng Rg Tg Vg Yg<br />（3）词性标注只标注基本词性，因此在数据清洗的过程中，将扩充标记归类到各个基本词类中，语素也归类到相应词类中<br />2.首先创建对象时要有转移概率矩阵、发射概率矩阵，初始状态矩阵、词性表、以及trans和emit 矩阵中各个 pos 的归一化分母

```python
class HmmPosTag:
    def __init__(self):
        self.trans_prop = {}    # 转移概率矩阵
        self.emit_prop = {}     # 发射概率矩阵
        self.start_prop = {}    # 初始状态矩阵
        self.poslist = []       # 词性表
        self.trans_sum = {}
        self.emit_sum = {}
```

3.更新转移概率矩阵函数

```python
    def __upd_trans(self, curpos, nxtpos):
        # curpos (string): 当前词性
        # nxtpos (string): 下一词性
        if curpos in self.trans_prop:
            if nxtpos in self.trans_prop[curpos]:
                self.trans_prop[curpos][nxtpos] += 1
            else:
                self.trans_prop[curpos][nxtpos] = 1
        else:
            self.trans_prop[curpos] = {nxtpos: 1}
```

4.更新发射概率矩阵函数

```python
    def __upd_emit(self, pos, word):
        # pos (string): 词性
        # word (string): 词语
        if pos in self.emit_prop:
            if word in self.emit_prop[pos]:
                self.emit_prop[pos][word] += 1
            else:
                self.emit_prop[pos][word] = 1
        else:
            self.emit_prop[pos] = {word: 1}
```

5.更新初始状态矩阵函数

```python
    def __upd_start(self, pos):
        # pos (string): 初始词语的词性
        if pos in self.start_prop:
            self.start_prop[pos] += 1
        else:
            self.start_prop[pos] = 1
```

6.训练 hmm 模型、求得转移矩阵、发射矩阵、初始状态矩阵

```python
	def train(self, data_path):
        # data_path (string): 训练数据的路径
        # 训练数据
        f = open(data_path, 'r', encoding='utf-8')
        print("正在训练模型……")
        stime = time.thread_time()

        for line in f.readlines():
            line = line.strip().split()
            # 统计初始状态的概率
            self.__upd_start(line[0].split('/')[1])
            # 统计转移概率、发射概率
            for i in range(len(line) - 1):
                self.__upd_emit(line[i].split('/')[1], line[i].split('/')[0])
                self.__upd_trans(line[i].split('/')[1],
                                 line[i + 1].split('/')[1])
            i = len(line) - 1
            self.__upd_emit(line[i].split('/')[1], line[i].split('/')[0])
        f.close()
```

接着记录所有的 pos

```python
        self.poslist = list(self.emit_prop.keys())
        self.poslist.sort()
```

统计 trans、emit 矩阵中各个 pos 的归一化分母

```python
        num_trans = [
            sum(self.trans_prop[key].values()) for key in self.trans_prop
        ]
        self.trans_sum = dict(zip(self.trans_prop.keys(), num_trans))
        num_emit = [
            sum(self.emit_prop[key].values()) for key in self.emit_prop
        ]
        self.emit_sum = dict(zip(self.emit_prop.keys(), num_emit))
```

最后

```python
    	etime = time.thread_time()
        print("训练完成，耗时{}s".format(etime - stime))
```

7.Viterbi 算法预测词性<br />首先初始化 dp 矩阵（DP 矩阵: posnum * wordsnum 存储每个 word 每个 pos 的最大概率）

```python
    def predict(self, sentence):
        # sentence (string): 分词后的句子（空格隔开）
        sentence = sentence.strip().split()
        # 词性数量
        posnum = len(self.poslist)

        dp = pd.DataFrame(index=self.poslist)
        path = pd.DataFrame(index=self.poslist)
        # 初始化 dp 矩阵（DP 矩阵: posnum * wordsnum 存储每个 word 每个 pos 的最大概率）
        start = []
        num_sentence = sum(self.start_prop.values()) + posnum
        for pos in self.poslist:
            sta_pos = self.start_prop.get(pos, 1e-16) / num_sentence
            sta_pos *= (self.emit_prop[pos].get(sentence[0], 1e-16) /
                        self.emit_sum[pos])
            sta_pos = math.log(sta_pos)
            start.append(sta_pos)
        dp[0] = start
```

初始化 path 矩阵<br />算法方面参考[https://zhuanlan.zhihu.com/p/112529258](https://zhuanlan.zhihu.com/p/112529258)

```python
        path[0] = ['_start_'] * posnum
        # 递推
        for t in range(1, len(sentence)):  # 句子中第 t 个词
            prob_pos, path_point = [], []
            for i in self.poslist:  # i 为当前词的 pos
                max_prob, last_point = float('-inf'), ''    # 设置评分与词性的初始值
                emit = math.log(self.emit_prop[i].get(sentence[t], 1e-16) / self.emit_sum[i])
                for j in self.poslist:  # j 为上一词的 pos
                    """
                        状态转移方程
                        dp[t] = max(dp.loc[j, t - 1] + emit 
                        + math.log(self.trans_prop[j].get(i, 1e-16) / self.trans_sum[j]))
                        其中
                        emit = math.log(self.emit_prop[i].get(sentence[t], 1e-16) / self.emit_sum[i])
                    """
                    tmp = dp.loc[j, t - 1] + emit
                    tmp += math.log(self.trans_prop[j].get(i, 1e-16) / self.trans_sum[j])
                    if tmp > max_prob:
                        max_prob, last_point = tmp, j
                prob_pos.append(max_prob)
                path_point.append(last_point)
            dp[t], path[t] = prob_pos, path_point
```

回溯

```python
        prob_list = list(dp[len(sentence) - 1])
        # 从获得最大评分的路径开始回溯
        cur_pos = self.poslist[prob_list.index(max(prob_list))]
        path_que = []
        path_que.append(cur_pos)
        for i in range(len(sentence) - 1, 0, -1):
            cur_pos = path[i].loc[cur_pos]
            path_que.append(cur_pos)
```

返回结果

```python
        postag = []
        for i in range(len(sentence)):
            postag.append(sentence[i] + '/' + path_que[-i - 1])
        return postag
        # 词性标注序列
```

8.测试词性标注模型

```python
if __name__ == "__main__":
    hmm = HmmPosTag()
    hmm.train("../data/人民日报词性标注版.txt")
    result = hmm.predict("新年  的  钟声  刚刚  敲响  ，  千  里  淮河  传来  喜讯")
    print(result)
```

词性标注结果：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/25419362/1671115423932-432843d7-04a1-4371-ad2a-f49fc0ae305e.png#averageHue=%232e2e2e&clientId=u65cd8d6f-7e04-4&from=paste&height=109&id=u9b667c03&originHeight=136&originWidth=958&originalType=binary&ratio=1&rotation=0&showTitle=false&size=8994&status=done&style=none&taskId=u100798bc-09d5-4d50-a7c4-63db45c3d2e&title=&width=766.4)
<a name="YdXc4"></a>

## （三）评估函数

1.计算预测结果的准确率、召回率、F1

```python
def eval(predict, truth):
    # predict(list): 预测结果
    # truth(list): 真实结果
    assert len(predict) == len(truth)
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(predict)):
        right = len([j for j in predict[i] if j in truth[i]])
        tp += right
        fn += len(truth[i]) - right
        fp += len(predict[i]) - right
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1    # 精确率, 召回率, f1
```

2.打印测试结果

```python
def evalutate(dataset, token_res):
    # dataset(list): 真实结果
    # token_res(list): 分词结果
    precision, recall, f1 = eval(token_res, dataset)
    print("精确率:\t{:.3%}".format(precision))
    print("召回率:\t{:.3%}".format(recall))
    print("f1:\t{:.3%}".format(f1))
```

<a name="An25k"></a>

## （四）主函数

```python
if __name__ == "__main__":
    # 评估分词模型
    token_res = word_seg_eval('./data/人民日报语料（UTF8）.txt')

    # 评估词性标注
    # 在最大概率分词集合上标注词性
    trainfile = './data/人民日报词性标注版.txt'
    testfile = './data/人民日报分词结果.txt'
    posTag_eval(trainfile, testfile)
```

其中分词以及评估函数word_seg_eval()如下：

```python
def word_seg_eval(trainfile):
    # trainfile (string): 训练数据文件路径
    # 返回分词结果
    # 打开训练语料，导入数据
    with open(trainfile, 'r', encoding='utf8') as f:
        dataset = [line.strip().split() for line in f.readlines()]
    # 取前五千行作为测试集
    dataset = dataset[0:5000]
    input_data = [''.join(line) for line in dataset]
    # 语料库大小
    dataset_size = float(os.path.getsize(trainfile)) / 1024  # 以 kb 为单位

    # 利用最大概率分词模型分词
    model = ShortTokenizer.ShortTokenizer()
    # 训练模型
    model.train(trainfile)
    # 分词结果
    token_result = []
    print("最大概率分词模型分词中……")
    stime = time.thread_time()      # 开始时间
    for line in input_data:
        token_result.append(model.Token(line))  # 预测分词
    etime = time.thread_time()      # 结束时间
    print("最大概率分词模型分词完成，用时{}s".format(etime-stime))
    print("--------分词评估结果--------")
    evalutate(dataset, token_result)
    print("效率:\t{:.3f} kb/s\n".format(dataset_size / (etime - stime)))

    # 保存分词结果
    with open('./data/人民日报分词结果.txt', 'w', encoding='utf8') as f:
        for i in token_result:
            f.write(' '.join(i) + '\n')
    return token_result
```

其中词性标注 以及评估函数posTag_eval()如下：

```python
def posTag_eval(trainfile, testfile):
    # trainfile (string): 训练数据集路径
    # testfile (string): 测试数据集路径
    # 返回词性标注结果
    hmm_pos = HmmPosTag.HmmPosTag()
    # 训练模型
    hmm_pos.train(trainfile)
    # 词性标注结果
    posTag_res = []
    # 测试集大小
    dataset_size = float(os.path.getsize(testfile)) / 1024  # 以 kb 为单位
    # 前一千五百行数据作测试集
    with open(trainfile, 'r', encoding='utf8') as f:
        dataset = [line.strip().split(' ') for line in f.readlines()[:1500]]
    with open(testfile, 'r', encoding='utf8') as f:
        print("HMM 词性标注模型预测分词中……")
        stime = time.thread_time()  # 开始时间
        for line in f.readlines()[:1500]:
            posTag_res.append(hmm_pos.predict(line.strip()))  # 预测分词
        etime = time.thread_time()  # 结束时间
    print("词性标注完成，用时{}s".format(etime-stime))
    print("------词性标注评估结果------")
    evalutate(dataset, posTag_res)
    print("效率:\t{:.3f} kb/s\n".format(dataset_size / (etime - stime)))
    return posTag_res
```

<a name="ZTulV"></a>

# 五、实验结果及评估

直接运行./evaluate.py得到结果如下<br />分词模型的输出：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/25419362/1671118778169-eded873b-7f7d-4bb7-bab5-4cbab50965c5.png#averageHue=%232e2d2d&clientId=u65cd8d6f-7e04-4&from=paste&height=228&id=u2731ee22&originHeight=285&originWidth=997&originalType=binary&ratio=1&rotation=0&showTitle=false&size=25260&status=done&style=none&taskId=uc6f276cd-9fdd-4832-b729-7e99c894273&title=&width=797.6)<br />词性标注模型的输出：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/25419362/1671119856344-c37138a4-cc9c-4a7c-9b08-f8eb1aaff5d2.png#averageHue=%232c2c2c&clientId=u65cd8d6f-7e04-4&from=paste&height=267&id=uf0502e0c&originHeight=334&originWidth=1039&originalType=binary&ratio=1&rotation=0&showTitle=false&size=18555&status=done&style=none&taskId=ub3fea670-1927-4ee7-a155-a48839dfa4a&title=&width=831.2)
<a name="Lv1rk"></a>

# 六、问题以及解决方法

通过本实验我熟悉了分词算法和词性标注原理。<br />在这个过程中我遇到不少问题，以下为部分问题。<br />1.首先是会产生以下警报：

```info
F:\Pycharm_project\NLP_ProjectTest\fenci\models\HmmPosTag.py:127: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead.  To get a de-fragmented frame, use `newframe = frame.copy()`<br />dp[t], path[t] = prob_pos, path_point
```

我在仔细研究后认为其不影响结果，使用如下代码将其忽略

```python
import warnings
warnings.filterwarnings('ignore')
```

2.使用的语料库不契合的问题<br />我在网上收集到人民日报有词性标注版的数据后，用有词性标注版的数据与课程发的人民日报语料库分词以及词性标注后做词性标注的评估时发现各评价指标只有80%左右，并不让人满意。<br />然后我将有词性标注版的语料库处理为分词语料库后，再进行训练。得到的词性标注评价指标均在90%以上。<br />3.分词时dp矩阵与sentence、graph错位的问题<br />这属于动态规划问题，但还是要非常小心，差一位就会输出非常离谱的结果<br />我在不断地对ShortTokenizer.py调试的过程中调好了这段代码

```python
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
```

4.对于Hmm模型中出现的未登录词（字）采用 Laplace 平滑处理。由于某些字、词出现很少，如果采用加一平滑会导致发射概率过大的问题，因此采用较小的$\lambda = 1e-6$<br />5.在Hmm模型中，大部分词语的发射概率较低，随着句子长度的增加（约为120词），路径的概率变得很小，程序下溢。<br />所以我将路径概率取对数，概率相乘转化为对数相加，避免路径概率下溢。<br />6.对Hmm模型的词性标注算法还是很模糊<br />算法方面参考[https://zhuanlan.zhihu.com/p/112529258](https://zhuanlan.zhihu.com/p/112529258)

```python
        path[0] = ['_start_'] * posnum
        # 递推
        for t in range(1, len(sentence)):  # 句子中第 t 个词
            prob_pos, path_point = [], []
            for i in self.poslist:  # i 为当前词的 pos
                max_prob, last_point = float('-inf'), ''    # 设置评分与词性的初始值
                emit = math.log(self.emit_prop[i].get(sentence[t], 1e-16) / self.emit_sum[i])
                for j in self.poslist:  # j 为上一词的 pos
                    """
                        状态转移方程
                        dp[t] = max(dp.loc[j, t - 1] + emit 
                        + math.log(self.trans_prop[j].get(i, 1e-16) / self.trans_sum[j]))
                        其中
                        emit = math.log(self.emit_prop[i].get(sentence[t], 1e-16) / self.emit_sum[i])
                    """
                    tmp = dp.loc[j, t - 1] + emit
                    tmp += math.log(self.trans_prop[j].get(i, 1e-16) / self.trans_sum[j])
                    if tmp > max_prob:
                        max_prob, last_point = tmp, j
                prob_pos.append(max_prob)
                path_point.append(last_point)
            dp[t], path[t] = prob_pos, path_point
```

7.开始做的时候总体思路不太清晰<br />我采取了自底向上的编程方式，先编写小的部分，最后再编写主函数
<a name="Hrl5c"></a>

# 七、运行方式

环境：<br />python3.9<br />pandas==1.3.5<br />项目结构：<br />./data里是语料库，包括人民日报词性标注版.txt和人民日报语料（UTF8）.txt<br />运行过程会产生人民日报分词结果.txt<br />./models里是分词模型ShortTokenizer.py以及词性标注模型HmmPosTag.py<br />evaluate.py是主函数<br />运行方式：<br />直接运行evaluate.py
