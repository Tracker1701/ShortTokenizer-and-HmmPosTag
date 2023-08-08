from models import ShortTokenizer
from models import HmmPosTag
import time
import os
import warnings

warnings.filterwarnings('ignore')

# 分词以及评估
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

# 词性标注 以及评估
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

# 计算预测结果的准确率、召回率、F1
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

# 打印测试结果
def evalutate(dataset, token_res):
    # dataset(list): 真实结果
    # token_res(list): 分词结果
    precision, recall, f1 = eval(token_res, dataset)
    print("精确率:\t{:.3%}".format(precision))
    print("召回率:\t{:.3%}".format(recall))
    print("f1:\t{:.3%}".format(f1))

if __name__ == "__main__":
    # 评估分词模型
    token_res = word_seg_eval('./data/人民日报语料（UTF8）.txt')

    # 评估词性标注
    # 在最大概率分词集合上标注词性
    trainfile = './data/人民日报词性标注版.txt'
    testfile = './data/人民日报分词结果.txt'
    posTag_eval(trainfile, testfile)

    


