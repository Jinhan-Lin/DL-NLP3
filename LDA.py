#!/usr/bin/python
# -*- coding:utf-8 -*-

import jieba, os, re
import numpy as np
import pandas as pd
from gensim import corpora, models
from sklearn import svm
from sklearn import model_selection


def data_gen():
    train_data = "./train_data.txt"
    test_data = "./test_data.txt"
    if not os.path.exists('./train_data.txt'):
        trains = open(train_data, 'w', encoding='UTF-8')
        tests = open(test_data, 'w', encoding='UTF-8')
        datasets_root = "./datasets"
        catalog = "inf.txt"

        test_num = 10
        test_length = 200
        with open(os.path.join(datasets_root, catalog), "r", encoding='utf-8') as f:
            all_files = f.readline().split(",")
            print(all_files)
        for name in all_files:

            with open(os.path.join(datasets_root, name + ".txt"), "r", encoding='utf-8') as f:
                file_read = f.readlines()
                train_num = len(file_read) - test_num
                choice_index = np.random.choice(len(file_read), test_num + train_num, replace=False)
                for train in choice_index[0:train_num]:
                    line = file_read[train]
                    line = re.sub('\s', '', line)
                    line = re.sub('[\u0000-\u4DFF]', '', line)
                    line = re.sub('[\u9FA6-\uFFFF]', '', line)
                    if len(line) == 0:
                        continue
                    seg_list = list(jieba.cut(line, cut_all=False))  # 使用精确模式
                    line_seg = ""
                    for term in seg_list:
                        line_seg += term + " "
                    trains.write(line_seg.strip() + '\n')

                tests.write(name)
                for test in choice_index[train_num:test_num + train_num]:
                    if test + test_length >= len(file_read):
                        continue
                    test_line = ""
                    line = file_read[test]
                    line = re.sub('\s', '', line)
                    line = re.sub('[\u0000-\u4DFF]', '', line)
                    line = re.sub('[\u9FA6-\uFFFF]', '', line)
                    seg_list = list(jieba.cut(line, cut_all=False))  # 使用精确模式
                    for term in seg_list:
                        test_line += term + " "
                    tests.write(test_line.strip()+'\n')
        trains.close()
        tests.close()
        print("数据生成完毕")


if __name__ == "__main__":
    #数据预处理，并抽取段落作为测试集
    data_gen()
    #获取训练数据，并调整其格式
    fr = open('./train_data.txt', 'r', encoding='utf-8')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ')]
        train.append(line)

    ##训练
    dictionary = corpora.Dictionary(train)
    # 将每个段落进行ID化
    corpus = [dictionary.doc2bow(text) for text in train]
    #构建LDA模型，设定主题数为12
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=12)
    topic_list_lda = lda.print_topics(12)
    print("LDA模型训练得到的12个主题的单词分布为：\n")
    for topic in topic_list_lda:
        print(topic)

    #获取测试段落的主题分布
    file_test = "./test_data.txt"
    test_org = open(file_test, 'r', encoding='UTF-8')
    test = []
    # 处理成正确的输入格式
    for line in test_org:
        line = [word.strip() for word in line.split(' ')]
        test.append(line)
    for text in test:
        corpus_test = dictionary.doc2bow((text))
    corpus_test = [dictionary.doc2bow(text) for text in test]
    # 得到每个测试集的主题分布
    topics_test = lda.get_document_topics(corpus_test)
    N = len(topics_test)
    #展示前10个段落的主题分布
    for i in range(10):
        print("第", i, "个段落的主题分布为", topics_test[i], '\n')
    ##分类
    data_train = [[0 for i in range(12)]for i in range(N)]
    for i in range(N):
        for j in range(len(topics_test[i])):
            k = topics_test[i][j][0]
            data_train[i][k] = topics_test[i][j][1]
    #获取数据标签
    label = pd.read_csv('./label.csv', header=None)
    label = np.array(label)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data_train, label, test_size = 0.3, random_state = 4321)
    print(X_test)
    ##构建SVM
    svm = svm.SVC(kernel='linear')  # 实例化
    svm.fit(X_train, Y_train)  # 拟合
    pred = svm.predict(X_test)
    #计算准确率
    count =0
    for i in range(len(X_test)):
        if pred[i] == Y_test[i]:
            count+1
    acc = (count/len(pred))*100
    print("准确率:", acc, "%")
    print(pred)
    print(Y_test)
    fr.close()
    test_org.close()