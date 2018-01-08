#!/usr/bin/env Python
# coding=utf-8
from nltk.corpus import stopwords
import numpy as np
import sys

class Ctext:
    def init(self,choice):
        """
        初始化数据：读文件，把训练集传输到
        """
        self.posi_train={}
        self.nege_train={}

        posi = [line.strip().lower() for line in open('./data/rt-polarity.pos', encoding = "ISO-8859-1").readlines()]#读取正例
        nege = [line.strip().lower() for line in open('./data/rt-polarity.neg', encoding = "ISO-8859-1").readlines()]#读取反例
        # indices = np.random.permutation(posi_size)
        
        train_num = int(len(posi) * 0.85)
        temp_posi_train = posi[:train_num]
        temp_negi_train = nege[:train_num]

        self.temp_posi_test = posi[train_num:]
        self.temp_negi_test = nege[train_num:]  # 测试集再分词

        self.subj = [line.split()[2].split('=')[-1] for line in open('./data/subjclueslen1-HLTEMNLP05.tff', 'r').readlines()]  # 读取词性
        stop = set([word.encode('ascii') for word in stopwords.words('english')])  # 借用nltk的英文停止词

        if choice==1:#去除停止词，命令行参数
            ret = list(set(self.subj) ^ set(stop))
            subj=ret

        for i in self.subj:
            self.posi_train[i]=0
            self.nege_train[i]=0

        for line in temp_posi_train:#选取有效词
            for voca in line.split():
                if voca in self.subj:
                    self.posi_train[voca]+=1

        for line in temp_negi_train:
            for voca in line.split():
                if voca in self.subj:
                    self.nege_train[voca]+=1

        for voca in self.subj:
            temp=self.posi_train[voca]#防止修改之后不知道去哪里找原先的值
            ntem=self.nege_train[voca]
            self.posi_train[voca]=temp/(temp+ntem+1)#防止除0
            self.nege_train[voca]=ntem/(temp+ntem+1)


    def bayes(self):
        tp=0
        fp=0
        tn=0
        fn=0
        for line in self.temp_posi_test:#判断正例
            pos_factor=1.0
            neg_factor=1.0#负因子/正因子
            for voca in line.split():
                if voca in self.subj:
                    try:
                        pos_factor*= self.posi_train[voca]
                        neg_factor*= self.nege_train[voca]
                    except KeyError:
                        print("error")
            if pos_factor >= neg_factor:
                tp += 1
            else:
                fn += 1

        for line in self.temp_negi_test:#判断反例
            pos_factor = 1.0
            neg_factor = 1.0  # 负因子/正因子
            for voca in line.split():
                if voca in self.subj:
                    try:
                        pos_factor *= self.posi_train[voca]
                        neg_factor *= self.nege_train[voca]
                    except KeyError:
                        print("error!")
            if pos_factor > neg_factor:
                fp += 1
            else:
                tn += 1

        return [tp/(tp+fp),tp/(tp+fn),tn/(tn+fn),tn/(tn+fp)]

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print('Usage: (1/0)stop word')
        exit(-1)
    temp=Ctext()
    temp.init(int(args[0]))
    ans=temp.bayes()

    print("Positive Precision:",ans[0])
    print("Positive Recall:",ans[1])
    print("Negative Precision:",ans[2])
    print("Negative Recall:",ans[3])
    print("Positive F-score:",(2*ans[0]*ans[1])/(ans[0]+ans[1]))
    print("Negative F-score:",(2*ans[2]*ans[3])/(ans[2]+ans[3]))