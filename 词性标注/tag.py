from nltk.corpus import brown
import numpy as np
"""
开发日志：今天重新读了以前写的代码，并且重新组织了结构，完成了初始化函数
明天完成hmm函数和rate函数，并写出实验报告
2018/01/04

"""

def Statistic(word_tag):  # 统计每一个词性总共有多少个，最后输出字典【nc，121】
    """
    使用brown语料库并且保存在一个
    统计每一个词性总共有多少个，最后输出字典【nc，121】
    :return wordback()
    """

    cate_table = {}
    for temp_tag in word_tag:
        y = temp_tag[1].split("+")

        for xtemp in y:
            x = xtemp.split("-")
            for temp in x:
                if temp in cate_table:
                    cate_table[temp] = int(cate_table[temp]) + 1
                else:
                    cate_table[temp] = 1
    return cate_table  # cixin_map


def row_normalization(X):
    """
    按行归一化
    :param X: 矩阵
    :return: 归一化后的矩阵
    """
    X.dtype = 'float'
    try:
        X.shape[1]
    except IndexError:
        Max = np.max(X)
        Min = np.min(X)
        X = (X - Min) / (Max - Min)
        return X

    for l in range(X.shape[0]):
        Max = np.max(X[l])
        Min = np.min(X[l])
        if (Max - Min) == 0:
            X[l] = np.zeros(X[l].shape)
            continue
        X[l] = (X[l] - Min) / (Max - Min)
    return X


class HMM:
    def __init__(self):
        self.Cixin_set=[]
        self.Cixin_map={}
        self.vocab_map={}
        self.Cixin_len=0
        #self.Ci_pro#
        #self.Tran_matrix=np.zeros()#转移矩阵
        #self.emitter_pro_matrix=np.zeros()#发射矩阵

    def Process(self):
        #载入词典
        words_tag = brown.tagged_words(fileids=['ca01','ca02','ca03'])  # words_tag[1][1]
        #print(words_tag)
        t=Statistic(words_tag)#cixinmap

        #构造词性的集合-----观察状态序列
        self.Cixin_set=[]
        for key in t.keys():
            self.Cixin_set.append(key)
        #声明词性的长度
        Cixin_num=len(self.Cixin_set)#长度
        self.Cixin_len=Cixin_num
        self.Cixin_map=dict(zip(list(self.Cixin_set),range(Cixin_num)))#'AT': 0, 'NP': 1, 'TL': 2, 'NN': 3各个词性在矩阵中的位置


        self.Tran_matrix=np.zeros((Cixin_num,Cixin_num))#转移矩阵

        self.Ci_pro=np.zeros(Cixin_num,dtype=int)
        pre_cixin=""
        for word in words_tag:#求出转移矩阵/然后再归一化
            y = word[1].split("+")
            for xtemp in y:
                x = xtemp.split("-")
                for temp in x:
                    no_cixin=temp
                    self.Ci_pro[self.Cixin_map[no_cixin]]+=1
                    try:
                        self.Tran_matrix[self.Cixin_map[no_cixin]][self.Cixin_map[pre_cixin]]+=1
                    except KeyError:
                        pass
                    pre_cixin=no_cixin

        self.Ci_pro=row_normalization(self.Ci_pro)#词性概率
        self.Tran_matrix=row_normalization(self.Tran_matrix)#转移矩阵

        vocab_list=[]
        for word in words_tag:
            vocab_list.append(word[0])
        self.vocab_map=dict(zip(vocab_list,range(vocab_list.__len__())))#词列表

        self.emitter_pro_matrix=np.zeros((vocab_list.__len__(),Cixin_num))
        for word in words_tag:
            try:
                self.emitter_pro_matrix[self.vocab_map[word[0]]][self.Cixin_map[word[1]]]+=1
            except KeyError:
                pass

        self.emitter_pro_matrix=row_normalization(self.emitter_pro_matrix)#发射矩阵
        print("finish Initialization")
        ###############################完成预处理/生成了后面需要的矩阵####################################

    def hmm(self,sentence_list):
        sentence_len=sentence_list.__len__()

        pro_table=np.zeros((sentence_len,self.Cixin_len,2))
        try:
            pro_table[0,:,0]=self.emitter_pro_matrix[self.vocab_map[sentence_list[0]]]
            pro_table[0,:,1]= -1
            for i in range(sentence_len)[1:]:
                for j in range(self.Cixin_len):
                    if self.emitter_pro_matrix[self.vocab_map[sentence_list[i]],j]==0:
                        continue
                    pre_cixin_pro = pro_table[i-1,:,0]
                    pre_cixin_pro += self.Tran_matrix[j]
                    pre_cixin_pro += self.emitter_pro_matrix[self.vocab_map[sentence_list[i]],j]
                    pro_table[i,j,0] = np.max(pre_cixin_pro)
                    pro_table[i,j,1] = np.where(pre_cixin_pro==np.max(pre_cixin_pro))[0][0]
            result_cixin_map=[]
            sy = int(np.where(pro_table[-1,:,0]==np.max(pro_table[-1,:,0]))[0][0])
            t=-1
        except KeyError:
            return "error!"
        while sy!=-1:
            result_cixin_map.append(sy)
            sy=int(pro_table[t,sy,1])
            t-=1
        result_cixin =[]

        for s in result_cixin_map[::-1]:
            result_cixin.append(self.Cixin_set[s])
        return result_cixin

    def rate(self):
        wor_tag = brown.tagged_words(fileids=['ca01'])
        sentence = []
        anse=[]
        pres=0
        presb=0
        for word in wor_tag:
            if(word[0]=="."):
                ans = self.hmm(sentence)
                #print(ans)
                for i in range(ans.__len__()):
                    str=[]
                    str=anse[i].split("+")
                    str=anse[i].split("-")
                    #print(anse[i])
                    #print(ans[i])
                    if ans[i] in str:
                        pres=pres+1
                        presb+=1
                    else:
                        pres+=1#最重要要修改的地方
                sentence=[]
                anse=[]
                print(presb/pres)
                pres=0
                presb=0
                break
            else:
                #print("hello")
                sentence.append(word[0])
                anse.append(word[1])



if __name__=='__main__':
    test=HMM()
    test.Process()
    test.rate()