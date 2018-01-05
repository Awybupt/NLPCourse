#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:40:29 2017

@author: weixiao
"""

import codecs
import sys

numMath = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9',u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'〇', u'零']
numMath_suffix = [u'.', u'%', u'亿', u'万', u'千', u'百', u'十', u'个',u'年', u'月', u'日']

def rules(line, start):
    if line[start] in numMath:
        oldstart=start
        while line[start]in numMath or line[start] in numMath_suffix:
            start=start+1
        if line[start]in numMath_suffix:
            start=start+1
        return start-oldstart

def rrules(line,start):
    if line[start]in numMath:
        oldstart=start
        while line[start]in numMath or line[start] in numMath_suffix:
            start=start-1
        return oldstart-start

def genDict(path):#生成字典
    f = codecs.open(path, 'r', 'utf-8')
    contents = f.read()
    contents = contents.replace(u'\r', u'')
    contents = contents.replace(u'\n', u'')
    mydict = contents.split(u' ')
    newdict = list(set(mydict))
    newdict.remove(u'')
    truedict = {}#字典真实
    for item in newdict:
        if len(item) > 0 and item[0] in truedict:
            value = truedict[item[0]]
            value.append(item)
            truedict[item[0]] = value
        else:
            truedict[item[0]] = [item]
    return truedict


def print_unicode_list(uni_list):
    for item in uni_list:
        print(item),


def FMM(mydict, sentence):#前向最大匹配
    ruleChar = []
    ruleChar.extend(numMath)
    result = []
    start = 0
    senlen = len(sentence)
    while start < senlen:
        curword = sentence[start]
        maxlen = 1
        if curword in numMath:
            maxlen = rules(sentence, start)
        if curword in mydict:
            words = mydict[curword]
            for item in words:
                itemlen = len(item)
                if sentence[start:start + itemlen] == item and itemlen > maxlen:
                    maxlen = itemlen
        result.append(sentence[start:start + maxlen])
        start = start + maxlen
    return result

def RMM(mydict, sentence):#后向最大匹配
    ruleChar = []
    ruleChar.extend(numMath)
    result = []
    start = 0
    senlen = len(sentence)
    sentence[::-1]
    for curword in mydict:
        curword[::-1]
    while start < senlen:
        curword = sentence[start]
        maxlen = 1
        if curword in mydict:
            words = mydict[curword]
            for item in words:
                itemlen = len(item)
                if sentence[start:start + itemlen] == item and itemlen > maxlen:
                    maxlen = itemlen
        if curword in numMath:
            maxlen = rrules(sentence, start)
        result.append(sentence[start:start + maxlen])
        start = start + maxlen
    return result


def main():
    args = sys.argv[1:]
    if len(args) < 4:
        print('Usage: python dw.py dict_path test_path result_path segment_way')
        exit(-1)
    dict_path = args[0]
    test_path = args[1]
    result_path = args[2]
    choice=args[3]

    dicts = genDict(dict_path)
    fr = codecs.open(test_path, 'r', 'utf-8')
    test = fr.read()
    result=[]
    if(choice=="RMM"):
        result = RMM(dicts, test)
    elif(choice=="FMM"):
        result = FMM(dicts, test)
    else:
        print('python dw.py dict_path test_path result_path segment_way')
        exit(-1)
    fr.close()
    fw = codecs.open(result_path, 'w', 'utf-8')
    for item in result:
        fw.write(item + ' ')
    fw.close()

if __name__ == "__main__":
    main()
