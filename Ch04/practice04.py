#!/usr/bin/python
# -*- coding:utf-8 -*-

import bayes
from numpy import *
import feedparser

#4.5.1 准备数据：从文本中构建向量
listOPosts,listClasses= bayes.loadDataSet()
myVocabList=bayes.createVocabList(listOPosts)
print "myVocabList:",myVocabList
print "listOPosts[0]出现词汇:",bayes.setOfWords2Vec(myVocabList,listOPosts[0])
print "listOPosts[3]出现词汇:",bayes.setOfWords2Vec(myVocabList,listOPosts[3])


#4.5.2 训练算法：从词向量计算概率
#4.5.3 测试算法：根据现实情况修改分类器
listOPosts, listClasses =bayes.loadDataSet()
myVocabList =bayes.createVocabList(listOPosts)
trainMat=[]
for postinDoc in listOPosts:
    print "postinDoc:",postinDoc
    trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))
print "trainMat:",trainMat
p0V, p1V, pAb= bayes.trainNB0(trainMat, listClasses)
print "p0V:",p0V
print "p1V:",p1V
print "pAb:",pAb
bayes. testingNB()

#4.6.2 测试算法：使用朴素贝叶斯进行交叉验证
bayes.spamTest()

#4.7.1 收集数据：导入RSS源！出现bug
ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
print "ny:",ny
print "sf:",sf
vocabList,pSF,pNY=bayes.localWords(ny,sf)
vocabList,pSF,pNY=bayes.localWords(ny,sf)
print "bayes.getTopWords(ny,sf):",bayes.getTopWords(ny,sf)

#4.7.1 分析数据：显示地域相关的用词！出现bug
bayes.getTopWords(ny,sf)