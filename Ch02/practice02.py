#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on Oct 27, 2010

@author: Peter
'''
from numpy import *
import kNN
import matplotlib
import matplotlib.pyplot as plt
import os

import operator
from os import listdir

# homedir= os.getcwd()+'/machinelearninginaction/ch02/'  #绝对路径
homedir= '' #相对路径

#2.1.1 使用python导入数据
group,labels=kNN.createDataSet()
print 'group:',group
print 'labels:',labels

#2.1.2 实施kNN分类算法
distance=kNN.classify0([0,0],group,labels,3)
print 'distance:',distance

#2.2.1 准备数据：从文本中解析数据
datingDataMat,datingLabels = kNN.file2matrix(homedir+'datingTestSet2.txt')
print 'datingDataMat:',datingDataMat
print 'datingLabels:',datingLabels

#2.2.2 分析数据：使用Matplotlib创建散点图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

#2.2.3 准备数据：归一化数值
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print 'normMat:',normMat
print 'ranges:',ranges
print 'minVals:',minVals

#2.2.4 测试算法：作为完整程序验证分类器
kNN.datingClassTest()


#2.2.5 使用算法：构建完整可用系统
kNN.classifyPerson()

#2.3.1准备数据：将图像转化为测试向量
testVector=kNN.img2vector(homedir+'testDigits/0_13.txt')
print 'testVector[0,0:31]:',testVector[0,0:31]

#2.3.2 测试算法：使用k近邻算法识别手写数字
kNN. handwritingClassTest()

# inX=[0,0]
# dataSet=group
# k=3
#
# dataSetSize = dataSet.shape[0]
# print 'dataSetSize:',dataSetSize
# diffMat = tile(inX, (dataSetSize, 1)) - dataSet
# print 'diffMat:',diffMat
# sqDiffMat = diffMat ** 2
# print 'sqDiffMat:',sqDiffMat
# sqDistances = sqDiffMat.sum(axis=1)
# print 'sqDistances:',sqDistances
# distances = sqDistances ** 0.5
# print 'distances:',distances
# sortedDistIndicies = distances.argsort()
# print 'sortedDistIndicies:',sortedDistIndicies
# classCount = {}
# for i in range(k):
#     voteIlabel = labels[sortedDistIndicies[i]]
#     print 'voteIlabel:', voteIlabel
#     classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
#     print 'classCount:', classCount
# sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
# print 'sortedClassCount:',sortedClassCount


# minVals = datingDataMat.min(0)
# print 'minVals:',minVals
# maxVals = datingDataMat.max(0)
# print 'maxVals:',maxVals
# ranges = maxVals - minVals
# print 'ranges:',ranges
# normDataSet = zeros(shape(datingDataMat))
# print 'normDataSet:',normDataSet
# m = datingDataMat.shape[0]
# print 'm:',m
# normDataSet = datingDataMat - tile(minVals, (m, 1))
# print 'normDataSet:',normDataSet
# normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
# print 'normDataSet:',normDataSet

# hoRatio = 0.50  # hold out 10%

# m = normMat.shape[0]
# print 'm:',m
# numTestVecs = int(m * hoRatio)
# print 'numTestVecs:',numTestVecs
# errorCount = 0.0
# for i in range(numTestVecs):
#     classifierResult = kNN.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
#     print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
#     if (classifierResult != datingLabels[i]): errorCount += 1.0
# print "the total error rate is: %f" % (errorCount / float(numTestVecs))
# print errorCount

# ax.axis([-2,25,-0.2,2.0])
# plt.xlabel('Percentage of Time Spent Playing Video Games')
# plt.ylabel('Liters of Ice Cream Consumed Per Week')


# ax.axis([-5000,100000,-2,25])
# plt.xlabel('Frequent Flyier Miles Earned Per Year')
# plt.ylabel('Percentage of Time Spent Playing Video Games')
#
# plt.show()
