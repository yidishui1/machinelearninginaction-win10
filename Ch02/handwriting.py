#!/usr/bin/python
# -*- coding:utf-8 -*-
from numpy import *
import kNN
import matplotlib
import matplotlib.pyplot as plt
import os

import operator
from os import listdir

# homedir= os.getcwd()+'/machinelearninginaction/ch02/'  #绝对路径
homedir= '' #相对路径
# testVector=kNN.img2vector(homedir+'testDigits/0_13.txt')
# print 'testVector[0,0:31]:',testVector[0,0:31]

hwLabels = []
trainingFileList = listdir(homedir+'trainingDigits')  # load the training set
m = len(trainingFileList)
trainingMat = zeros((m, 1024))
# print 'trainingFileList:',trainingFileList
# print 'm:',m
# print 'trainingMat:',trainingMat
for i in range(m):
    fileNameStr = trainingFileList[i]
    fileStr = fileNameStr.split('.')[0]  # take off .txt
    classNumStr = int(fileStr.split('_')[0])
    hwLabels.append(classNumStr)
    trainingMat[i, :] =kNN.img2vector(homedir+'trainingDigits/%s' % fileNameStr)
    # print 'fileNameStr:', fileNameStr
    # print 'fileStr:', fileStr
    # print 'classNumStr:', classNumStr
    # print 'hwLabels:', hwLabels
    # print 'trainingMat[i, :]:', trainingMat[i, :]
testFileList = listdir(homedir+'testDigits')  # iterate through the test set
errorCount = 0.0
mTest = len(testFileList)
# print 'testFileList:',testFileList
# print 'mTest:',mTest
for i in range(mTest):
    fileNameStr = testFileList[i]
    fileStr = fileNameStr.split('.')[0]  # take off .txt
    classNumStr = int(fileStr.split('_')[0])
    vectorUnderTest =kNN.img2vector(homedir+'testDigits/%s' % fileNameStr)
    classifierResult =kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
    if (classifierResult != classNumStr): errorCount += 1.0
    # print 'fileNameStr:', fileNameStr
    # print 'fileStr:', fileStr
    # print 'classNumStr:', classNumStr
    # print 'vectorUnderTest:', vectorUnderTest
    # print 'classifierResult:', classifierResult
    print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
    print 'errorCount:', errorCount
print "\nthe total number of errors is: %d" % errorCount
print "\nthe total error rate is: %f" % (errorCount / float(mTest))