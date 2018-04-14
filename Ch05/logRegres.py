#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *
import os
# homedir= os.getcwd()+'/machinelearninginaction/ch05/'  #绝对路径
homedir= '' #相对路径

#logistic回归梯度上升优化算法
#打开文本文件testSet.txt 并逐行读取。
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open(homedir+'testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # print "lineArr:",lineArr
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    # print "dataMat:",dataMat
    # print "labelMat:",labelMat
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升算法的实际工作示在这里完成
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    # print "dataMatrix:", dataMatrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    # print "labelMat:",labelMat
    m,n = shape(dataMatrix)
    # print "m:",m
    # print "n:",n
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    # print "weights:",weights
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    # print "h:", h
    # print "error:", error
    # print "weights:", weights
    return weights

#画出数据集和logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    # print "dataMat:",dataMat
    # print "labelMat:",labelMat
    dataArr = array(dataMat)
    # print "dataArr:",dataArr
    n = shape(dataArr)[0]
    # print "n:",n
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    # print "xcord1:",xcord1
    # print "ycord1:",ycord1
    # print "xcord2:",xcord2
    # print "ycord2:",ycord2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

#随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    print "m:", m
    print "n:", n
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    print "weights:", weights
    return weights

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=500):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        # print "dataIndex:",dataIndex
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #alpha每次迭代时需要调整apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
        # print "weights:", weights
    return weights

#logistic回归分类函数
#它以回归系数和特征向量作为输入来计算对应的Sigmoid值。如果Sigmoid值大于0.5函数返回1，否则返回0。
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

#是用来打开测试集和训练集，并对数据进行格式化处理的函数
def colicTest():
    frTrain = open(homedir+'horseColicTraining.txt'); frTest = open(homedir+'horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # print "lineArr:", lineArr
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # print "trainingSet:",trainingSet
    # print "trainingLabels:",trainingLabels
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

#其功能是调用函数colicTest()10次并求结果的平均值。
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
