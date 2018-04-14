#!/usr/bin/python
# -*- coding:utf-8 -*-

import svmMLiA
from numpy import *
from time import sleep
import os

# homedir= os.getcwd()+'/machinelearninginaction/ch06/'  #绝对路径
homedir= '' #相对路径

#6.3.2 应用简化版SMO算法处理小规模数据集
dataArr, labelArr = svmMLiA.loadDataSet(homedir+'testSet.txt')
print "labelArr:",labelArr
print ":",

b,alphas = svmMLiA.smoSimple(dataArr,labelArr,0.6,0.001,40)
print "b:",b
print "alphas[alphas>0]:",alphas[alphas>0]

for i in range(100):
    if alphas[i] > 0.0: print dataArr[i],labelArr[i]

#6.4 利用完整Plat SMO算法加速优化
dataArr,labelArr=svmMLiA.loadDataSet(homedir+'testSet.txt')
b,alphas=svmMLiA.smoP(dataArr,labelArr,0.6,0.001,40)
ws=svmMLiA.calcWs(alphas,dataArr,labelArr)
print "ws:",ws
datMat=mat(dataArr)
print ":",datMat[0]*mat(ws)+ b
print "labelArr[0]:",labelArr[0]

#6.5.3 在测试中使用核函数
svmMLiA.testRbf()

#6.6　示例:手写识别问题回顾
svmMLiA.testDigits(('rbf',20))