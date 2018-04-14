#!/usr/bin/python
# -*- coding:utf-8 -*-
import adaboost
from numpy import *
import os

# homedir= os.getcwd()+'/machinelearninginaction/ch07/'  #绝对路径
homedir= '' #相对路径

#7.3　基于单层决策树构建弱分类器
datMat,classLabels=adaboost.loadSimpData()
D=mat(ones((5,1))/5)
print "datMat:",datMat
print "classLabels:",classLabels
print "D:",D
print ":",
adaboost.buildStump(datMat,classLabels, D)

#7.4　完整AdaBoost算法的实现
classifierArr=adaboost.adaBoostTrainDS(datMat,classLabels,9)
print "classifierArr:",classifierArr

#7.5　测试算法：基于AdaBoost的分类
datMat,classLabels=adaboost.loadSimpData()
classifierArr=adaboost.adaBoostTrainDS(datMat,classLabels,30)
print "分类1:",adaboost.adaClassify([0,0],classifierArr)
print "分类2:",adaboost.adaClassify([[5,5],[0,0]],classifierArr)

#7.6　示例：在一个难数据集上应用AdaBoost
datArr,labelArr=adaboost.loadDataSet(homedir+'horseColicTraining2.txt')
print "datArr:",datArr
print "labelArr:",labelArr
classifierArray= adaboost.adaBoostTrainDS(datArr,labelArr,500)
testArr,testLabelArr=adaboost.loadDataSet(homedir+'horseColicTest2.txt')
prediction10 =adaboost.adaClassify(testArr,classifierArray)
errArr= mat(ones((67,1)))
errorNum=errArr[prediction10!=mat(testLabelArr).T].sum()
errorrate=errorNum/67
print "prediction10:",prediction10
print "errArr:",errArr
print "errorrate:",errorrate

#7.7.1　其他分类性能度量指标：正确率、召回率及ROC曲线
datArr,labelArr=adaboost.loadDataSet(homedir+'horseColicTraining2.txt')
# # print "datArr:",datArr
# # print "labelArr:",labelArr
classifierArray, aggClassEst= adaboost.adaBoostTrainDS(datArr,labelArr,10)
adaboost.plotROC(aggClassEst.T,labelArr)