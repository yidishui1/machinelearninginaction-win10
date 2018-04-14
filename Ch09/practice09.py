#!/usr/bin/python
# -*- coding:utf-8 -*-
from numpy import *
import os
import matplotlib.pyplot as plt
import regTrees
# homedir= os.getcwd()+'/machinelearninginaction/ch09/'  #绝对路径
homedir= '' #相对路径


#9.2　连续和离散型特征的树的构建
testMat=mat(eye(4))
print "testMat:",testMat
print "testMat[:,1]:",testMat[:,1]
print "nonzero(testMat[:,1]):",nonzero(testMat[:,1])
mat0,mat1=regTrees.binSplitDataSet(testMat,1,0.5)
print "mat0:",mat0
print "mat1:",mat1
print ":",

#9.3.1　构建树
myDat=regTrees.loadDataSet(homedir+'ex00.txt')
# print "myDat:",myDat
myMat=mat(myDat)
# print "myMat:",myMat
print "regTrees.createTree(myMat):",regTrees.createTree(myMat)
# print ":",

myDat1=regTrees.loadDataSet(homedir+'ex0.txt')
myMat1=mat(myDat1)
# print "myMat1:",myMat1
print "regTrees.createTree(myMat1):",regTrees.createTree(myMat1)

#9.4.1　预剪枝
myDat2=regTrees.loadDataSet(homedir+'ex2.txt')
myMat2=mat(myDat2)
print "regTrees.createTree(myMat2):",regTrees.createTree(myMat2)

#9.4.2　后剪枝
myDat2=regTrees.loadDataSet(homedir+'ex2.txt')
myMat2=mat(myDat2)
myTree=regTrees.createTree(myMat2,ops=(0,1))
print "myTree:",myTree
myDatTest=regTrees.loadDataSet(homedir+'ex2test.txt')
myMat2Test= mat(myDatTest)
print "regTrees.prune(myTree,myMat2Test)",regTrees.prune(myTree,myMat2Test)


#9.5　模型树
myMat2=mat(regTrees.loadDataSet(homedir+'exp2.txt'))
print "regTrees.createTree(myMat2,regTrees.modelLeaf,regTrees.modelErr,(1,10)):",regTrees.createTree(myMat2,regTrees.modelLeaf,regTrees.modelErr,(1,10))

#9.6　示例：树回归与标准回归的比较
trainMat= mat(regTrees.loadDataSet(homedir+'bikeSpeedVsIq_train.txt'))
testMat= mat(regTrees.loadDataSet(homedir+'bikeSpeedVsIq_test.txt'))

myTree=regTrees.createTree(trainMat,ops=(1,20))
yHat =regTrees.createForeCast(myTree,testMat[:,0])
corrcoef1=corrcoef(yHat,testMat[:,1],rowvar= 0)[0,1]

myTree=regTrees.createTree(trainMat,regTrees.modelLeaf,regTrees.modelErr,(1,20))
yHat=regTrees.createForeCast(myTree,testMat[:,0],regTrees.modelTreeEval)
corrcoef2=corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]

ws,X,Y=regTrees.linearSolve(trainMat)

for i in range(shape(testMat)[0]):
    yHat[i]= testMat[i,0]* ws[1,0]+ ws[0,0]

corrcoef3=corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]

print "corrcoef1:",corrcoef1
print "corrcoef2:",corrcoef2
print "ws:",ws
print "corrcoef3:",corrcoef3
print ":",