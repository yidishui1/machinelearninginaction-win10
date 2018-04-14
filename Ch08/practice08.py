#!/usr/bin/python
# -*- coding:utf-8 -*-
import regression
from numpy import *
import matplotlib.pyplot as plt
import os

# homedir= os.getcwd()+'/machinelearninginaction/ch08/'  #绝对路径
homedir= '' #相对路径

# 8.1　用线性回归找到最佳拟合直线
xArr,yArr=regression.loadDataSet(homedir+'ex0.txt')
print ":",
print "xArr:",xArr
print "yArr:",yArr
ws=regression.standRegres(xArr,yArr)
print "ws:",ws

xMat= mat(xArr)
yMat= mat(yArr)
yHat = xMat* ws

fig = plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xCopy=xMat.copy()
xCopy.sort(0)
yHat=xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()

yHat = xMat* ws
print "corrcoef(yHat.T,yMat):",corrcoef(yHat.T,yMat)

#8.2　局部加权线性回归
xArr,yArr=regression.loadDataSet(homedir+'ex0.txt')
print "yArr[0]:",yArr[0]
print "xArr[0]:",xArr[0]
print "regression.lwlr(xArr[0],xArr,yArr,1.0):",regression.lwlr(xArr[0],xArr,yArr,1.0)
print "regression.lwlr(xArr[0],xArr,yArr,0.001):",regression.lwlr(xArr[0],xArr,yArr,0.001)
print ":",
yHat=regression.lwlrTest(xArr,xArr,yArr,0.02)
xMat=mat(xArr)
srtInd=xMat[:,1].argsort(0)
xSort=xMat[srtInd][:,0,:]
# print "xMat",xMat
# print "srtInd:",srtInd
# print "xSort:",xSort
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
plt.show()

#8.3　示例：预测鲍鱼的年龄
abX,abY=regression.loadDataSet(homedir+'abalone.txt')
print "abX:",abX
print "abY:",abY
yHat01=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
print "regression.rssError(abY[0:99],yHat01.T):",regression.rssError(abY[0:99],yHat01.T)
print "regression.rssError(abY[0:99],yHat1.T):",regression.rssError(abY[0:99],yHat1.T)
print "regression.rssError(abY[0:99],yHat10.T):",regression.rssError(abY[0:99],yHat10.T)
yHat01=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
print "regression.rssError(abY[100:199],yHat01.T):",regression.rssError(abY[100:199],yHat01.T)
yHat1=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
print "regression.rssError(abY[100:199],yHat1.T):",regression.rssError(abY[100:199],yHat1.T)
yHat10=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
print "regression.rssError(abY[100:199],yHat10.T):",regression.rssError(abY[100:199],yHat10.T)
ws=regression.standRegres(abX[0:99],abY[0:99])
yHat=mat(abX[100:199])*ws
print "regression.rssError(abY[100:199],yHat.T.A):",regression.rssError(abY[100:199],yHat.T.A)

#8.4.1　岭回归
abX,abY=regression.loadDataSet(homedir+'abalone.txt')
ridgeWeights=regression.ridgeTest(abX,abY)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()

#8.4.3　前向逐步回归
xArr,yArr=regression.loadDataSet(homedir+'abalone.txt')
regression.stageWise(xArr,yArr,0.01,200)
regression.stageWise(xArr,yArr,0.001,5000)
xMat=mat(xArr)
yMat=mat(yArr).T
xMat=regression.regularize(xMat)
yM=mean(yMat,0)
yMat=yMat-yM
weights=regression.standRegres(xMat,yMat.T)
weights.T
print "weights.T:",weights.T

#8.6.1　收集数据：使用Google购物的API
# from time import sleep
# import json
# import urllib2
# lgX=[];lgY=[]
# regression.setDataCollect(lgX,lgY)