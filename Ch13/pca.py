#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *
import os

homedir=os.getcwd()+'/machinelearninginaction/ch13/'

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    # print "meanVals:",meanVals
    # print "meanRemoved:",meanRemoved
    # print "covMat:",covMat
    eigVals,eigVects = linalg.eig(mat(covMat))
    # print "eigVals:",eigVals
    # print "eigVects:",eigVects
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    # print "eigValInd:",eigValInd
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    # print "eigValInd:",eigValInd
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    # print "redEigVects:", redEigVects
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean(): 
    datMat = loadDataSet(homedir+'secom.data', ' ')
    numFeat = shape(datMat)[1]
    # print "datMat[:,1]:",datMat[:,1]
    # print "datMat[:,1].A:",datMat[:,1].A
    # print "~isnan(datMat[:,1].A):",~isnan(datMat[:,1].A)
    # print "nonzero(~isnan(datMat[:,1].A)):",nonzero(~isnan(datMat[:,1].A))
    # print "nonzero(~isnan(datMat[:,1].A))[0]:",nonzero(~isnan(datMat[:,1].A))[0]
    # print "datMat[nonzero(~isnan(datMat[:,i].A))[0],1]:",datMat[nonzero(~isnan(datMat[:,1].A))[0],1]
    # print "isnan(datMat[:,1].A):",isnan(datMat[:,1].A)
    # print "nonzero(isnan(datMat[:,1].A)):",nonzero(isnan(datMat[:,1].A))
    # print "nonzero(isnan(datMat[:,1].A))[0]:",nonzero(isnan(datMat[:,1].A))[0]
    # print "datMat[nonzero(isnan(datMat[:,i].A))[0],1]:",datMat[nonzero(isnan(datMat[:,1].A))[0],1]

    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat

#在NumPy中实现PCA
# dataMat=loadDataSet(homedir+'testSet.txt')
# # print "dataMat:",dataMat
# lowDMat,reconMat = pca(dataMat, 0)
# print "lowDMat:",lowDMat
# print "reconMat:",reconMat
# print "shape(lowDMat):",shape(lowDMat)
# import matplotlib
# import matplotlib. pyplot as plt
# fig=plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
# ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
# plt.show()

