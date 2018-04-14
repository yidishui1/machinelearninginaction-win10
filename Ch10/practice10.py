#!/usr/bin/python
# -*- coding:utf-8 -*-
import kMeans
from numpy import *
import os
from time import sleep
import urllib
import json
# homedir= os.getcwd()+'/machinelearninginaction/ch10/'  #绝对路径
homedir= '' #相对路径


#10.1　k均值聚类算法
datMat= mat(kMeans.loadDataSet(homedir+'testSet.txt'))
myCentroids,clustAssing=kMeans.kMeans(datMat,4)

print "datMat:",datMat
print "min(datMat[:,0]):",min(datMat[:,0])
print "min(datMat[:,1]):",min(datMat[:,1])
print "max(datMat[:,0]):",max(datMat[:,0])
print "max(datMat[:,1]):",max(datMat[:,1])
print "randCent(datMat,2):",kMeans.randCent(datMat,2)
print "distEclud( datMat[ 0], datMat[ 1]):",kMeans.distEclud( datMat[ 0], datMat[ 1])
print "myCentroids:",myCentroids
print "clustAssing:",clustAssing
print ":",
print ":",

#10.3　二分k均值算法
datMat3=mat(kMeans.loadDataSet(homedir+'testSet2.txt'))
centList,myNewAssments=kMeans.biKmeans(datMat3,3)
print "datMat3:",datMat3
print "centList:",centList
print "myNewAssments:",myNewAssments

#10.4.1　Yahoo!PlaceFinder API
# geoResults=kMeans.geoGrab('1 VA Center', 'Augusta, ME')
# print "geoResults:",geoResults
# print "geoResults['ResultSet']['Error']:",geoResults['ResultSet']['Error']
# print "geoResults['ResultSet']['Results'][0]['longitude']:",geoResults['ResultSet']['Results'][0]['longitude']
# print "kMeans.massPlaceFind(homedir+'portlandClubs.txt'):",kMeans.massPlaceFind(homedir+'portlandClubs.txt')


#10.4.2　对地理坐标进行聚类
print "kMeans.clusterClubs(5):",kMeans.clusterClubs(5)