'''
Created on Jun 14, 2011

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import pca
import os

homedir=os.getcwd()+'/machinelearninginaction/ch13/'

dataMat = pca.replaceNanWithMean()
# print "dataMat:",dataMat
#below is a quick hack copied from pca.pca()
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
eigValInd = eigValInd[::-1]#reverse
# print "eigValInd:",eigValInd
sortedEigVals = eigVals[eigValInd]
total = sum(sortedEigVals)
varPercentage = sortedEigVals/total*100
# print "sortedEigVals:",sortedEigVals
# print "total:",total
# print "varPercentage:",varPercentage

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, 21), varPercentage[:20], marker='^')
plt.xlabel('Principal Component Number')
plt.ylabel('Percentage of Variance')
plt.show()