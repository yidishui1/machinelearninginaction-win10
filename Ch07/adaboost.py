#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *
import os

# homedir= os.getcwd()+'/machinelearninginaction/ch07/'  #绝对路径
homedir= '' #相对路径

#建立初始数据
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

#从文件中导入数据
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#单层决策树生成函数
#通过阈值比较对数据进行分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    
#将会遍历stumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树。
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    # print "m:",m
    # print "n:",n
    # print ":",
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        # print "i:",i
        # print "rangeMin:",rangeMin
        # print "rangeMax:",rangeMax
        # print "stepSize:",stepSize
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            # print "j:",j
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                # print "inequal:",inequal
                # print "threshVal:",threshVal
                # print "predictedVals:",predictedVals
                errArr = mat(ones((m,1)))
                # print "errArr:",errArr
                errArr[predictedVals == labelMat] = 0
                # print "errArr:",errArr
                weightedError = D.T*errArr  #calc total error multiplied by D
                # print "weightedError:",weightedError
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                    # print "minError:",minError
                    # print "bestClasEst:",bestClasEst
                    # print "bestStump:",bestStump
    # print "minError:", minError
    # print "bestClasEst:", bestClasEst
    # print "bestStump:", bestStump
    return bestStump,minError,bestClasEst

#程序清单7-2　基于单层决策树的AdaBoost训练过程
#完整的AdaBoost算法的核心函数，通过不断优化D值来达到训练数据错误率为0
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    # print " weakClassArr:", weakClassArr
    # print "m:",m
    # print "D:",D
    # print "aggClassEst:",aggClassEst
    for i in range(numIt):
        # print "i:", i
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        # print "bestStump:", bestStump
        # print "error:", error
        # print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        # print "bestStump:", bestStump
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        # print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        # print "expon:",expon
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        # print "D:",D
        D = D/D.sum()
        # print "D:",D
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        # print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        # print "aggErrors:",aggErrors
        errorRate = aggErrors.sum()/m
        if errorRate == 0.0: break
    print "total error: ", errorRate
    # print "weakClassArr:", weakClassArr
    # print "aggClassEst:", aggClassEst
    # return weakClassArr
    return weakClassArr, aggClassEst

#程序清单7-3　AdaBoost分类函数
#就可以利用它基于adaboostTrainDS()中的弱分类器对数据进行分类。
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    # print "dataMatrix:",dataMatrix
    # print "m:",m
    # print "aggClassEst:",aggClassEst
    # print "len(classifierArr):", len(classifierArr)
    for i in range(len(classifierArr)):
        # print "i:",i
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print "classEst:",classEst
        # print "aggClassEst:",aggClassEst
    # print "sign(aggClassEst):", sign(aggClassEst)
    return sign(aggClassEst)

#ROC曲线的绘制及AUC计算函数
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    # print "numPosClas:",numPosClas
    # print "yStep:",yStep
    # print "xStep:",xStep
    # print "sortedIndicies:",sortedIndicies
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        # print "index:",index
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        # print "delX:",delX
        # print "delY:",delY
        # print "cur[1]:", cur[1]
        # print "ySum:",ySum
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    # print "ySum:", ySum
    # print "xStep:", xStep
    print "the Area Under the Curve is: ",ySum*xStep
