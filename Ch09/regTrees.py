#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *
import os
import matplotlib.pyplot as plt

# homedir= os.getcwd()+'/machinelearninginaction/ch09/'  #绝对路径
homedir= '' #相对路径


#程序清单9-1　CART算法的实现代码
#第一个函数是loadDataSet()，该函数与其他章节中同名函数功能类似。
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # print "curLine:",curLine
        fltLine = map(float,curLine) #map all elements to float()
        # print "fltLine:",fltLine
        dataMat.append(fltLine)
    # print "dataMat:",dataMat
    return dataMat

#第二个函数是binSplitDataSet()，该函数有3个参数：数据集合、待切分的特征和该特征的某个值。在给定特征和特征值的情况下，该函数通过数组过滤方式将上述数据集合切分得到两个子集并返回。
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

# testMat=mat(eye(4))
# print "testMat:",testMat
# print "testMat[:,1]:",testMat[:,1]
# print "nonzero(testMat[:,1]):",nonzero(testMat[:,1])
# mat0,mat1=binSplitDataSet(testMat,1,0.5)
# print "mat0:",mat0
# print "mat1:",mat1
# print ":",

#程序清单9-2　回归树的切分函数
#第一个函数是regLeaf()，它负责生成叶节点。当chooseBestSplit()函数确定不再对数据进行切分时，将调用该regLeaf()函数来得到叶节点的模型。在回归树中，该模型其实就是目标变量的均值。
def regLeaf(dataSet):#returns the value used for each leaf
    # print "mean(dataSet[:,-1]):",mean(dataSet[:,-1])
    return mean(dataSet[:,-1])

#第二个函数是误差估计函数regErr()。该函数在给定数据上计算目标变量的平方误差。当然也可以先计算出均值，然后计算每个差值再平方。但这里直接调用均方差函数var()更加方便。因为这里需要返回的是总方差，所以要用均方差乘以数据集中样本的个数。
def regErr(dataSet):
    # print "var(dataSet[:,-1]) * shape(dataSet)[0]:", var(dataSet[:,-1]) * shape(dataSet)[0]
    # print "var(dataSet[:,-1]):",var(dataSet[:,-1])
    # print "shape(dataSet)[0]:",shape(dataSet)[0]
    return var(dataSet[:,-1]) * shape(dataSet)[0]

#程序清单9-4　模型树的叶节点生成函数
#linearSolve()，它会被其他两个函数调用。其主要功能是将数据集格式化成目标变量Y和自变量X❶。与第8章一样，X和Y用于执行简单的线性回归。另外在这个函数中也应当注意，如果矩阵的逆不存在也会造成程序异常。
def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

#第二个函数modelLeaf()与程序清单9-2里的函数regLeaf()类似，当数据不再需要切分的时候它负责生成叶节点的模型。该函数在数据集上调用linearSolve()并返回回归系数ws。
def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

#最后一个函数是modelErr()，可以在给定的数据集上计算误差。它与程序清单9-2的函数regErr()类似，会被chooseBestSplit()调用来找到最佳的切分。该函数在数据集上调用linearSolve()，之后返回yHat和Y之间的平方误差。
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

#第三个函数是chooseBestSplit()，它是回归树构建的核心函数。该函数的目的是找到数据的最佳二元切分方式。如果找不到一个“好”的二元切分，该函数返回None并同时调用createTree()方法来产生叶节点，叶节点的值也将返回None。接下来将会看到，在函数chooseBestSplit()中有三种情况不会切分，而是直接创建叶节点。如果找到了一个“好”的切分方式，则返回特征编号和切分特征值。一开始为ops设定了tolS和tolN这两个值。它们是用户指定的参数，用于控制函数的停止时机。其中变量tolS是容许的误差下降值，tolN是切分的最少样本数。
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    # print "tolS:",tolS
    # print "tolN:",tolN
    # print ":",
    #if all the target variables are the same value: quit and return value
    # print "dataSet[:,-1]:",dataSet[:,-1]
    # print "dataSet[:,-1].T:",dataSet[:,-1].T
    # print "dataSet[:,-1].T.tolist():",dataSet[:,-1].T.tolist()
    # print "dataSet[:,-1].T.tolist()[0]:",dataSet[:,-1].T.tolist()[0]
    # print "set(dataSet[:,-1].T.tolist()[0]):",set(dataSet[:,-1].T.tolist()[0])
    # print "len(set(dataSet[:,-1].T.tolist()[0])):",len(set(dataSet[:,-1].T.tolist()[0]))
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        print "len(set(dataSet[:,-1].T.tolist()[0])) == 1"
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    # print "m:",m
    # print "n:",n
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    # print "S:",S
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        # print "featIndex:",featIndex
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            # print "splitVal:", splitVal
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # print "mat0:",mat0
            # print "mat1:",mat1
            # print "shape(mat0)[0]:",shape(mat0)[0]
            # print "shape(mat1)[0]:",shape(mat1)[0]
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            # print "newS:",newS
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
                # print "bestIndex:",bestIndex
                # print "bestValue:",bestValue
                # print "bestS:",bestS
                # print ":",
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        print "(S - bestS) < tolS"
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        print "(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN)"
        return None, leafType(dataSet)
    # print "mat0:",mat0
    # print "mat1:",mat1
    # print "bestIndex:",bestIndex
    # print "bestValue:",bestValue
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

#最后一个函数是树构建函数createTree()，它有4个参数：数据集和其他3个可选参数。这些可选参数决定了树的类型：leafType给出建立叶节点的函数；errType代表误差计算函数；而ops是一个包含树构建所需其他参数的元组。函数createTree()是一个递归函数。该函数首先尝试将数据集分成两个部分，切分由函数chooseBestSplit()完成（这里未给出该函数的实现）。如果满足停止条件，chooseBestSplit()将返回None和某类模型的值❷。如果构建的是回归树，该模型是一个常数。如果是模型树，其模型是一个线性方程。后面会看到停止条件的作用方式。如果不满足停止条件，chooseBestSplit()将创建一个新的Python字典并将数据集分成两份，在这两份数据集上将分别继续递归调用createTree()函数。
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # print "spInd:",feat
    # print "spVal:",val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # print "left("
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    # print ")left"
    # print "right("
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    # print ")right"
    # print "retTree:",retTree
    return retTree  

# myDat=loadDataSet(homedir+'ex00.txt')
# # print "myDat:",myDat
# myMat=mat(myDat)
# # print "myMat:",myMat
# print "createTree(myMat):",createTree(myMat)
# # print ":",

# myDat1=loadDataSet(homedir+'ex0.txt')
# myMat1=mat(myDat1)
# # print "myMat1:",myMat1
# print "createTree(myMat1):",createTree(myMat1)

# myDat2=loadDataSet(homedir+'ex2.txt')
# myMat2=mat(myDat2)
# print "createTree(myMat2):",createTree(myMat2)

#程序清单9-3　回归树剪枝函数
#其中isTree()用于测试输入变量是否是一棵树，返回布尔类型的结果。换句话说，该函数用于判断当前处理的节点是否是叶节点。
def isTree(obj):
    # print "type(obj).__name__:",type(obj).__name__
    return (type(obj).__name__=='dict')

#函数getMean()是一个递归函数，它从上往下遍历树直到叶节点为止。如果找到两个叶节点则计算它们的平均值。
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
        # print "tree['right']:",tree['right']
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
        # print "tree['left']:", tree['left']

    return (tree['left']+tree['right'])/2.0

#程序清单9-3的主函数是prune()，它有两个参数：待剪枝的树与剪枝所需的测试数据testData。prune()函数首先需要确认测试集是否为空❶。一旦非空，则反复递归调用函数prune()对测试数据进行切分。因为树是由其他数据集（训练集）生成的，所以测试集上会有一些样本与原数据集样本的取值范围不同。一旦出现这种情况应当怎么办？数据发生过拟合应该进行剪枝吗？或者模型正确不需要任何剪枝？这里假设发生了过拟合，从而对树进行剪枝。接下来要检查某个分支到底是子树还是节点。如果是子树，就调用函数prune()来对该子树进行剪枝。在对左右两个分支完成剪枝之后，还需要检查它们是否仍然还是子树。如果两个分支已经不再是子树，那么就可以进行合并。具体做法是对合并前后的误差进行比较。如果合并后的误差比不合并的误差小就进行合并操作，反之则不合并直接返回。
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean
        else: return tree
    else: return tree

# myDat2=loadDataSet(homedir+'ex2.txt')
# myMat2=mat(myDat2)
# myTree=createTree(myMat2,ops=(0,1))
# print "myTree:",myTree
# myDatTest=loadDataSet(homedir+'ex2test.txt')
# myMat2Test= mat(myDatTest)
# print "prune(myTree,myMat2Test)",prune(myTree,myMat2Test)



# myMat2=mat(loadDataSet(homedir+'exp2.txt'))
#
# print "createTree(myMat2,modelLeaf,modelErr,(1,10)):",createTree(myMat2,modelLeaf,modelErr,(1,10))


#程序清单9-5　用树回归进行预测的代码
#要对回归树叶节点进行预测，就调用函数regTreeEval()
def regTreeEval(model, inDat):
    return float(model)

#要对模型树节点进行预测时，就调用modelTreeEval()函数。
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

#对于输入的单个数据点或者行向量，函数treeForeCast()会返回一个浮点值。在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。调用函数treeForeCast()时需要指定树的类型，以便在叶节点上能够调用合适的模型。参数modelEval是对叶节点数据进行预测的函数的引用。函数treeForeCast()自顶向下遍历整棵树，直到命中叶节点为止。一旦到达叶节点，它就会在输入数据上调用modelEval()函数，而该函数的默认值是regTreeEval()。
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)

#最后一个函数是createForCast()，它会多次调用treeForeCast()函数。由于它能够以向量形式返回的一组预测值，因此该函数在对整个测试集进行预测时非常有用。
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat



# trainMat= mat(loadDataSet(homedir+'bikeSpeedVsIq_train.txt'))
# testMat= mat(loadDataSet(homedir+'bikeSpeedVsIq_test.txt'))
#
# myTree=createTree(trainMat,ops=(1,20))
# yHat =createForeCast(myTree,testMat[:,0])
# corrcoef1=corrcoef(yHat,testMat[:,1],rowvar= 0)[0,1]
#
# myTree=createTree(trainMat,modelLeaf,modelErr,(1,20))
# yHat=createForeCast(myTree,testMat[:,0],modelTreeEval)
# corrcoef2=corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
#
# ws,X,Y=linearSolve(trainMat)
#
# for i in range(shape(testMat)[0]):
#     yHat[i]= testMat[i,0]* ws[1,0]+ ws[0,0]
#
# corrcoef3=corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
#
# print "corrcoef1:",corrcoef1
# print "corrcoef2:",corrcoef2
# print "ws:",ws
# print "corrcoef3:",corrcoef3
# print ":",