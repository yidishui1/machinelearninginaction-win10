#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import trees
import treePlotter

# homedir= os.getcwd()+'/machinelearninginaction/ch03/'  #绝对路径
homedir= '' #相对路径

#3.1.1 信息增益
myDat,labels= trees. createDataSet()
print "计算香农熵:",trees. calcShannonEnt(myDat)
myDat[0][-1]=' maybe'
print "计算香农熵:",trees. calcShannonEnt(myDat)

#3.1.2 划分数据集
myDat,labels=trees.createDataSet()
trees.splitDataSet( myDat,0,1)
trees.splitDataSet( myDat,0,0)
print "选择最好的数据集划分方式:",trees.chooseBestFeatureToSplit(myDat)

#3.1.3 递归构建决策树
myDat,labels=trees.createDataSet()
myTree =trees.createTree(myDat,labels)
print "myTree:",myTree

#3.2.1 Matplotlib注解
treePlotter.createPlot()

#3.2.2 构造注解树
treePlotter.retrieveTree(1)
myTree=treePlotter.retrieveTree(0)
print "获取叶节点的数目:",treePlotter.getNumLeafs(myTree)
print "获取树的层数:",treePlotter.getTreeDepth(myTree)
treePlotter.createPlot(myTree)
myTree['no surfacing'][3]='maybe'
print "myTree:",myTree
treePlotter.createPlot(myTree)

#3.3.1 测试算法：使用决策树执行分类
myDat,labels=trees.createDataSet()
print "labels:",labels
myTree=treePlotter.retrieveTree(0)
print "myTree:",myTree
print "分类1:",trees.classify(myTree,labels,[1,0])
print "分类2:",trees.classify(myTree,labels,[1,1])

#3.3.2  决策树的存储
trees.storeTree(myTree,homedir+'classifierStorage.txt')
print "决策树调取:",trees.grabTree(homedir+'classifierStorage.txt')
print ":",
print ":",

#3.4 示例：使用决策树预测隐形眼镜类型
fr=open(homedir+'lenses.txt')
print 'fr:',fr
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
print 'lenses:',lenses
lensesLabels=[' age', 'prescript', 'astigmatic', 'tearRate']
print 'lensesLabels:',lensesLabels
lensesTree=trees.createTree(lenses,lensesLabels)
treePlotter.createPlot(lensesTree)
