#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on Oct 14, 2010

@author: Peter Harrington
'''
import matplotlib.pyplot as plt

#❶ （以下 三行） 定义 文本 框 和 箭头 格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    # print "numLeafs:",numLeafs
    firstStr = myTree.keys()[0]
    # print "firstStr:",firstStr
    secondDict = myTree[firstStr]
    # print "secondDict:",secondDict
    for key in secondDict.keys():
        # print "key:", key
        # print "secondDict[key]:", secondDict[key]
        # print "type(secondDict[key]):", type(secondDict[key])
        # print "type(secondDict[key]).__name__:", type(secondDict[key]).__name__
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            # print "**************"
            numLeafs += getNumLeafs(secondDict[key])
            # print "!!!!!!!!!!!!!!"
        else:   numLeafs +=1
        # print "numLeafs:",numLeafs
    return numLeafs

#获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    # print "maxDepth:",maxDepth
    firstStr = myTree.keys()[0]
    # print "firstStr:",firstStr
    secondDict = myTree[firstStr]
    # print "secondDict:",secondDict
    for key in secondDict.keys():
        # print "key:", key
        # print "secondDict[key]:", secondDict[key]
        # print "type(secondDict[key]):", type(secondDict[key])
        # print "type(secondDict[key]).__name__:", type(secondDict[key]).__name__
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            # print "**************"
            thisDepth = 1 + getTreeDepth(secondDict[key])
            # print "!!!!!!!!!!!!!!"
        else:   thisDepth = 1
        # print "thisDepth:",thisDepth
        if thisDepth > maxDepth: maxDepth = thisDepth
        # print "maxDepth:",maxDepth
    return maxDepth

#❷ （以下 两行） 绘制 带 箭头 的 注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

#❶ （以下 四行） 在 父子 节点 间 填充 文本 信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    # print "numLeafs:",numLeafs
    depth = getTreeDepth(myTree)
    # print "depth:",depth
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    # print "firstStr:",firstStr
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    # print "plotTree.xOff:", plotTree.xOff
    # print "plotTree.totalW:", plotTree.totalW
    # print "plotTree.yOff:", plotTree.yOff
    # print "cntrPt:",cntrPt
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # print "secondDict:",secondDict
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    # print "plotTree.yOff:",plotTree.yOff
    for key in secondDict.keys():
        # print "key:", key
        # print "secondDict[key]:", secondDict[key]
        # print "type(secondDict[key]):", type(secondDict[key])
        # print "type(secondDict[key]).__name__:", type(secondDict[key]).__name__
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            # print "**************"
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
            # print "!!!!!!!!!!!!!!"
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

# def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()

#预先存储树的信息
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


