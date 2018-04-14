#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import fpGrowth

# homedir=os.getcwd()+'/machinelearninginaction/ch12/'
homedir=''

#12.2.1　创建FP树的数据结构
# rootNode=fpGrowth.treeNode('pyramid',9,None)
# rootNode.children['eye']=fpGrowth.treeNode('eye',13,None)
# rootNode.disp()
# rootNode.children['phoenix']=fpGrowth.treeNode('phoenix',3,None)
# rootNode.disp()

#12.2.2　构建FP树
# simpDat=fpGrowth.loadSimpDat()
# print "simpDat:",simpDat
# # print ":",
# initSet=fpGrowth.createInitSet(simpDat)
# print "initSet:",initSet
# myFPtree,myHeaderTab=fpGrowth.createTree(initSet,3)
# # print "myFPtree:",myFPtree
# # print "myHeaderTab:",myHeaderTab
# myFPtree.disp()

# # 12.3　从一棵FP树中挖掘频繁项集
# # 12.3.1　抽取条件模式基
# print ":",fpGrowth.findPrefixPath('x',myHeaderTab['x'][1])
# print ":",fpGrowth.findPrefixPath('z',myHeaderTab['z'][1])
# print ":",fpGrowth.findPrefixPath('r',myHeaderTab['r'][1])

#12.3.2　创建条件FP树
# freqItems = []
# fpGrowth.mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)
# print "freqItems:",freqItems

# #12.5　示例：从新闻网站点击流中挖掘
parsedDat=[line.split() for line in open(homedir+'kosarak.dat').readlines()]
# print "parsedDat:",parsedDat
initSet=fpGrowth.createInitSet(parsedDat)
# print "initSet:",initSet
# print ":",
myFPtree,myHeaderTab=fpGrowth.createTree(initSet,100000)
myFPtree.disp()
print "myHeaderTab:",myHeaderTab
myFreqList=[]
fpGrowth.mineTree(myFPtree,myHeaderTab,100000,set([]),myFreqList)
# print "len(myFreqList):",len(myFreqList)
print "myFreqList:",myFreqList