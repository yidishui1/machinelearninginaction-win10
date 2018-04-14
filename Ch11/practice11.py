#!/usr/bin/python
# -*- coding:utf-8 -*-
from numpy import *
import os
import apriori

# homedir= os.getcwd()+'/machinelearninginaction/ch11/'
homedir=''

#11.3.1　生成候选项集
# dataSet=apriori.loadDataSet()
# C1=apriori.createC1(dataSet)
# D= map(set,dataSet)
# L1,suppData0=apriori.scanD(D,C1,0.5)
# print "dataSet:",dataSet
# print "C1:",C1
# print "D:",D
# print "L1:",L1
# print "suppData0:",suppData0
# print ":",

#11.3.2　组织完整的Apriori算法
# dataSet=apriori.loadDataSet()
# L, suppData=apriori.apriori(dataSet)
# print "dataSet:",dataSet
# print "L:",L
# print "suppData:",suppData
# print "apriori.aprioriGen(L[0],2):",apriori.aprioriGen(L[0],2)
# L,suppData=apriori.apriori(dataSet,minSupport=0.7)
# print "L:",L
# print "suppData:",suppData

#11.4　从频繁项集中挖掘关联规则
# dataSet=apriori.loadDataSet()
# L, suppData=apriori.apriori(dataSet,minSupport= 0.5)
# rules=apriori.generateRules(L,suppData,minConf= 0.5)
# print "dataSet:",dataSet
# print "L:",L
# print "suppData:",suppData
# print "rules:",rules
# print ":",

# #11.5　示例：发现国会投票中的模式
# # from time import sleep
# # from votesmart import votesmart
# # votesmart. apikey = '49024thereoncewasamanfromnantucket94040'
# # bills = votesmart. votes. getBillsByStateRecent()
# # for bill in bills:
# # 	print bill. title, bill. billId
# # bill = votesmart. votes. getBill( 11820)
# # bill. actions
# # for action in bill. actions:
# # 	if action. stage==' Passage':
# # 		print action. actionId
# # voteList = votesmart. votes. getBillActionVotes( 31670)
# # voteList[22]
# # voteList[21]
# # actionIdList, billTitles = apriori. getActionIds()
# # transDict, itemMeaning= apriori. getTransList( actionIdList[: 2], billTitles[: 2])
# # transDict.keys()[ 6]
# # for item in transDict[' Doyle, Michael 'Mike'']:
# # 	print itemMeaning[ item]
# # transDict, itemMeaning= apriori. getTransList( actionIdList, billTitles)
# # dataSet = [transDict[ key] for key in transDict. keys()]
#
# #11.5.2　测试算法：基于美国国会投票记录挖掘关联规则
# # L, suppData= apriori.apriori(dataSet,minSupport= 0.5)
# # print "L:",L
# # L, suppData= apriori.apriori(dataSet,minSupport= 0.3)
# # print "len(L):",len(L)
# # print "L[3]:",L[3]
# # rules = apriori.generateRules(L,suppData)
# # print "rules:",rules
# # rules = apriori.generateRules(L,suppData,minConf=0.95)
# # print "rules:",rules
# # rules = apriori.generateRules(L,suppData,minConf=0.99)
# # print "rules:",rules
# # # print "itemMeaning[26]:",itemMeaning[26]
# # # print "itemMeaning[3]:",itemMeaning[3]
# # # print "itemMeaning[9]:",sitemMeaning[9]
#
# #11.6　示例：发现毒蘑菇的相似特征
mushDatSet= [line.split() for line in open(homedir+'mushroom.dat').readlines()]
L, suppData=apriori.apriori(mushDatSet,minSupport= 0.4)
print "mushDatSet:",mushDatSet
print "L:",L
print "suppData:",suppData
for item in L[1]:
    if item.intersection('2'): print item
for item in L[3]:
    if item.intersection('2'): print item