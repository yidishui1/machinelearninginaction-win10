#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *
import os

# homedir= os.getcwd()+'/machinelearninginaction/ch11/'
homedir=''

#程序清单11-1　Apriori算法中的辅助函数
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

#不言自名，函数createC1()将构建集合C1。C1是大小为1的所有候选项集的集合。Apriori算法首先构建集合C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。那些满足最低要求的项集构成集合L1。而L1中的元素相互组合构成C2，C2再进一步过滤变为L2。到这里，我想读者应该明白了该算法的主要思路。
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return map(frozenset, C1)#use frozen set so we
                            #can use it as a key in a dict    

#第二个函数是scanD()，它有三个参数，分别是数据集、候选项集列表Ck以及感兴趣项集的最小支持度minSupport。该函数用于从C1生成L1。另外，该函数会返回一个包含支持度值的字典以备后用。scanD()函数首先创建一个空字典ssCnt，然后遍历数据集中的所有交易记录以及C1中的所有候选集。如果C1中的集合是记录的一部分，那么增加字典中对应的计数值。这里字典的键就是集合。当扫描完数据集中的所有项以及所有候选集时，就需要计算支持度。不满足最小支持度要求的集合不会输出。函数也会先构建一个空列表，该列表包含满足最小支持度要求的集合。下一个循环遍历字典中的每个元素并且计算支持度❷。如果支持度满足最小支持度要求，则将字典元素添加到retList中。可以使用语句retList.insert(0,key)在列表的首部插入任意新的集合。当然也不一定非要在首部插入，这只是为了让列表看起来有组织。函数最后返回最频繁项集的支持度supportData，该值会在下一节中使用。
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        # print "tid:",tid
        for can in Ck:
            # print "can:",can
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1
                # print "ssCnt[",can,"]:",ssCnt[can]
    numItems = float(len(D))
    retList = []
    supportData = {}
    # print "numItems:",numItems
    for key in ssCnt:
        support = ssCnt[key]/numItems
        # print "support:",support
        if support >= minSupport:
            retList.insert(0,key)
            # print "retList:",retList
        supportData[key] = support
        # print "supportData[",key,"]:",supportData[key]
    return retList, supportData

# dataSet=loadDataSet()
# C1=createC1(dataSet)
# D= map(set,dataSet)
# L1,suppData0=scanD(D,C1,0.5)
# print "dataSet:",dataSet
# print "C1:",C1
# print "D:",D
# print "L1:",L1
# print "suppData0:",suppData0
# print ":",



#程序清单11-2　Apriori算法
#函数aprioriGen()的输入参数为频繁项集列表Lk与项集元素个数k，输出为Ck。举例来说，该函数以{0}、{1}、{2}作为输入，会生成{0,1}、{0,2}以及{1,2}。要完成这一点，首先创建一个空列表，然后计算Lk中的元素数目。接下来，比较Lk中的每一个元素与其他元素，这可以通过两个for循环来实现。紧接着，取列表中的两个集合进行比较。如果这两个集合的前面k-2个元素都相等，那么就将这两个集合合成一个大小为k的集合❶。这里使用集合的并操作来完成，在Python中对应操作符|。
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    # print "lenLk:",lenLk
    for i in range(lenLk):
        # print "i:",i
        for j in range(i+1, lenLk):
            # print "j:",j
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            # print "list(Lk[i]):",list(Lk[i])
            # print "list(Lk[j]):",list(Lk[j])
            # print "L1:",L1
            # print "L2:",L2
            L1.sort(); L2.sort()
            # print "L1.sort():",L1.sort()
            # print "L2.sort():",L2.sort()
            if L1==L2: #if first k-2 elements are equal
                # print "Lk[i]:",Lk[i]
                # print "Lk[j]:",Lk[j]
                retList.append(Lk[i] | Lk[j]) #set union
                # print "retList:",retList
    return retList

#其中主函数是apriori()，它会调用aprioriGen()来创建候选项集Ck。
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    # print "C1:",C1
    # print "D:",D
    # print "L1:",L1
    # print "supportData:",supportData
    # print "L:",L
    # print ":",

    while (len(L[k-2]) > 0):
        # print "L[",k-2,"]:", L[k-2]
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        # print "Ck:",Ck
        # print "Lk:",Lk
        # print "supK:",supK
        # print ":",
        supportData.update(supK)
        L.append(Lk)
        k += 1
        # print "supportData:",supportData
        # print "L:",L
        # print "k:",k
        # print "Len(L[k-2]) > 0:",len(L[k-2]) > 0

    return L, supportData

# dataSet=loadDataSet()
# L, suppData=apriori(dataSet)
# # print "dataSet:",dataSet
# # print "L:",L
# # print "suppData:",suppData
# print "aprioriGen(L[0],2):",aprioriGen(L[0],2)
# L,suppData=apriori(dataSet,minSupport=0.7)
# # print "L:",L
# # print "suppData:",suppData


#程序清单11-3　关联规则生成函数
#第一个函数generateRules()是主函数，它调用其他两个函数。其他两个函数是rulesFromConseq()和calcConf()，分别用于生成候选规则集合以及对规则进行评估。
# 函数generateRules()有3个参数：频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值。函数最后要生成一个包含可信度的规则列表，后面可以基于可信度对它们进行排序。这些规则存放在bigRuleList中。如果事先没有给定最小可信度的阈值，那么默认值设为0.7。
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    # print "len(L):", len(L)
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        # print "i:",i
        # print "L[",i,"]:", L[i]
        for freqSet in L[i]:
            # print "freqSet:",freqSet
            H1 = [frozenset([item]) for item in freqSet]
            # print "H1:",H1
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
            # print "bigRuleList:",bigRuleList
    return bigRuleList         

#我们的目标是计算规则的可信度以及找到满足最小可信度要求的规则。所有这些可以使用函数calcConf()来完成，而程序清单11-3中的其余代码都用来准备规则。
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        # print "freqSet:",freqSet
        # print "conseq:",conseq
        # print "freqSet-conseq:",freqSet-conseq
        # print ":",
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        # print "conf:",conf
        if conf >= minConf: 
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
            # print "prunedH:", prunedH
    return prunedH

#为从最初的项集中生成更多的关联规则，可以使用rulesFromConseq()函数。该函数有2个参数：一个是频繁项集，另一个是可以出现在规则右部的元素列表H。
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    # print "freqSet:", freqSet
    # print "H[0]:",H[0]
    m = len(H[0])
    # print "m:",m
    # print "len(freqSet):",len(freqSet)
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        # print "Hmp1:",Hmp1
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # print "Hmp1:",Hmp1
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# dataSet=loadDataSet()
# L, suppData=apriori(dataSet,minSupport= 0.5)
# rules=generateRules(L,suppData,minConf= 0.5)
# print "dataSet:",dataSet
# print "L:",L
# print "suppData:",suppData
# # print "rules:",rules
# print ":",

def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print itemMeaning[item]
        print "           -------->"
        for item in ruleTup[1]:
            print itemMeaning[item]
        print "confidence: %f" % ruleTup[2]
        print       #print a blank line
        
            
from time import sleep
# from votesmart import votesmart
# votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
# #votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open(homedir+'recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList

#程序清单11-5基于投票数据的事务列表填充函数
#函数getTransList()会创建一个事务数据库，于是在此基础上可以使用前面的Apriori代码来生成频繁项集与关联规则。该函数也会创建一个标题列表，所以很容易了解每个元素项的含义。
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning


# actionIdList, billTitles =getActionIds()
# print "actionIdList:",actionIdList
# print "billTitles:",billTitles

# mushDatSet= [line.split() for line in open(homedir+'mushroom.dat').readlines()]
# L, suppData=apriori(mushDatSet,minSupport= 0.3)
# print "mushDatSet:",mushDatSet
# print "L:",L
# print "suppData:",suppData
# for item in L[1]:
#     if item.intersection('2'): print item
# for item in L[3]:
#     if item.intersection('2'): print item
