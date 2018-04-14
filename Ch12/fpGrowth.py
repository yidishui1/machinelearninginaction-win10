#!/usr/bin/python
# -*- coding:utf-8 -*-

'''
Created on Jun 14, 2011
FP-Growth FP means frequent pattern
the FP-Growth algorithm needs: 
1. FP-tree (class treeNode)
2. header table (use dict)

This finds frequent itemsets similar to apriori but does not 
find association rules.  

@author: Peter
'''
import os

# homedir=os.getcwd()+'/machinelearninginaction/ch12/'
homedir=''

#12.2.1　创建FP树的数据结构
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name =nameValue
        self.count =numOccur
        self.nodeLink=None
        self.parent=parentNode      #needs to be updated
        self.children={}
        # print "self.name:",self.name
        # print "self.count:",self.count
        # print "self.nodeLink:",self.nodeLink
        # print "self.parent:",self.parent
        # print "self.children:",self.children
        # print "-----",""

    #其中inc()对count变量增加给定值
    def inc(self, numOccur):
        self.count += numOccur
        # print "self.count:",self.count

    #disp()用于将树以文本形式显示。后者对于树构建来说并不是必要的，但是它对于调试非常有用。
    def disp(self, ind=1):
        print '   '*ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)


# rootNode=treeNode('pyramid',9,None)
# rootNode.children['eye']=treeNode('eye',13,None)
# rootNode.disp()
# rootNode.children['phoenix']=treeNode('phoenix',3,None)
# rootNode.disp()

#程序清单12-2　FP树构建函数
#第一个函数createTree()使用数据集以及最小支持度作为参数来构建FP树。树构建过程中会遍历数据集两次。第一次遍历扫描数据集并统计每个元素项出现的频度。这些信息被存储在头指针表中。接下来，扫描头指针表删掉那些出现次数少于minSup的项❶。如果所有项都不频繁，就不需要进行下一步处理❷。接下来，对头指针表稍加扩展以便可以保存计数值及指向每种类型第一个元素项的指针。然后创建只包含空集合Ø的根节点。最后，再一次遍历数据集，这次只考虑那些频繁项❸。这些项已经如表12-2所示那样进行了排序，然后调用updateTree()方法❹。
def createTree(dataSet, minSup=1): #create FP-tree from dataset but don't mine
    headerTable = {}
    #go over dataSet twice
    for trans in dataSet:#first pass counts frequency of occurance 计算所有子元素的频率
        # print "trans:",trans
        for item in trans:
            # print "item:",item
            # print "headerTable.get(item, 0):",headerTable.get(item, 0)
            # print "dataSet[trans]:",dataSet[trans]
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
            # print "headerTable[item]:", headerTable[item]
            # print ":",
    # print "headerTable:",headerTable
    for k in headerTable.keys():  #remove items not meeting minSup移除所有低于最小频率的元素
        if headerTable[k] < minSup: 
            del(headerTable[k])
    # print "headerTable:", headerTable
    freqItemSet = set(headerTable.keys())
    # print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0: return None, None  #if no items meet min support -->get out
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] #reformat headerTable to use Node link 
    # print 'headerTable: ',headerTable
    retTree = treeNode('Null Set', 1, None) #create tree
    # print "retTree:",retTree
    for tranSet, count in dataSet.items():  #go through dataset 2nd time
        localD = {}
        # print "tranSet:",tranSet
        for item in tranSet:  #put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        # print "localD:",localD
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # print "orderedItems:",orderedItems
            updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset
    return retTree, headerTable #return tree and header table

#updateTree()中的执行细节。该函数首先测试事务中的第一个元素项是否作为子节点存在。如果存在的话，则更新该元素项的计数；如果不存在，则创建一个新的treeNode并将其作为一个子节点添加到树中。这时，头指针表也要更新以指向新的节点。更新头指针表需要调用函数updateHeader()，接下来会讨论该函数的细节。updateTree()完成的最后一件事是不断迭代调用自身，每次调用时会去掉列表中第一个元素❺。
def updateTree(items, inTree, headerTable, count):
    # print "items:",items
    # print "inTree:",inTree
    # print "inTree.children:", inTree.children
    # print "headerTable:",headerTable
    # print "count:",count
    if items[0] in inTree.children:#check if orderedItems[0] in retTree.children
        # print "1:",""
        inTree.children[items[0]].inc(count) #incrament count
        # inTree.disp()
    else:   #add items[0] to inTree.children
        # print "2:", ""
        # print "items[0]:",items[0]
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # inTree.disp()
        # print "headerTable[items[0]][1]:", headerTable[items[0]][1]
        if headerTable[items[0]][1] == None: #update header table 
            headerTable[items[0]][1] = inTree.children[items[0]]
            # print "inTree.children[items[0]]:", inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:#call updateTree() with remaining ordered items
        # print "3:", ""
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

#updateHeader()，它确保节点链接指向树中该元素项的每一个实例。从头指针表的nodeLink开始，一直沿着nodeLink直到到达链表末尾。这就是一个链表。当处理树的时候，一种很自然的反应就是迭代完成每一件事。当以相同方式处理链表时可能会遇到一些问题，原因是如果链表很长可能会遇到迭代调用的次数限制。
def updateHeader(nodeToTest, targetNode):   #this version does not use recursion
    while (nodeToTest.nodeLink != None):    #Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


#程序清单12-3　简单数据集及数据包装器
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    # print "retDict:", retDict
    return retDict

#程序清单12-4　发现以给定元素项结尾的所有路径的函数
#每遇到一个元素项都会调用ascendTree()来上溯FP树，并收集所有遇到的元素项的名称❶。该列表返回之后添加到条件模式基字典condPats中。
def ascendTree(leafNode, prefixPath):  # ascends from leaf node to root
    # print "leafNode:",leafNode
    # print "prefixPath:",prefixPath
    # print "leafNode.parent:",leafNode.parent
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        # print "prefixPath:",prefixPath
        ascendTree(leafNode.parent, prefixPath)

#函数findPrefixPath()遍历链表直到到达结尾。
def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    # print "basePat:",basePat
    # print "treeNode:",treeNode
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        # print "prefixPath:",prefixPath
        if len(prefixPath) > 1:
            # print "prefixPath[1:]:",prefixPath[1:]
            # print "frozenset(prefixPath[1:]):",frozenset(prefixPath[1:])
            # print "treeNode.count:",treeNode.count
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    #     print "treeNode:", treeNode
    # print "condPats:",condPats
    return condPats

#程序清单12-5　递归查找频繁项集的mineTree函数
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # print "inTree:",inTree
    # print "headerTable:",headerTable
    # print "minSup:",minSup
    # print "preFix:",preFix
    # print "freqItemList:",freqItemList
    # print ":",

    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]  # (sort header table)
    # print "bigL:",bigL
    for basePat in bigL:  # start from bottom of header table
        # print "basePat:",basePat
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet)
        # print "freqItemList:",freqItemList
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # print 'condPattBases :',basePat, condPattBases
        # 2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        # print 'head from conditional tree: ', myHead
        if myHead != None:  # 3. mine cond. FP-tree
            print 'conditional tree for: ',newFreqSet
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


# simpDat=loadSimpDat()
# # print "simpDat:",simpDat
# # print ":",
# initSet=createInitSet(simpDat)
# # print "initSet:",initSet
# myFPtree,myHeaderTab=createTree(initSet,3)
# # print "myFPtree:",myFPtree
# # print "myHeaderTab:",myHeaderTab
# myFPtree.disp()
#
# findPrefixPath('x',myHeaderTab['x'][1])
# findPrefixPath('z',myHeaderTab['z'][1])
# findPrefixPath('r',myHeaderTab['r'][1])
#
# freqItems = []
# mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)



# import twitter
from time import sleep
import re

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print "fetching page %d" % i
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages



def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList

# lotsOtweets =getLotsOfTweets('RIMM')
# lotsOtweets[0][4].text
# listOfTerms =mineTweets(lotsOtweets, 20)
# for t in listOfTerms:
#     print t

#minSup = 3
#simpDat = loadSimpDat()
#initSet = createInitSet(simpDat)
#myFPtree, myHeaderTab = createTree(initSet, minSup)
#myFPtree.disp()
#myFreqList = []
#mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)


# parsedDat=[line.split() for line in open(homedir+'kosarak.dat').readlines()]
# # print "parsedDat:",parsedDat
# initSet=createInitSet(parsedDat)
# # print "initSet:",initSet
# # print ":",
# myFPtree,myHeaderTab=createTree(initSet,100000)
# myFPtree.disp()
# print "myHeaderTab:",myHeaderTab
# myFreqList=[]
# mineTree(myFPtree,myHeaderTab,100000,set([]),myFreqList)
# # print "len(myFreqList):",len(myFreqList)
# # print "myFreqList:",myFreqList
