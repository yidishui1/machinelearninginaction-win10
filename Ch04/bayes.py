#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *
import feedparser
import os

# homedir= os.getcwd()+'/machinelearninginaction/ch04/'  #绝对路径
homedir= '' #相对路径

#词表到向量的转换函数
#创建了一些实验样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

#创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

#该函数的输入参数为词汇表及某个文档，输出的示文档向量，向量的每一个元素为1或0，分别表示词汇表中单词在输入文档中是否出现。
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones()
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    # print "numTrainDocs:",numTrainDocs
    # print "numWords:",numWords
    # print "pAbusive:",pAbusive
    # print "p0Num:",p0Num
    # print "p1Num:",p1Num
    # print "p0Denom:",p0Denom
    # print "p1Denom:",p1Denom

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        # print "i:",i
        # print "trainCategory[i]:", trainCategory[i]
        # print "trainMatrix[i]:", trainMatrix[i]
        # print "sum(trainMatrix[i]):", sum(trainMatrix[i])
        # print "p0Num:", p0Num
        # print "p1Num:", p1Num
        # print "p0Denom:", p0Denom
        # print "p1Denom:", p1Denom
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

#朴素贝叶斯分类函数，有4个输入：要分类的向量vec2Classify以及使用函数trainNB0()计算得到的三个概率。
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

#朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    # print "listOPosts:", listOPosts
    # print "listClasses:", listClasses
    # print "trainMat:", trainMat
    # print "p0V:", p0V
    # print "p1V:", p1V
    # print "pAb:", pAb
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # print "thisDoc:", thisDoc
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # print "thisDoc:", thisDoc
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

#文件解析及完整的垃圾邮件测试函数
#接受一个大字符串并将其解析为字符串列表
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    # print "listOfTokens:",listOfTokens
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

#对贝叶斯垃圾邮件分类器进行自动化处理
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open(homedir+'email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # print "wordList:", wordList
        # print "docList:", docList
        # print "fullText:", fullText
        # print "classList:", classList
        wordList = textParse(open(homedir+'email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        # print "wordList:", wordList
        # print "docList:", docList
        # print "fullText:", fullText
        # print "classList:", classList
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = range(50); testSet=[]           #create test set
    # print "vocabList:", vocabList
    # print "trainingSet:", trainingSet
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        # print "randIndex:", randIndex
        # print "testSet:", testSet
        del(trainingSet[randIndex])
        # print "trainingSet:", trainingSet
    trainMat=[]; trainClasses = []
    # print "docList:", docList
    # print "trainingSet:", trainingSet
    # print "vocabList:", vocabList
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
        # print "docIndex:", docIndex
        # print "trainMat:", trainMat
        # print "trainClasses:", trainClasses
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    # print "p0V:", p0V
    # print "p1V:", p1V
    # print "pSpam:", pSpam
    errorCount = 0
    print "testSet:", testSet
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            # print "classifyNB(array(wordVector),p0V,p1V,pSpam)",classifyNB(array(wordVector),p0V,p1V,pSpam)
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText

#计算出现频率
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

#使用了两个RSS源作为参数
def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    # print "minLen:",minLen
    for i in range(minLen):
        # print "i:", i
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        # print "wordList:", wordList
        # print "docList:", docList
        # print "fullText:", fullText
        # print "classList:", classList
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        # print "wordList:", wordList
        # print "docList:", docList
        # print "fullText:", fullText
        # print "classList:", classList
    vocabList = createVocabList(docList)#create vocabulary
    # print "vocabList:", vocabList
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    # print "top30Words:", top30Words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    # print "trainingSet:",trainingSet
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    # print "trainingSet:", trainingSet
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # print "trainMat:", trainMat
    # print "trainClasses:", trainClasses
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    # print "p0V:", p0V
    # print "p1V:", p1V
    # print "pSpam:", pSpam
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        # print "wordVector:", wordVector
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is:',float(errorCount)/len(testSet)

    return vocabList,p0V,p1V

#最具表征性的词汇显示函数
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]








