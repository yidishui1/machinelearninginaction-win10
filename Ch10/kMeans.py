#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *
import os


# homedir= os.getcwd()+'/machinelearninginaction/ch10/'  #绝对路径
homedir= '' #相对路径

#程序清单10-1k均值聚类支持函数
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

#distEclud()计算两个向量的欧式距离。
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

#最后一个函数是randCent()，该函数为给定数据集构建一个包含k个随机质心的集合。随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。然后生成0到1.0之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

#上述清单给出了k均值算法。kMeans()函数接受4个输入参数。只有数据集及簇的数目是必选参数，而用来计算距离和创建初始质心的函数都是可选的。kMeans()函数一开始确定数据集中数据点的总数，然后创建一个矩阵来存储每个点的簇分配结果。簇分配结果矩阵clusterAssment包含两列：一列记录簇索引值，第二列存储误差。这里的误差是指当前点到簇质心的距离，后边会使用该误差来评价聚类的效果。按照上述方式（即计算质心-分配-重新计算）反复迭代，直到所有数据点的簇分配结果不再改变为止。程序中可以创建一个标志变量clusterChanged，如果该值为True，则继续迭代。上述迭代使用while循环来实现。接下来遍历所有数据找到距离每个点最近的质心，这可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成❶。计算距离是使用distMeas参数给出的距离函数，默认距离函数是distEclud()，该函数的实现已经在程序清单10-1中给出。如果任一点的簇分配结果发生改变，则更新clusterChanged标志。
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids=createCent(dataSet,k)
    # print "m:",m
    # print "clusterAssment:",clusterAssment
    # print "centroids:",centroids,"centroids:"
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        # print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            # print "clusterAssment[:,0]:",clusterAssment[:,0]
            # print "clusterAssment[:,0].A:",clusterAssment[:,0].A
            # print "clusterAssment[:,0].A==cent:",clusterAssment[:,0].A==cent
            # print "nonzero(clusterAssment[:,0].A==cent):",nonzero(clusterAssment[:,0].A==cent)
            # print "ptsInClust:",ptsInClust
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment

# datMat= mat(loadDataSet(homedir+'testSet.txt'))
# myCentroids,clustAssing=kMeans(datMat,4)

# print "datMat:",datMat
# print "min(datMat[:,0]):",min(datMat[:,0])
# print "min(datMat[:,1]):",min(datMat[:,1])
# print "max(datMat[:,0]):",max(datMat[:,0])
# print "max(datMat[:,1]):",max(datMat[:,1])
# print "randCent(datMat,2):",randCent(datMat,2)
# print "distEclud( datMat[ 0], datMat[ 1]):",distEclud( datMat[ 0], datMat[ 1])
# print "myCentroids:",myCentroids
# print "clustAssing:",clustAssing
# print ":",
# print ":",

#程序清单10-3　二分K均值聚类算法
#上述程序中的函数与程序清单10-2中函数kMeans()的参数相同。在给定数据集、所期望的簇数目和距离计算方法的条件下，函数返回聚类结果。同kMeans()一样，用户可以改变所使用的距离计算方法。
# 该函数首先创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差，然后计算整个数据集的质心，并使用一个列表来保留所有的质心❶。得到上述质心之后，可以遍历数据集中所有点来计算每个点到质心的误差值。这些误差值将会在后面用到。
# 接下来程序进入while循环，该循环会不停对簇进行划分，直到得到想要的簇数目为止。可以通过考察簇列表中的值来获得当前簇的数目。然后遍历所有的簇来决定最佳的簇进行划分。为此需要比较划分前后的SSE。一开始将最小SSE置设为无穷大，然后遍历簇列表centList中的每一个簇。对每个簇，将该簇中的所有点看成一个小的数据集ptsInCurrCluster。将ptsInCurrCluster输入到函数kMeans()中进行处理（Ｋ=2）。k均值算法会生成两个质心（簇），同时给出每个簇的误差值❷。这些误差与剩余数据集的误差之和作为本次划分的误差。如果该划分的SSE值最小，则本次划分被保存。一旦决定了要划分的簇，接下来就要实际执行划分操作。划分操作很容易，只需要将要划分的簇中所有点的簇分配结果进行修改即可。当使用kMeans()函数并且指定簇数为2时，会得到两个编号分别为0和1的结果簇。需要将这些簇编号修改为划分簇及新加簇的编号，该过程可以通过两个数组过滤器来完成❸。最后，新的簇分配结果被更新，新的质心会被添加到centList中。当while循环结束时，同kMeans()函数一样，函数返回质心列表与簇分配结果。
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0= mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    # print "m:",m
    # print "clusterAssment:",clusterAssment
    # print "centroid0:",centroid0
    # print "centList:",centList
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    # print "clusterAssment:",clusterAssment
    while (len(centList) < k):
        # print "len(centList):",len(centList)
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # print "i:",i
            # print "ptsInCurrCluster:",ptsInCurrCluster
            # print "centroidMat:",centroidMat
            # print "splitClustAss:",splitClustAss
            # print ":",
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            # print "(splitClustAss[:,1]):",(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            # print "clusterAssment[:,0]:",clusterAssment[:,0]
            # print "clusterAssment[:,0].A:",clusterAssment[:,0].A
            # print "clusterAssment[:,0].A!=i:",clusterAssment[:,0].A!=i
            # print "nonzero(clusterAssment[:,0].A!=i):",nonzero(clusterAssment[:,0].A!=i)
            # print "nonzero(clusterAssment[:,0].A!=i)[0]:",nonzero(clusterAssment[:,0].A!=i)[0]
            # print "clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]:",clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]
            # print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
                # print "bestCentToSplit :",bestCentToSplit
                # print "bestNewCents:",bestNewCents
                # print "bestClustAss:",bestClustAss
                # print "lowestSSE:",lowestSSE
        # print "bestCentToSplit :",bestCentToSplit
        # print "bestNewCents:",bestNewCents
        # print "bestClustAss:",bestClustAss
        # print "lowestSSE:",lowestSSE
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        # print "bestClustAss:", bestClustAss
        # print "bestClustAss:", bestClustAss
        # print 'the bestCentToSplit is: ',bestCentToSplit
        # print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        # print "centList[bestCentToSplit]:", centList[bestCentToSplit]
        centList.append(bestNewCents[1,:].tolist()[0])
        # print "centList:", centList
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
        # print "clusterAssment[:,0]:",clusterAssment[:,0]
        # print "clusterAssment[:,0].A:",clusterAssment[:,0].A
        # print "clusterAssment[:,0].A== bestCentToSplit:",clusterAssment[:,0].A== bestCentToSplit
        # print "nonzero(clusterAssment[:,0].A == bestCentToSplit)[0]:",nonzero(clusterAssment[:,0].A == bestCentToSplit)[0]
        # print "clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]",clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]

    return mat(centList), clusterAssment

# datMat3=mat(loadDataSet(homedir+'testSet2.txt'))
# centList,myNewAssments=biKmeans(datMat3,3)
# print "datMat3:",datMat3
# print "centList:",centList
# print "myNewAssments:",myNewAssments

#程序清单10-4　Yahoo!PlaceFinderAPI
import urllib
import json
#在函数geoGrab()中，首先为YahooAPI设置apiStem，然后创建一个字典。你可以为字典设置不同值，包括flags=J，以便返回JSON格式的结果❶。（不用担心你不熟悉JSON，它是一种用于序列化数组和字典的文件格式，本书不会看到任何JSON。JSON是JavaScriptObjectNotation的缩写，有兴趣的读者可以在www.json.org找到更多信息。）接下来使用urllib的urlencode()函数将创建的字典转换为可以通过URL进行传递的字符串格式。最后，打开URL读取返回值。由于返回值是JSON格式的，所以可以使用JSON的Python模块来将其解码为一个字典。一旦返回了解码后的字典，也就意味着你成功地对一个地址进行了地理编码。
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print yahooApi
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
#程序清单10-4中的第二个函数是massPlaceFind()。该函数打开一个tab分隔的文本文件，获取第2列和第3列结果。这些值被输入到函数geoGrab()中，然后需要检查geoGrab()的输出字典判断有没有错误。如果没有错误，就可以从字典中读取经纬度。这些值被添加到原来对应的行上，同时写到一个新的文件中。如果有错误，就不需要去抽取纬度和经度。最后，调用sleep()函数将massPlaceFind()函数延迟1秒。这样做是为了确保不要在短时间内过于频繁地调用API。如果频繁调用，那么你的请求可能会被封掉，所以将massPlaceFind()函数的调用延迟一下比较好。
def massPlaceFind(fileName):
    fw = open(homedir+'places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print "error fetching"
        sleep(1)
    fw.close()

# geoResults=geoGrab('1 VA Center', 'Augusta, ME')
# print "geoResults:",geoResults
# print "geoResults['ResultSet']['Error']:",geoResults['ResultSet']['Error']
# print "geoResults['ResultSet']['Results'][0]['longitude']:",geoResults['ResultSet']['Results'][0]['longitude']
# print "massPlaceFind('portlandClubs.txt'):",massPlaceFind(homedir+'portlandClubs.txt')

#程序清单10-5　球面距离计算及簇绘图函数
#函数distSLC()返回地球表面两点间的距离，单位是英里。给定两个点的经纬度，可以使用球面余弦定理来计算两点的距离。这里的纬度和经度用角度作为单位，但是sin()以及cos()以弧度为输入。可以将角度除以180然后再乘以圆周率pi转换为弧度。导入NumPy的时候就会导入pi。
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
#第二个函数clusterClubs()只有一个参数，即所希望得到的簇数目。该函数将文本文件的解析、聚类以及画图都封装在一起，首先创建一个空列表，然后打开places.txt文件获取第4列和第5列，这两列分别对应纬度和经度。基于这些经纬度对的列表创建一个矩阵。接下来在这些数据点上运行biKmeans()并使用distSLC()函数作为聚类中使用的距离计算方法。最后将簇以及簇质心画在图上。
def clusterClubs(numClust=5):
    datList = []
    for line in open(homedir+'places.txt').readlines():
        lineArr = line.split('\t')
        # print "lineArr:",lineArr
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    # print "datMat:",datMat
    # print "myCentroids:",myCentroids
    # print "clustAssing:",clustAssing
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    # print ":",
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread(homedir+'Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        # print "i:",i
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        # print "ptsInCurrCluster:",ptsInCurrCluster
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        # print "markerStyle:", markerStyle
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
        # print "ptsInCurrCluster[:,0]:",ptsInCurrCluster[:,0]
        # print "ptsInCurrCluster[:,0].flatten():",ptsInCurrCluster[:,0].flatten()
        # print "ptsInCurrCluster[:,0].flatten().A:",ptsInCurrCluster[:,0].flatten().A
        # print "ptsInCurrCluster[:,0].flatten().A[0]:",ptsInCurrCluster[:,0].flatten().A[0]
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    # print "myCentroids[:,0].flatten().A[0]:",myCentroids[:,0].flatten().A[0]
    plt.show()


# print "clusterClubs(5):",clusterClubs(5)