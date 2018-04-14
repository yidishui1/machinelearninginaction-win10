#!/usr/bin/python
# -*- coding:utf-8 -*-
import logRegres
from numpy import*

#5.2.2 训练算法：使用梯度上升找到最佳参数
dataArr,labelMat=logRegres.loadDataSet()
print "回归系数:", logRegres.gradAscent(dataArr,labelMat)

#5.2.3 分析数据：画出决策边界
weights = logRegres.gradAscent(dataArr,labelMat)
logRegres. plotBestFit(weights.getA())

#5.2.4 训练算法：随机梯度上升
dataArr,labelMat=logRegres.loadDataSet()
weights=logRegres.stocGradAscent0(array(dataArr),labelMat)
logRegres.plotBestFit(weights)

weights= logRegres.stocGradAscent1(array(dataArr),labelMat)
logRegres.plotBestFit(weights)

#5.3.2 测试算法：用Logistic回归进行分类
logRegres.multiTest()