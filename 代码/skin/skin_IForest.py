from __future__ import division
from __future__ import print_function

import os
import sys
import sklearn

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from scipy.io import loadmat


from pyod.models.iforest import IForest
from pyod.models.combination import aom, moa, average, maximization, median
from pyod.utils.utility import standardizer
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

#获取数据集文件所在的目录
path='/Users/mac/Desktop/课程/数据挖掘/skin/benchmarks'

#对目录下的所有csv文件进行排序
files=os.listdir(path)
files.sort(key= lambda x:int(x[15:-4]))

#初始化两个变量分别保存训练集和测试集中每个csv数据集得到的AUC值的和
sumAuc_train=0
sumAuc_test=0
i=0
for info in files:
    #得到文件的路径
    info = os.path.join(path,info)    
    data = pd.read_csv(info) 
    
    #取相关属性
    cols=['diff.score','R','G','B']
    x = data[cols].values
   
    #把label标签加入，把该问题当成有监督问题来处理
    data['s']=data['original.label']
    data.loc[data['original.label']!=1,'s']=0
    y=data['s']
    #print(data['s'])
    
    #划分测试集和训练集
    X_train,X_test,y_train,y_test= train_test_split(x,y, test_size=0.33)

    #使用pyod中的IForest算法拟合数据
    clf_name = 'IForest'
    clf = IForest()
    clf.fit(X_train)
   
    #预测得到由0和1组成的数组，1表示离群点，0表示飞离群点
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores，The outlier scores of the training data.
    
    
    #预测样本是不是离群点，返回0和1 的数组
    y_test_pred = clf.predict(X_test)
   
    y_test_scores = clf.decision_function(X_test)  # outlier scores，The anomaly score of the input samples.
    #使用sklearn中的roc_auc_score方法得到auc值，即roc曲线下面的面积
    try:
        sumAuc_train+=sklearn.metrics.roc_auc_score(y_train,y_train_scores, average='macro')
        sumAuc_test+=sklearn.metrics.roc_auc_score(y_test,y_test_scores, average='macro')
        #s=precision_score(y_train, y_train_scores, average='macro')
        i+=1
        print(sumAuc_train,sumAuc_test)
    except ValueError:
        pass

   
    #得到ROC值和精确度 prn
    evaluate_print(clf_name, y_train, y_train_scores)
    #evaluate_print(clf_name, y_test, y_test_scores)
print("\nOn Training Data:")
print(sumAuc_train/i)#
print("\nOn Test Data:")
print(sumAuc_test/i)#