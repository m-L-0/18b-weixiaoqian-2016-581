# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 08:22:09 2018
9个类别数据集参见压缩包 
每个样本由200个波段组成(即200个光谱特征) 
每个数据文件对应一个类别 
测试集周二晚提供 ,测试集：2310个样本
光谱之间可能存在相关性 
最后决策结果的类别标号来自【2,3,5,6,8,10,11,12,14】
下周二下午截止。 （12月11号）
@author: weixiaoqian
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#2,3,5,6,8,10,11,12,14

'''加载数据'''
str0 = '..//9data-train/'
str1 = 'data'
str2 = '_train.mat'
for i in [2,3,5,6,8,10,11,12,14]:
    name = str0 + str1 + str(i) + str2
#    print('data',i)
    Data = sio.loadmat(name)
    col = str1 + str(i) + '_train'
    x = Data[col]
    #print('shape:',x.shape)
    #将数据导出为csv格式并保存
    dfdata = pd.DataFrame(x)
    datapath ='D://大三//Training-2018b-2016//18b-weixiaoqian-2016-581//9data-train//x//' + str(i) +'.csv'
    dfdata.to_csv(datapath,index = False)
    y = [i]* int(x.shape[0])
    Ydatapath ='D://大三//Training-2018b-2016//18b-weixiaoqian-2016-581//9data-train//y//' + str(i) +'.csv'
    dfY = pd.DataFrame(y)
    dfY.to_csv(Ydatapath,index = False)
    
'''数组读取'''
data2 = pd.read_csv('..//9data-train//x//2.csv')
data3 = pd.read_csv('..//9data-train//x//3.csv')
data5 = pd.read_csv('..//9data-train//x//5.csv')
data6 = pd.read_csv('..//9data-train//x//6.csv')
data8 = pd.read_csv('..//9data-train//x//8.csv')
data10 = pd.read_csv('..//9data-train//x//10.csv')
data11 = pd.read_csv('..//9data-train//x//11.csv')
data12 = pd.read_csv('..//9data-train//x//12.csv')
data14 = pd.read_csv('..//9data-train//x//14.csv')

label2 = pd.read_csv('..//9data-train//y//2.csv')
label3 = pd.read_csv('..//9data-train//y//3.csv')
label5 = pd.read_csv('..//9data-train//y//5.csv')
label6 = pd.read_csv('..//9data-train//y//6.csv')
label8 = pd.read_csv('..//9data-train//y//8.csv')
label10 = pd.read_csv('..//9data-train//y//10.csv')
label11 = pd.read_csv('..//9data-train//y//11.csv')
label12 = pd.read_csv('..//9data-train//y//12.csv')
label14 = pd.read_csv('..//9data-train//y//14.csv')
'''测试集'''
TestDataflie = '..//9data-train//data_test_final.mat'
TestData = sio.loadmat(TestDataflie)
testdata = pd.DataFrame(TestData['data_test_final'])
testdata.to_csv('..//9data-train//xtest.csv',index = False)

xtest = pd.read_csv('..//9data-train//xtest.csv')

'''数组合并'''

xx = np.vstack((data2,data3,data5,data6,data8,data10,data11,data12,data14))
#print(xData.shape)     #(6924, 200)
yy = np.vstack((label2,label3,label5,label6,label8,label10,label11,label12,label14))

'''数据预处理（降维）'''
#
#def pca(data,topNfeat = 99999):
#    meanval = np.mean(data,axis = 0)#计算每行的平均值axis = 1
#    #去平均值，可以直接对两个维度不同的矩阵进行运算
#    meanrem = data - meanval
#    #计算协方差矩阵
#    covdata = np.cov(meanrem,rowvar = 0)
#    #计算协方差矩阵的特征值和特征向量
#    eigenval,eigenvector = np.linalg.eig(np.mat(covdata))
#    #对特征值按升序排序
#    eigenvalsInd = np.argsort(eigenval)
#    #对特征值进行逆序排序
#    eigenvalsInd = eigenvalsInd[:-(topNfeat+1):-1]
#    #计算最大特征值对应的特征向量
#    redEigVec = eigenvector[:,eigenvalsInd]
#    #计算降维之后的数据集
#    Lowdata = meanrem * redEigVec
#    #重构原始数据
#    recondata = (Lowdata*redEigVec.T) + meanval
#    return Lowdata,recondata
#
#
#lowdata,recondata = pca(xx,1)
#print(recondata)

#随机划分数据集
X_train,X_test,y_train,y_test = train_test_split(xx,yy,test_size = 0.2,random_state = 0)


#直方图
#plt.hist(xx,bins = 50)
#plt.show()

#KNN分类模型对生成的样本集进行类别预测，返回与xx,yy格式一致的预测结果

'''KNN分类及预测结果的可视化'''
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuary = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        accuary +=1
acc = accuary/len(y_test)
print(acc)
y_test = pd.DataFrame(y_test)
y_test.to_csv('D://大三//Training-2018b-2016//18b-weixiaoqian-2016-581//9data-train//y.csv')



#'''决策树'''
#from sklearn.tree import DecisionTreeClassifier
#dtc = DecisionTreeClassifier()
#dtc.fit(X_train,y_train)
#y_predict = dtc.predict(X_test)
#
#'''模型评估'''
#from sklearn.metrics import classification_report
#
#print(dtc.score(X_test,y_test))
#print(classification_report(y_predict,y_test,target_names=['buy','no']))
#
#print(sum(y_predict==y_test)/len(y_test))
#
#dtc.predict(X_test)
