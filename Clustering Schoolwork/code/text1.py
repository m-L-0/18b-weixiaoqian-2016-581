# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:56:19 2018

@author: 18344
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
##鸢尾花数据集下载

#def ReadAndSave(target_url = None,save = False):
#    datas = pd.read_csv(target_url,header=0,sep=",")
#    if save == True:
#        datas.to_csv("irisdata.csv",index = False)
#    return datas
#
#target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#ReadAndSave(target_url,True)
  



'''数据集读取'''

iris_data = pd.read_csv('irisdata.csv',header = None,names=['a','b','c','d','feature'])
iris_data = pd.DataFrame(iris_data)

#2类型编码
feature_mapping = {'Iris-setosa': 0,'Iris-versicolor':1,'Iris-virginica':2}
iris_data['feature'] = iris_data['feature'].map(feature_mapping)


X = iris_data[iris_data.columns[:-1]]
X = np.array(X)
y = iris_data["feature"]
y = np.array(y)
#print(y)
#df_y.to_csv('y.csv')



'''散点图绘制,获取其中两列数据或特征'''

DX = [x[0] for x in X]
#print(DX)
DY = [x[1] for x in X]
        
plt.scatter(DX[:50],DY[:50],color = 'red',marker='+',label='setosa')#前50个
plt.scatter(DX[50:100],DY[50:100],color='purple',marker='*',label='versicolor')#中间50个
plt.scatter(DX[100:],DY[100:],color='green',marker='.',label='Virginica')#后50个
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend(loc=1)
plt.show()


#后两列特征数据的散点图描绘
#DC = [x[2] for x in X]
#DD = [x[3] for x in X]
#        
#plt.scatter(DC[:50],DD[:50],color = 'red',marker='+',label='setosa')#前50个
#plt.scatter(DC[50:100],DD[50:100],color='purple',marker='*',label='versicolor')#中间50个
#plt.scatter(DC[100:],DD[100:],color='green',marker='.',label='Virginica')#后50个
#plt.legend(loc=2)
#plt.show()




'''KMeans聚类'''



est=KMeans(n_clusters=3)
est.fit(X)
kc=est.cluster_centers_
y_kmeans=est.predict(X)

#print(y_kmeans)
#print(kc)
#print(kc.shape,y_kmeans.shape,X.shape)
plt.scatter(X[:,0],X[:,1],c=y_kmeans,s=50,marker='.',cmap='rainbow');
plt.show()

G = nx.Graph()
Y_K = np.array(y_kmeans)
#重新分配长度内存
Y_K.resize(50,3)
print(Y_K)



#连线

#for i in range(len(Y_K)):
#    for j in range(len(Y_K)):
#        G.add_edge(i,j)
#
#nx.draw(G)
#plt.show()






