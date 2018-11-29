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



'''获取其中两列数据或特征，散点图绘制'''

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

#est=KMeans(n_clusters=3)
#est.fit(X)
#kc=est.cluster_centers_
#y_kmeans=est.predict(X)
#
##print(y_kmeans)
##print(kc)
##print(kc.shape,y_kmeans.shape,X.shape)
#plt.scatter(X[:,0],X[:,1],c=y_kmeans,s=50,marker='.',cmap='rainbow');
#plt.show()
#
#G = nx.Graph()
#Y_K = np.array(y_kmeans)



from kmeans import KMEANS


class Spectrum:
    def __init__(self, n_cluster, epsilon=1e-3, maxstep=1000, method='unnormalized',
                 criterion='gaussian', sigma=2.0, dis_epsilon=70, k=5):
        self.n_cluster = n_cluster  # k-means中簇的数量
        self.epsilon = epsilon  # k-means中的参数
        self.maxstep = maxstep  # k-means执行的最大迭代次数
        self.method = method  # method=unnormalized时，选用计算特征向量，method=normalized时，选用归一化的拉普拉斯矩阵计算特征向量
        self.criterion = criterion  # 数据集如果是向量形式，将这些数据转化成图数据时需要用到的方法(比如课上讲的高斯核以及算法中的k近邻法)
        self.sigma = sigma  # 高斯方法中的sigma参数
        self.dis_epsilon = dis_epsilon  # epsilon-近邻方法的参数
        self.k = k  # k近邻方法的参数

        self.W = None  # 图的相似性矩阵
        self.L = None  # 图的拉普拉斯矩阵
        self.L_norm = None  # 归一化后的拉普拉斯矩阵
        self.D = None  # 图的度矩阵
        self.cluster = None  #簇的数量

        self.N = None  # 图中顶点的数量或者数据集中样本的数量

    def init_param(self, data):
        # 初始化参数
        self.N = data.shape[0]  # 获取数据集中有多少数据
        dis_mat = self.cal_dis_mat(data)  # 计算距离平方的矩阵
        self.cal_weight_mat(dis_mat)  # 计算相似度矩阵
        self.D = np.diag(self.W.sum(axis=1))  # 计算度数矩阵
        self.L = self.D - self.W  # 计算拉普拉斯矩阵
        return

    def cal_dis_mat(self, data):
        # 计算距离平方的矩阵
        dis_mat = np.zeros((self.N, self.N))  # 生成一个n×n的零矩阵
        for i in range(self.N):
            for j in range(i + 1, self.N):  # 距离矩阵是对称阵，所以没必要j也从0循环到n-1
                dis_mat[i, j] = (data[i] - data[j]).dot((data[i] - data[j]))  # 计算两个向量之间距离的平方赋值给dis_mat
                dis_mat[j, i] = dis_mat[i, j]  # 距离矩阵是对称阵
        return dis_mat

    def cal_weight_mat(self, dis_mat):
        # 计算相似性矩阵
        if self.criterion == 'gaussian':  # 以高斯核计算样本间的相似度，组成邻接矩阵，由于形成的是全连接的图，所以适合于较小样本集
            if self.sigma is None:  #处理异常
                raise ValueError('sigma is not set')
            self.W = np.exp(-((2*(self.sigma**2))**(-1)) * dis_mat)  # 高斯核计算相似度
        elif self.criterion == 'k_nearest':  # 如果数据集较大，构建全连接图进行谱聚类时间复杂度太高，可以使用k-近邻方法
            if self.k is None or self.sigma is None:
                raise ValueError('k or sigma is not set')
            self.W = np.zeros((self.N, self.N))
            for i in range(self.N):
                inds = np.argpartition(dis_mat[i], self.k + 1)[:self.k + 1]  # 由于包括自身，所以+1
                tmp_w = np.exp(-((2*(self.sigma**2))**(-1)) * dis_mat[i][inds])
                self.W[i][inds] = tmp_w
        elif self.criterion == 'eps_nearest':  # 适合于较大样本集
            if self.dis_epsilon is None:
                raise ValueError('epsilon is not set')
            self.W = np.zeros((self.N, self.N))
            for i in range(self.N):
                inds = np.where(dis_mat[i] < self.dis_epsilon)
                self.W[i][inds] = 1.0 / len(inds)
        else:
            raise ValueError('the criterion is not supported')
        return

    def fit(self, data):
        # 训练主函数
        self.init_param(data)
        if self.method == 'unnormalized':
            w, v = np.linalg.eig(self.L)  # 求L的特征值和特征向量
            inds = np.argsort(w)[:self.n_cluster]  # 特征值排序，取最小的n_cluster个
            Vectors = v[:, inds]  # 将特征向量组成矩阵，每一行可不归一化
        elif self.method == 'normalized':
            D = np.linalg.inv(np.sqrt(self.D))
            D = D**(-0.5)
            L = D.dot(self.L).dot(D)  # 计算归一化的对称拉普拉斯矩阵
            w, v = np.linalg.eig(L)
            inds = np.argsort(w)[:self.n_cluster]
            Vectors = v[:, inds]
            normalizer = np.linalg.norm(Vectors, axis=1)  # 归一化，
            normalizer = np.repeat(np.transpose([normalizer]), self.n_cluster, axis=1)
            Vectors = Vectors / normalizer
        else:
            raise ValueError('the method is not supported')
        km = KMEANS(self.n_cluster,epsilon=1e-3, maxstep=2000)  # 创建k-means实例，当mse小于1e-3或者连续迭代2000次后停止计算
        km.fit(Vectors)  # k-means聚类
        self.cluster = km.cluster  # 分簇结果
        return


if __name__ == '__main__':
    from itertools import cycle  # 循环器
    import matplotlib.pyplot as plt

#    #数据集X
    sp = Spectrum(n_cluster=3, method='unnormalized', criterion='gaussian', sigma=0.6)  # 创建Spectrum实例
    sp.fit(X)
    cluster = sp.cluster
    
    
    
#    km = KMEANS(3)
#    km.fit(X)
#    cluster_km = km.cluster
    
    def visualize(data, cluster):
        # 可视化
        color = 'bgrym'
        for col, inds in zip(cycle(color), cluster.values()):
            partial_data = data[inds]
            plt.scatter(partial_data[:, 0], partial_data[:, 1], color=col,marker='.')
        plt.show()
        return

    visualize(X, cluster)

    def cal_err(data, cluster):
        # 计算MSE(平均平方误差)
        mse = 0
        for label, inds in cluster.items():
            partial_data = data[inds]
            center = partial_data.mean(axis=0)
            for p in partial_data:
                mse += (center - p) @ (center - p)
        return mse / data.shape[0]

    print(cal_err(X, cluster))
#    print(cal_err(X, cluster_km))


'''连线'''
##重新分配长度内存
#Y_K.resize(50,3)
#print(Y_K)

#for i in range(len(Y_K)):
#    for j in range(len(Y_K)):
#        G.add_edge(i,j)
#nx.draw(G)
#plt.show()







