# -*- coding: utf-8 -*-
"""
Created on Mon May 21 00:10:37 2018

@author: kimhongji
"""
#default library
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

#in[2]
#분류 알고리즘 데이터 예제
x,y = mglearn.datasets.make_forge()
#산점도를 그립니다
mglearn.discrete_scatter(x[:,0], x[:,1],y)
plt.legend(["클래스 0","클래스 1"], loc=4)
plt.xlabel("첫번재 특성")
plt.ylabel("두번째 특성")
print("x.shape: {}".format(x.shape))

#in[3]
#회귀 알고리즘 데이터 예제
x,y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(x,y,'o')
plt.ylim(-3,3)
plt.xlabel("특성")
plt.ylabel("타깃")

#in[5]
#sklearn 데이터 불러오기 ( cancer 예제 )
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

#in[6]
print("유방암 데이터의 형태: {}".format(cancer.data.shape))
#in[7]
print("클래스별 샘플 개수:\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
#benign : 양성, malignant: 악성 

#in[8]
print("특성 이름:\n{}".format(cancer.feature_names))

#in[9] 
#회귀분석 용으로는 (boston housing) 보스턴 주택 가격
from sklearn.datasets import load_boston
boston = load_boston()
print("데이터의 형태: {}".format(boston.data.shape))

#in[10]
#특성공학 이용
x, y = mglearn.datasets.load_extended_boston()
print("x.shape: {}".format(x.shape))

#in[11]
#knn 적용 #n_neighbors = 1,3,5 등으로 확인 가능 
mglearn.plots.plot_knn_classification(n_neighbors = 1)

#in[13]
#sklearn 으로 k-nn 알고리즘 적용
from sklearn.model_selection import train_test_split
x,y = mglearn.datasets.make_forge()

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 0 )

#in[14]
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 3)

#in[15]
#학습
clf.fit(x_train, y_train)

#in[16]
print("테스트 세트 예측: {}".format(clf.predict(x_test)))

#in[17]
print("테스트 세트 정확도 : {:.2f}".format(clf.score(x_test,y_test)))