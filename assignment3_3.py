#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:39:33 2019

@author: harry
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from scipy.misc import imread, imresize
from glob import glob
import warnings
warnings.filterwarnings('ignore')

def load_image(image_path, image_size):
  file_name=glob(image_path+"/*pgm")
  sample = []
  for file in file_name:
    pic = imread(file).astype(np.float32)
    pic = imresize(pic, (image_size, image_size)).astype(np.float32)
    sample.append(pic)
  
  sample = np.array(sample)
  return sample

images = load_image('face_dection/lfw1000',64)

def integral(img):
    # 积分图
    integ_graph = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    for x in range(img.shape[0]):
        sum_clo = 0
        for y in range(img.shape[1]):
            sum_clo = sum_clo + img[x][y]
            integ_graph[x][y] = integ_graph[x - 1][y] + sum_clo
    return integ_graph

def Haar(interM, weigth, height, size=1, deep=2):
    # 计算一种Haar特征
    dst = []
    for i in range(height - deep + 1):
        dst.append([0 for x in range(weigth - size)])
        for j in range(weigth - 2 * size + 1):
            if j == 0 and i == 0:
                whithe = int(interM[i + deep - 1][j + size - 1])
            elif i != 0 and j == 0:
                whithe = int(interM[i + deep - 1][j + size - 1]) - int(interM[i - 1][j + size - 1])
            elif i == 0 and j != 0:
                whithe = int(interM[i + deep - 1][j + size - 1]) - int(interM[i + 1][j - 1])
            else:
                whithe = int(interM[i + deep - 1][j + size - 1]) + int(interM[i - 1][j - 1]) - int(
                    interM[i + 1][j - 1]) - int(interM[i - 1][j + size - 1])
            _i = i
            _j = j + size
            if _i == 0:
                black = int(interM[_i + deep - 1][_j + size - 1]) - int(interM[_i + 1][_j - 1])
            else:
                black = int(interM[_i + deep - 1][_j + size - 1]) + int(interM[_i - 1][_j - 1]) - int(
                    interM[_i + 1][_j - 1]) - int(interM[_i - 1][_j + size - 1])
            dst[i][j] = black - whithe
    return [i for j in dst for i in j]

def extract_feature_image(img):
    #提取出Haar特征
    _w, _h = img.shape[0], img.shape[1]
    in_b = integral(img)
    Haar_b = Haar(in_b, _h, _w)
    return Haar_b

feature_types = ['type-2-x', 'type-2-y','type-3-x', 'type-3-y', 'type-4'] 
X = []
for img in images[:100]:
    res = extract_feature_image(img)
    X.append(res)
X = np.array(X)
print(X.shape)
# y是50张人脸和50张不是人脸，作为标签
y = np.array([1] * 50 + [0] * 50)
# 分割验证集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100, random_state=0, stratify=y)

feature_coord, feature_type = haar_like_feature_coord(width=images.shape[2],
                                                      height=images.shape[1],
                                                      feature_type=feature_types)


# 建立一个基于随机森林弱分类器的adaboost分类模型
clf = RandomForestClassifier(n_estimators=1000, max_depth=None, max_features=100, n_jobs=-1, random_state=0)
bclf = AdaBoostClassifier(base_estimator=clf, n_estimators=clf.n_estimators)
bclf.fit(X_train, y_train)
# roc和 auc 的值
auc_full_features = roc_auc_score(y_test, bclf.predict_proba(X_test)[:, 1])
print(auc_full_features)


# 筛选重要的特征，下面画出最重要的6个
idx_sorted = np.argsort(bclf.feature_importances_)[::-1]

fig, axes = plt.subplots(3, 2)
for idx, ax in enumerate(axes.ravel()):
    image = images[0]
    image = draw_haar_like_feature(image, 0, 0,
                                   images.shape[2],
                                   images.shape[1],
                                   [feature_coord[idx_sorted[idx]]])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('The most important features')



big3 = cv2.imread('Big3.jpg',1)
gray_big3 = cv2.imread('Big3.jpg',0)
solvay=cv2.imread('Solvay.jpg',1)
gray_solvay=cv2.imread('Solvay.jpg',0)

def face_detection(path):
    face_cascade = cv2.CascadeClassifier('./harr.xml')
    img = cv2.imread(path,1)
    gray = cv2.imread(path,0)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face:
        img = cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0),2)
    cv2.namedWindow('Face')
    cv2.imwrite('./'+str(path),img)
    
face_detection('Big3.jpg')
face_detection('Solvay.jpg')
plt.imshow(plt.imread('Big3.jpg'))
plt.imshow(plt.imread('Solvay.jpg'))






















