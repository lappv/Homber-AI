#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:02:31 2018

@author: tthnguye
"""
#import from core
import cv2
import numpy as np
from numpy.linalg import inv
import urllib

#import from our files
from get32points import main_32

def find_contours(im):
    thresh=np.uint8(im)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours#return vector contours

#points chinh la contours nghia la list cac vong tron 
#contours la list cac array vong tron dc khoanh vung
def find_latlong(contours,fpp,h,w):
    #contours tra lai gia tri bi nguoc nen phai dao vi tri truoc
    j=0
    for cnt in contours:
        try:
            cnt=np.squeeze(cnt,axis=1)
        except:
            cnt=cnt
        a=np.zeros_like(cnt)
        for i in range(len(cnt)):
            a[i][0]=cnt[i][1]
            a[i][1]=cnt[i][0]
        contours[j]=a.copy()
        j+=1
    #cong viec convert bat dau tu day
    f=np.zeros_like(fpp)
    for i in [0,1,3]:
        f[i][1]=fpp[i][1]-fpp[0][1]
        f[i][0]=fpp[i][0]-fpp[0][0]   
    X= np.matrix([f[1],f[3]]) # map
    Y= np.matrix([ [0,w],[h,0]]) #img
    M=inv(Y)*X
    latlong=[]
    for cnt in contours:
        latlong.append(cnt*M+np.array(fpp[0]))
    return latlong

def map_to_img(fpp, path, lat_long):
    h,w=cv2.imread(path).shape[0:2]
    f=np.zeros_like(fpp)
    for i in [0,1,3]:
        f[i][1]=fpp[i][1]-fpp[0][1]
        f[i][0]=fpp[i][0]-fpp[0][0]
    pts2=np.array(lat_long)-np.array(fpp[0])
    X= np.matrix([f[1],f[3]]) # map
    Y= np.matrix([[0,w],[h,0]]) #img
    M=inv(X)*Y
    re=np.round(np.array(pts2*M))
    a=np.zeros_like(re)
    for i in range(len(re)):
        a[i][0]=re[i][1]
        a[i][1]=re[i][0]
    return a

def predict_grass(path,width,height,proba,m_grass):
    X=cv2.imread(path)
    shapeX=X.shape[0:2]
    X=cv2.resize(X, ( width , height ))
    X=X.astype(np.float32)
    X=X/255.0
    pr1= m_grass.predict( np.array([X]) )[1][0]
    pr=pr1.reshape(( height ,  width) )
    X=(pr>proba).astype(float)
    X=cv2.resize(X , (shapeX[1] , shapeX[0]))
    return X


  #trong nay da co lenh resize anh
def predict_latlong_house_path(path,width,height,fpp,m_house):
    X=cv2.imread(path)
    return do_predict_latlong_house(X, width, height, fpp, m_house)

def predict_latlong_house_url(url,width,height,fpp,m_house):
    url_response = urllib.request.urlopen(url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    X = cv2.imdecode(img_array, 1)
    return do_predict_latlong_house(X, width, height, fpp, m_house)

def do_predict_latlong_house(X,width,height,fpp,m_house):
    shapeX=X.shape[0:2]
    X=cv2.resize(X, ( width , height ))
    X=X.astype(np.float32)
    X=X/255.0
    pr1= m_house.predict( np.array([X]) )[0]
    pr=pr1.reshape((height ,width ,2) ).argmax( axis=2 )
    X=(1-pr).astype(float)
    X=cv2.resize(X , (2048, 2048))
    points=main_32(X,2048)#tim 32 diem moi
    Xtem=np.zeros_like(X)
    a=[]#doan nay de dao vi tri points
    for i in range(len(points)):
        a.append([0,0])
    for i in range(len(points)):
        a[i][0]=points[i][1]
        a[i][1]=points[i][0]
    X=cv2.drawContours(Xtem, [np.array(a).astype(int)], -1, 255, thickness=-1)
    X=cv2.resize(X , (shapeX[1] , shapeX[0]))
    contours=find_contours(X)
    cnt=contours.copy()
    long_lat=find_latlong(contours,fpp,shapeX[0]-1,shapeX[1]-1)
    return long_lat,cnt



#ham nay se long anh predict co vao nha trung tam
#grass la array anh tu ham predict grass, 3 channels
#points la pixel anh nha trung tam
def grass_house(grass,fpp,points,path):
    shapeX=grass.shape[0:2]
    Xtem=np.zeros_like(grass)
    Y=cv2.drawContours(Xtem, points, -1, 255, thickness=-1)
    grass[:,:]=grass[:,:]*(Y[:,:]==255).astype(int)
    contours=find_contours(grass*255)
    long_lat=find_latlong(contours,fpp,shapeX[0]-1,shapeX[1]-1)
    return long_lat 

