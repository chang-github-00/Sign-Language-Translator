import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageEnhance
                
import pickle
import random
import os

path ='../dataset_yj/'
save_path = '../data_augmented'
label_list = []


classes = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
class2index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, 'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17, 't': 18, 'u': 19, 'v': 20, 'w': 21, 'x':22,'y': 23}


# 不需要重复跑
# 数据增强
for p in os.listdir(path):
    print("processing",p)
    if p >='a' and p<='z':
        cnt = 1
        for pic in os.listdir(os.path.join(path,p)):
            if 'IMG' in pic:
                save_pic_path = os.path.join(save_path,p)
                img = cv.imread(os.path.join(os.path.join(path,p),pic))
                new_img = cv.resize(img,(128,128)) # original
                plt.imsave(save_pic_path +'_'+str(cnt)+'.jpg', new_img)
                label_list.append(p+'_'+str(cnt))
                cnt +=1
                

test_list = []
index = [random.randint(1,50) for i in range(5)]
for c in classes:
    for i in index:
        test_list.append(c+'_'+str(i))
train_list = list(set(label_list) - set(test_list))
with open('train_keys.pkl','wb') as f:
    pickle.dump(train_list,f)
with open('test_keys.pkl','wb') as f:
    pickle.dump(test_list,f)