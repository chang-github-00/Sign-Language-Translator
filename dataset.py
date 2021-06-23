import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import os

classes = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
class2index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, 'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17, 't': 18, 'u': 19, 'v': 20, 'w': 21, 'x':22,'y': 22}

class SignDataset(Dataset):
    def __init__(self, save_path, keys_path):
        self.samples = []
        self.label = []
        keys = pickle.load(open(keys_path,"rb"))
        print("start loading dataset")
        for key in keys:
            #print(key)
            label = class2index[key.split('_')[0]]
            img = cv.imread(os.path.join(save_path, key+'.jpg'))
            sample = np.array(img)
            #print(np.shape(sample))
            self.label.append(label)
            self.samples.append(sample)
        self.samples = np.array(self.samples)
        self.label = np.array(self.label)
        r = self.samples[:,:,:,0]
        g = self.samples[:,:,:,1]
        b = self.samples[:,:,:,2]
        self.mean = np.mean(r), np.mean(g), np.mean(b)
        self.std =  np.std(r), np.std(g), np.std(b)
    def transform(self, array) :
        result = []
        for i in range(3):
            channel = array[:,:,i]
            m = self.mean[i]
            d = self.std[i]
            normalized = (channel-m)/d
            result.append(normalized)
        return np.array(result)
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        pic = self.samples[idx]
        sample = torch.FloatTensor(self.transform(pic))
        label = torch.tensor(self.label[idx])
        return sample, label