import torch
from torch.utils.data import Dataset, DataLoader
from model import Net
from dataset import SignDataset
import torch.optim as optim
from tqdm import tqdm
import pickle
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np

classes = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
class2index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, 'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17, 't': 18, 'u': 19, 'v': 20, 'w': 21, 'x':22,'y': 23}

save_path = '../data_augmented'

train_set = SignDataset(save_path, 'train_keys.pkl')
test_set = SignDataset(save_path, 'test_keys.pkl')
train_set.__getitem__(0)
print(train_set.__len__())

train_loader = DataLoader(train_set, shuffle = True, batch_size = 4)
test_loader = DataLoader(test_set, shuffle = False, batch_size = 1)

net = Net(24)

net.load_state_dict(torch.load('./ckpt/epoch_'+str(21)+'.pt'))

img = cv.imread('f.jpg')
x = train_set.transform(img)
x = torch.from_numpy(x.reshape(1, 3, 128, 128).astype(np.float32))
y = net(x)
index = torch.argmax(y).item()
print(index)