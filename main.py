import torch
from torch.utils.data import Dataset, DataLoader
from model import Net
from dataset import SignDataset
import torch.optim as optim
from tqdm import tqdm
import pickle
import torch.nn as nn
import torch.nn.functional as F

classes = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
class2index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, 'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17, 't': 18, 'u': 19, 'v': 20, 'w': 21, 'x':22,'y': 23}


def accuracy(pred, gt):
    pred_l = torch.argmax(pred, axis=1)
    return (float(torch.sum(pred_l == gt).item())/ len(gt))


save_path = '../data_augmented'

train_set = SignDataset(save_path, 'train_keys.pkl')
test_set = SignDataset(save_path, 'test_keys.pkl')
train_set.__getitem__(0)
print(train_set.__len__())

train_loader = DataLoader(train_set, shuffle = True, batch_size = 4)
test_loader = DataLoader(test_set, shuffle = False, batch_size = 4)

net = Net(24)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

loss_ = 1e6

for epoch in range(50):
    running_loss = 0.0
    running_accuracy = 0.0
    for sample,label in tqdm(train_loader):
        optimizer.zero_grad()
        output = net(sample)
        loss = criterion(output, label)
        loss.backward()
        acc = accuracy(output, label) 
        optimizer.step()
        running_loss += loss.item()
        running_accuracy += acc
        
    print("train loss: ",running_loss/len(train_loader))
    print("train accuracy: " ,running_accuracy/len(train_loader))
    
    validation_loss = 0.0
    validation_accuracy = 0.0
    
    with torch.no_grad():
        for sample,label in tqdm(test_loader):
            optimizer.zero_grad()
            output = net(sample)
            #print(output)
            loss = criterion(output, label)
            acc = accuracy(output, label) 
            validation_loss += loss.item()
            validation_accuracy += acc
    
    print("test loss: ", validation_loss/len(test_loader))
    print("test accuracy: ",validation_accuracy/len(test_loader))
    
    if validation_loss< loss_:
        loss_ = validation_loss
        torch.save(net.state_dict(),'./ckpt/epoch_'+str(epoch)+'.pt')