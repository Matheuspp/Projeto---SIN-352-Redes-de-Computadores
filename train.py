import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from collections import OrderedDict

# ### Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# ### GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_loaders(path):
     # ### define image transformations
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
    test_transforms =  transforms.Compose([transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])

    # ### setting loaders
    train_set = datasets.ImageFolder(f'{path}/train', transform=train_transforms)
    val_set = datasets.ImageFolder(f'{path}/val', transform=train_transforms)
    test_set  = datasets.ImageFolder(f'{path}/test', transform=test_transforms)

    # ### dataloaders
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=64, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader

def define_model():
    model = models.resnet34(pretrained=True) 

    classifier = nn.Sequential(OrderedDict([
                            ('fc', nn.Linear(512, 3))
                            ]))

    # ### freezing layers
    for weights in model.parameters():
        weights.requires_grad = False

    model.fc = classifier      
    model.to(device)

    return model

def validate(model, val_loader, criterion, val_loss):
    model.eval() 
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        loss = criterion(output, labels)
        val_loss += loss.item()*images.size(0)
            
    val_loss = val_loss/len(val_loader.dataset)
    return val_loss

def train(model, optimizer, train_loader, criterion, train_loss):
    model.train() # setando modelo para treinamento
    for data, target in train_loader:
        data, target = data.to(device), target.to(device) # movendo tensors para GPU
        optimizer.zero_grad() 
        output = model.forward(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
            
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    return train_loss

def run(model, optimizer, train_loader, val_loader, criterion, epochs):
    val_loss_min = np.inf
    start_at = time.time() 
    remain_ep = epochs
    train_ = []
    val = []
    for ep in range(epochs):
        train_loss = 0.0 
        val_loss = 0.0
        remain_ep -= 1
        
        train_loss = train(model, optimizer, train_loader, criterion, train_loss)
        val_loss = validate(model, val_loader, criterion, val_loss)           
        
        # ### savinf losses
        train_.append(train_loss)
        val.append(val_loss)        
        
        tempo_treino = time.time() - start_at # ### training time
        if ep == 0:
          tempo_epoch = (tempo_treino // 60) # ### training for each epoch
          
        print('training time {:.0f}m {:.0f}s    time left: {:.0f} minutes'.format(tempo_treino // 60, tempo_treino % 60, tempo_epoch*remain_ep))
        print('epoch: {} \t train_loss: {:.7f} \t val_loss: {:.7f}'.format(ep+1, train_loss, val_loss))
        
        if val_loss <= val_loss_min:
            print('savind model...')
            torch.save(model.state_dict(), 'Models/resnet32-vpn.pth')
            val_loss_min = val_loss 

    return train_, val

def accuracy(model):
    model.to(device)
    test_loss = 0.0
    correct_classes = list(np.zeros(3)) 
    all_classes = list(np.zeros(3))
    
    model.eval() 
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)    
        test_loss += loss.item()*data.size(0)
    
        prediction = torch.max(output, 1)[1] 
        correct = prediction == target       
        for i in range(target.shape[0]):     
          label = target.data[i]            
          
          correct_classes[label] += correct[i].item()
          all_classes[label] += 1    
    
    test_loss = test_loss / len(test_loader.dataset)
    print('test_loss : {:.6f}'.format(test_loss))  
    acc = np.sum(correct_classes) / np.sum(all_classes)
    chat = np.sum(correct_classes[0]) / np.sum(all_classes[0])
    email = np.sum(correct_classes[1]) / np.sum(all_classes[1])
    hangout = np.sum(correct_classes[2]) / np.sum(all_classes[2])

    print(f'Chat Accuracy:{chat} {int(np.sum(correct_classes[0]))}/{int(np.sum(all_classes[0]))}')
    print(f'Email Accuracy:{email} {int(np.sum(correct_classes[1]))}/{int(np.sum(all_classes[1]))}')
    print(f'Hangout Accuracy:{hangout} {int(np.sum(correct_classes[2]))}/{int(np.sum(all_classes[2]))}')

    print(f'Total Accuracy:{acc} {int(np.sum(correct_classes))}/{int(np.sum(all_classes))}')

def train_val_plot(train, val):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(221)
    ax.plot(train)
    ax.set_title('train loss')
    ax = fig.add_subplot(222)
    ax.plot(val,'r')
    ax.set_title('validation loss')
    ax = fig.add_subplot(223)
    ax.plot(train, 'b')
    ax.plot(val, 'r')
    ax.set_title('train and validation')

if __name__ in '__main__':
    train_loader, val_loader, test_loader = set_loaders('img_dataset')
    model = define_model()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() 
    
    train_, val = run(model, optimizer, train_loader, val_loader, criterion, 30)
    info = {'train':train_, 'val':val}
    with open('losses.json', 'w') as fp:
        json.dump(info, fp, indent=2)

    # ### getting accuracy from a previews trained model
    model.load_state_dict(torch.load('Models/resnet32-vpn.pth')) 
    accuracy(model) 
    train_val_plot(train_, val)

