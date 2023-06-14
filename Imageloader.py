from itertools import count
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as tfms
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from itertools import chain
#import scipy as sc
import os
import PIL
import PIL.Image as Image
#import seaborn as sns
import warnings
#import nibabel as nib
#from loader_mnist import dataset
#from oversampling import oversample
#def custom_dataset(fpath):
#    return dataset(fpath)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

'''def sample_return(root):
    newdataset =[]
    classes = [d for d in os.listdir(root)]
    classes = sorted(classes)
    diseases = {'Breast_0': 0, 'Breast_1':1, 'Oct_0':2, 'Oct_1':3, 'Oct_2':4, 'Oct_3':5, 'Oct_4':6, 'Chestxray_0':7, 'Chestxray_1':8, 'Chestxray_2':9, 'Chestxray_3':10, 'Chestxray_4':11, 'Chestxray_5':12, 'Chestxray_6':13, 'Chestxray_7':14, 'Chestxray_8':15, 'Chestxray_9':16, 'Chestxray_10':17, 'Chestxray_11':18, 'Chestxray_12':19, 'Chestxray_13':20}
    cls_to_idx = {classes[i]:i for i in range(len(classes))}
    for cls_name in classes:
        for im in os.listdir(os.path.join(root,cls_name)):
            path1 = os.path.join(os.path.join(root, cls_name), im)
            img1 = im.split('_')[0] +'_' +im.split('_')[1]
            dis = diseases[img1]
            item = (path1, cls_to_idx[cls_name], cls_to_idx[cls_name], dis)
            #item = [(im,cls_to_idx[cls_name]) for im in os.listdir(os.path.join(root,cls_name))]
            newdataset.append(item)
    
    #newdataset = list(chain.from_iterable(newdataset1))
    return newdataset'''

def sample_return(root):
    newdataset =[]
    classes = [d for d in os.listdir(root)]
    classes = sorted(classes)
    modalities = {'CT':0, 'OCT':1, 'X-ray':2, 'US':3, 'MRI':4}
    organs = {'Brain':0, 'Breast':1, 'Chest':2, 'Retina':3}
    diseases = {'Bnormal': 0, 'benign':1, 'malignant':2, 'ChestNormal':3, 'COVID':4, 'PNEUMONIA':5, 'NORMAL':6, 'CNV':7, 'DME':8, 'DRUSEN':9, 'glioma':10, 'meningioma':11, 'pituitaryTumor':12}

    for cls_name in classes:
        for im in os.listdir(os.path.join(root,cls_name)):
            path1 = os.path.join(os.path.join(root, cls_name), im)
            img_o = im.split('_')[0] 
            img_m = im.split('_')[1]
            img_d = im.split('_')[2]
            modality = modalities[img_m]
            organ = organs[img_o]
            dis = diseases[img_d]
            item = (path1, modality, organ, dis)
            #item = [(im,cls_to_idx[cls_name]) for im in os.listdir(os.path.join(root,cls_name))]
            newdataset.append(item)
    
    #newdataset = list(chain.from_iterable(newdataset1))
    return newdataset




def similar_index(yd,yd1):
    #ym = ym.cpu().numpy()
    #yo = yo.cpu().numpy()
    yd = yd.cpu().numpy()
    #ym1 = ym1.cpu().numpy()
    #yo1 = yo1.cpu().numpy()
    yd1 = yd1.cpu().numpy()
    '''print(ym)
    print(ym1)
    print(yo)
    print(yo1)'''
    #print(yd)
    #print(yd1)
    index = np.where( (yd == yd1))[0].tolist()
    return index

class customDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        #classes, class_to_idx = find_classes(root)
        #samples = oversample(root)
        print(os.path.basename(root))
        if (os.path.basename(root) == 'train'):
            samples = sample_return(root)
            samples1 = random.sample(samples, len(samples))
       # print(len(samples))
        self.root = root
      
        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = samples
        self.samples1 = samples1

        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        #print('index:',(index))
        img, ym, yo, yd = self.samples[index]
        img1, ym1, yo1, yd1 = self.samples1[index]
        #print(img)
        #print(img1)
        
        a = np.load(img)
        a1 = np.load(img1)
        #print(a1)
        #print(a1.dtype)
       #print(a2.dtype)
        #sample1 = torch.from_numpy(np.load(path1))
        #sample2 = torch.from_numpy(np.load(path2))
       # print(sample1.dtype)
       # print(sample2.dtype)
        #sample1 = Image.open(sample1).convert("RGB")
        
        img = Image.fromarray(a)
        #print(sample1)
        img1 = Image.fromarray(a1)
        if self.transform is not None:
            img = self.transform(img)
            img1 = self.transform(img1)

        if self.target_transform is not None:
            ym = self.target_transform(ym)
            yo = self.target_transform(yo)
            yd = self.target_transform(yd)
            ym1 = self.target_transform(ym1)
            yo1 = self.target_transform(yo1)
            yd1 = self.target_transform(yd1)
        return img, ym, yo, yd, img1, ym1, yo1, yd1
    
    def __len__(self):
        return len(self.samples)


'''def sample_return(root):
    newdataset =[]
    classes = [d for d in os.listdir(root)]
    classes = sorted(classes)
    cls_to_idx = {classes[i]:i for i in range(len(classes))}
    for cls_name in classes:
        for im in os.listdir(os.path.join(root,cls_name)):
            path1 = os.path.join(os.path.join(root, cls_name), im) 
            item = (path1, cls_to_idx[cls_name])
            #item = [(im,cls_to_idx[cls_name]) for im in os.listdir(os.path.join(root,cls_name))]
            newdataset.append(item)
    
    #newdataset = list(chain.from_iterable(newdataset1))
    return newdataset

class customDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        #super(customDataset, self).__init__()
        samples = sample_return(root)
        #print(samples)
        self.root = root
        self.transform = transform
        self.target_tarnsform = target_transform
        self.samples = samples
        #self.n =n
        #a1,a2 = self.samples[0]
        #print((self.samples[0]))
        #print(a1)
        #print(a2)

        
    def __getitem__(self, index):
        #print(len(self.samples))
        path,  target = self.samples[index]  
        img = np.load(path)
        #print(img)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
            

        if self.target_tarnsform is not None:
            target = self.target_tarnsform(target)
            
        
        return img, target

    def __len__(self):
        #print(len(self.samples))
        return len(self.samples)
        #return self.n

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=2),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])'''

'''trainpath = '/storage/asim/Main_combine_dataset/Query'

trainset = customDataset(trainpath, transform=None, target_transform=None)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,  batch_size=32, num_workers=4)
newdataset = sample_return_test(trainpath)
print((newdataset))'''
