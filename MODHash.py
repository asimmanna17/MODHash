import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
use_cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from AlexNet import Encoder, Modality, Organ, Disease, Discriminator
from Imageloader import customDataset, similar_index

torch.manual_seed(0)


#### Intilization
modalities = {'CT':0, 'OCT':1, 'X-ray':2, 'US':3, 'MRI':4}
organs = {'Brain':0, 'Breast':1, 'Chest':2, 'Retina':3}
diseases = {'Bnormal': 0, 'benign':1, 'malignant':2, 'ChestNormal':3, 'COVID':4, 'PNEUMONIA':5, 'NORMAL':6, 'CNV':7, 'DME':8, 'DRUSEN':9, 'glioma':10, 'meningioma':11, 'pituitaryTumor':12}

trainingDataPath = "./data/train"
nModality = len(modalities)
nOrgan = len(organs)
nDisease = len(diseases)

#hypeerparametets
mcode = 16
ocode = 16
dcode = 16
dis_code = mcode+ ocode+ dcode
batch_size =256
lr = 0.0001
epochs = 200
alpha = 0.5

# model Intilization
encoder = Encoder()
mClassifier = Modality(nModality, mcode)
oClassifier = Organ(nOrgan, ocode)
dClassifier = Disease(nDisease, dcode)
discriminator = Discriminator(dis_code)
#print(model)
#mdl_map = {}
''''model_path = "/storage/asim/parellel_hashing_store/encode-200.pkl"
m_model_path = '/storage/asim/parellel_hashing_store/modality-200.pkl'
o_model_path = '/storage/asim/parellel_hashing_store/organ-200.pkl'
d_model_path = '/storage/asim/parellel_hashing_store/disease-200.pkl'

encoder.load_state_dict(torch.load(model_path))
mClassifier.load_state_dict(torch.load(m_model_path))
oClassifier.load_state_dict(torch.load(o_model_path))
dClassifier.load_state_dict(torch.load(d_model_path))'''
if torch.cuda.is_available():
    encoder.cuda()
    mClassifier.cuda()
    oClassifier.cuda()
    dClassifier.cuda()
    discriminator.cuda()

### data preprocessing

transform = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


#data loading
trainset = customDataset(trainingDataPath, transform=transform, target_transform=None)
print(len(trainset))
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,  batch_size=batch_size, num_workers=4)
print("\nDataset generated. \n\n")

#optimizers
eff_optimizer = optim.Adam(encoder.parameters(), lr=lr, amsgrad = True)
m_optimizer = optim.Adam(mClassifier.parameters(), lr=lr, amsgrad = True)
o_optimizer = optim.Adam(oClassifier.parameters(), lr=lr, amsgrad = True)
d_optimizer = optim.Adam(dClassifier.parameters(), lr=lr, amsgrad = True)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, amsgrad = True)

#scheduler
m_scheduler = ReduceLROnPlateau(m_optimizer, 'min', factor=0.1, patience=10, verbose=True)
o_scheduler = ReduceLROnPlateau(o_optimizer, 'min', factor=0.1, patience=10, verbose=True)
d_scheduler = ReduceLROnPlateau(d_optimizer, 'min', factor=0.1, patience=10, verbose=True)

multi_loss = nn.CrossEntropyLoss()
bin_loss = nn.BCELoss()
disc_loss = nn.BCEWithLogitsLoss()


def shuffler(h1, h2):
    #print("Shuffler",h1.size(0))
    #print("1", h1.size(1))
    #print("2;", h2.size(1))
    #print("2:", h2.size(0))
    sh = torch.randint(0, 2, (h1.size(0),))  # 1 means keeping h1 and 0 means keeping h2
    #print(sh)
    h1new = torch.matmul(torch.diag(1-sh).float(), h2.cpu())+torch.matmul(torch.diag(sh).float(),h1.cpu())
    h2new = torch.matmul(torch.diag(1-sh).float(), h1.cpu())+torch.matmul(torch.diag(sh).float(),h2.cpu())
    #print(h1new.shape)
    #print(h2new.shape)
    #return h1new, h2new, sh
    return h1new.cuda(), h2new.cuda(), sh


m_Classifier_loss_dict = {}
o_Classifier_loss_dict = {}
d_Classifier_loss_dict = {}
discriminator_loss_dict = {}

temp_d_acc =0

encoder.train()

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch+1}/{epochs}.")
    start_time = time.time()

    m_classifier_loss = 0
    m_classifier_count_full = 0
    
    o_classifier_loss = 0
    o_classifier_count_full = 0
    
    d_classifier_loss = 0
    d_classifier_count_full = 0
    
    discriminator_running_loss = 0
    discriminator_count_full = 0

    for i, data1 in tqdm(enumerate(trainloader, 0)):
        img, ym, yo, yd, img1, ym1, yo1, yd1 = data1
        
        ym, yo, yd = Variable(ym).cuda(), Variable(yo).cuda(), Variable(yd).cuda()
        
        eff_optimizer.zero_grad()
        m_optimizer.zero_grad()
        o_optimizer.zero_grad()
        d_optimizer.zero_grad()
        #print(img)
        img= Variable(img).cuda()
        img1 = Variable(img1).cuda()
        x1 = encoder(img)
        x2 = encoder(img1)
        
        # Modality classifier
        
        mPred, mHash1 = mClassifier(x1)
        _, mHash2 = mClassifier(x2)
        oPred, oHash1 = oClassifier(x1)
        _, oHash2 = oClassifier(x2)
        dPred, dHash1 = dClassifier(x1)
        _, dHash2 = dClassifier(x2)
        
        mLoss = multi_loss(torch.squeeze(mPred), ym)
        oLoss = multi_loss(oPred, yo)
        dLoss = multi_loss(dPred, yd)
        
        total_loss =mLoss+oLoss+dLoss
        total_loss.backward()
        m_optimizer.step()
        o_optimizer.step()
        d_optimizer.step()
        eff_optimizer.step()
        
        
        m_classifier_count_full += 1
        o_classifier_count_full += 1
        d_classifier_count_full += 1
        
        
        x1 = encoder(img)
        x2 = encoder(img1)
        
        # reproduce output for discriminator
        
        mPred, mHash1 = mClassifier(x1)
        _, mHash2 = mClassifier(x2)
        oPred, oHash1 = oClassifier(x1)
        _, oHash2 = oClassifier(x2)
        dPred, dHash1 = dClassifier(x1)
        _, dHash2 = dClassifier(x2)
        
        h1 = torch.cat((mHash1, oHash1, dHash1),dim=1)
        h2 = torch.cat((mHash2, oHash2, dHash2), dim=1)    
        #print(h1.shape)
        index = similar_index(yd,yd1)
        #print(index)
        # Discriminator---------------------------------------------------------
        if (len(index)>0):
            d_h1 = h1[index]
            d_h2 = h2[index]
            d_h1, d_h2, dlabels = shuffler(h1, h2)
            #print(d_h1.shape)
            dlabels = Variable(dlabels).cuda()
            d_input = torch.stack((d_h1, d_h2), 1)
            d_input = Variable(d_input).cuda()
            #print("1",d_input.shape)
            discriminator_optimizer.zero_grad()
            
            d_output = discriminator(d_input).view(-1)
            #d_output = Variable(d_output).cuda()
            #print(dlabels.shape)
            #print(d_output.shape)
            #discriminator_loss = alpha*disc_loss(dlabels.float(), d_output)
            discriminator_loss = alpha*disc_loss(d_output, dlabels.float())
            discriminator_loss.backward()
            discriminator_optimizer.step()
            discriminator_count_full += 1
        
        discriminator_running_loss += discriminator_loss.item()/alpha
        
        m_classifier_loss += mLoss.item()

        o_classifier_loss += oLoss.item()

        d_classifier_loss += dLoss.item()

    end_time = time.time()
    epoch_time = end_time - start_time

    m_Classifier_loss_dict[epoch] = m_classifier_loss/m_classifier_count_full
    o_Classifier_loss_dict[epoch] = o_classifier_loss/o_classifier_count_full
    d_Classifier_loss_dict[epoch] = d_classifier_loss/m_classifier_count_full
    discriminator_loss_dict[epoch] = discriminator_running_loss/discriminator_count_full

    print("Time taken for one epoch:", epoch_time, "seconds")
    print(f'Modality classification loss: {m_Classifier_loss_dict[epoch]}')
    print(f'Organ classification loss: {o_Classifier_loss_dict[epoch]}')
    print(f'Disease classification loss: {d_Classifier_loss_dict[epoch]}')
    
    d_scheduler.step(d_Classifier_loss_dict[epoch])
    
    # Model Save
    dataStorePath = './models/'

    encoder_path = os.path.join(dataStorePath, f'encoder_{mcode}_{ocode}_{dcode}.pkl')
    torch.save(encoder.state_dict(), encoder_path)

    modality_path = os.path.join(dataStorePath, f'modality_{mcode}_{ocode}_{dcode}.pkl')
    torch.save(mClassifier.state_dict(), modality_path)

    organ_path = os.path.join(dataStorePath, f'organ_{mcode}_{ocode}_{dcode}.pkl')
    torch.save(oClassifier.state_dict(), organ_path)

    disease_path = os.path.join(dataStorePath, f'disease_{mcode}_{ocode}_{dcode}.pkl')
    torch.save(dClassifier.state_dict(), disease_path)
   
    print('Weight files saved') 
    print('-------------------------------------------------------------------------------------------')
print(dis_code)

