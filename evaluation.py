import torch
import os
import numpy as np
import random
import operator
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from torchvision import transforms
from PIL import Image

from AlexNet import Encoder, Modality, Organ, Disease
from metrics import mAP, nDCG


random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)


def relevenceClasses(sorted_pool,q_name):
    value = []
    q_labels = q_name.split("_")[0:3]
    for i in range(len(sorted_pool)):
        #print(q_labels)
        sorted_pool_labels = sorted_pool[i][0].split("_")[0:3]
        #print(sorted_pool_labels)
        common_labels = len(set(sorted_pool_labels).intersection(q_labels))
        value.append(common_labels)  
        #print(value)
    value2 = sorted(value, reverse=True)
    return value, value2

def hammingDistance(h1, h2):
    hash_code = h1.shape[1]
    h1norm = torch.div(h1, torch.norm(h1, p=2))
    h2norm = torch.div(h2, torch.norm(h2, p=2))
    distH = torch.pow(torch.norm(h1norm - h2norm, p=2), 2) * hash_code / 4
    return distH



#### Hyperparemetr Details ######
mcode = 16
ocode = 16
dcode = 16
hash_code = mcode+ ocode+ dcode
#model load######################
nModality = 5
nOrgan = 4
nDisease = 13
encoder = Encoder()
mClassifier = Modality(nModality, mcode)
oClassifier = Organ(nOrgan, ocode)
dClassifier = Disease(nDisease, dcode)

if torch.cuda.is_available():
    encoder.cuda()
    mClassifier.cuda()
    oClassifier.cuda()
    dClassifier.cuda()

dataStorePath = './models/'

encoder_path = os.path.join(dataStorePath, f'encoder_{mcode}_{ocode}_{dcode}.pkl')

modality_path = os.path.join(dataStorePath, f'modality_{mcode}_{ocode}_{dcode}.pkl')

organ_path = os.path.join(dataStorePath, f'organ_{mcode}_{ocode}_{dcode}.pkl')

disease_path = os.path.join(dataStorePath, f'disease_{mcode}_{ocode}_{dcode}.pkl')

encoder.load_state_dict(torch.load(encoder_path))
mClassifier.load_state_dict(torch.load(modality_path))
oClassifier.load_state_dict(torch.load(organ_path))
dClassifier.load_state_dict(torch.load(disease_path))

print(encoder_path)
galleryfolderpath = "./data/gallery"
queryfolderpath = "./data/query"
gallery_files = os.listdir(galleryfolderpath)
gallery_files = random.sample(gallery_files, len(gallery_files))
query_files = os.listdir(queryfolderpath)
query_files = random.sample(query_files, len(query_files))
print(len(gallery_files))
querynumber = len((query_files))
print(querynumber)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

gallery = {}
print("\n\n Building Gallery .... \n")
with torch.no_grad():
    # Process each gallery image
    for img in gallery_files:
        image_path = os.path.join(galleryfolderpath, img)

        # Load and transform the image
        image = np.load(image_path)
        # transfer to one channel
        if len(image.shape)!= 2:
            image = np.mean(image,axis=-1)

        image = Image.fromarray(image)
        tensor_image = transform(image).unsqueeze(0).cuda()

        # Pass the tensor through the  model
        x_e = encoder(tensor_image)
        _, mh1 = mClassifier(x_e)
        _, oh1 = oClassifier(x_e)
        _, dh1 = dClassifier(x_e)
        h = torch.cat((mh1, oh1, dh1), dim = 1)
        h = torch.sign(h)
        gallery[img] = h # Store the result in the gallery dictionary
        
        # Clean up
        del tensor_image
    print("\n Building Complete. \n")

    count = 0

    q_prec_10 = 0
    q_prec_100 = 0
    q_prec_1000 = 0
    
    nDCG_list_10 = []
    nDCG_list_100 = []
    nDCG_list_1000 = []

    #print(len(qNimage[0:100]))
    for q_name in query_files:
        count = count+1
        query_image_path = os.path.join(queryfolderpath, q_name)
        # Load and transform the image
        query_image = np.load(query_image_path)
        # transfer to one channel
        if len(query_image.shape)!= 2:
            query_image = np.mean(query_image,axis=-1)
        query_image = Image.fromarray(query_image)
        query_tensor_image = transform(query_image).unsqueeze(0).cuda()

        # Pass the tensor through the model
        q_x_e = encoder(query_tensor_image)
        _, q_mh1 = mClassifier(q_x_e)
        _, q_oh1 = oClassifier(q_x_e)
        _, q_dh1 = dClassifier(q_x_e)
        h_q = torch.cat((q_mh1, q_oh1, q_dh1), dim = 1)
        h_q = torch.sign(h_q)
        dist = {}
        for key, h1 in gallery.items():
            dist[key] = hammingDistance(h1, h_q)

        print(count)   
        ### images with sorted distance 
        sorted_pool_10 = sorted(dist.items(), key=operator.itemgetter(1))[0:10]
       
        sorted_pool_100 = sorted(dist.items(), key=operator.itemgetter(1))[0:100]
        sorted_pool_1000 = sorted(dist.items(), key=operator.itemgetter(1))[0:1000]

        #### mean average precision
        q_prec_10 += mAP(q_name, sorted_pool_10)
        q_prec_100 += mAP(q_name, sorted_pool_100)
        q_prec_1000 += mAP(q_name, sorted_pool_1000)

        ### nDCG
        r_i_10, sorted_r_i_10 = relevenceClasses(sorted_pool_10, q_name)
        r_i_100, sorted_r_i_100 = relevenceClasses(sorted_pool_100, q_name)
        r_i_1000, sorted_r_i_1000 = relevenceClasses(sorted_pool_1000, q_name)
        #print(r_i, sorted_r_i)

        nDCG_value_10 = nDCG(r_i_10, sorted_r_i_10)
        nDCG_list_10.append(nDCG_value_10)

        nDCG_value_100 = nDCG(r_i_100, sorted_r_i_100)
        nDCG_list_100.append(nDCG_value_100)

        nDCG_value_1000 = nDCG(r_i_1000, sorted_r_i_1000)
        nDCG_list_1000.append(nDCG_value_1000)

        if count % 10 == 0:
            print("mAP@10 :", q_prec_10/count)
            print("mAP@100 :", q_prec_100/count)
            print("mAP@1000 :", q_prec_1000/count)
            print('-------------------------------')
            print('nDCG@10:', sum(nDCG_list_10)/len(nDCG_list_10))
            print('nDCG@100:', sum(nDCG_list_100)/len(nDCG_list_100))
            print('nDCG@1000:', sum(nDCG_list_1000)/len(nDCG_list_1000))


print('-----------------------------------------------')       
print("mAP@10 :", q_prec_10/count)
print("mAP@100 :", q_prec_100/count)
print("mAP@1000 :", q_prec_1000/count)
print('-------------------------------')
print('nDCG@10:', sum(nDCG_list_10)/len(nDCG_list_10))
print('nDCG@100:', sum(nDCG_list_100)/len(nDCG_list_100))
print('nDCG@1000:', sum(nDCG_list_1000)/len(nDCG_list_1000))
print(hash_code)
print(f'encoder_{mcode}_{ocode}_{dcode}-100.pkl')
