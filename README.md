# # Codes for paper:  Structured hashing with deep learning for modality, organ, and disease content sensitive medical image retrieval [(Paper)](https://www.nature.com/articles/s41598-025-93418-2/)


## Prerequisites
* Ubuntu\* 20.04
* Python\* 3.9
* NVidia\* GPU for training
* 16GB RAM for inference


## Datasets
### Large size dataset:
  Figshare [(link)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427/5), kaggle [(link)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) and Mendeley [(link)](https://data.mendeley.com/datasets/rscbjbr9sj/2).
### Small size dataset:
MedMNIST v2. (#medmnist) [(link)](https://zenodo.org/record/6496656), [(link2)](https://medmnist.com/)


Sample dataset availabel on './data/'.

## Train
```
python MODHash.py
```
Trained model will be saved in './models/'.

## Evalutions
Trained models are available in [Here](https://iitkgpacin-my.sharepoint.com/:f:/g/personal/asimmanna17_kgpian_iitkgp_ac_in/EhAbL4IyLiFFrkMdJbRIuHMBj8cHos1ThDWzZN-nrSRzeg?e=mx7o1N).
```
python evalution.py
```



**Contributor**

The codes/model is contributed  by

<a href="https://www.linkedin.com/in/asimmanna17/">Asim Manna</a>, </br>
Centre of Excellence in Artificial Intelligence, </br>
Indian Institute of Technology Kharagpur </br>
email: asimmanna17@kgpian.iitkgp.ac.in </br> 

