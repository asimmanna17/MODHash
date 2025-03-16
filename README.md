# # Codes for paper:  Structured hashing with deep learning for modality, organ, and disease content sensitive medical image retrieval [(Paper)][(https://www.nature.com/articles/s41598-025-93418-2/)]


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

### Results
Score of mean avergae precision for different hash code lengths.
| mAP@p |  16  |  32 | 48| 64|
|--|--|--|--|--|
| mAP@10 | 0.8242| 0.9068 |0.9855 |0.9368 |
| mAP@100 | 0.7854 |0.8878 |0.9806| 0.9336|
| mAP@1000 | 0.7545 |0.8849 |0.9556 |0.9242|

Score of normalized discounted cummulative gain for different hash code lengths.
| nDCG@p |  16  |32 | 48| 64|
|--|--|--|--|--|
| nDCG@10 | 0.8805| 0.9398| 0.9886| 0.9582 |
| nDCG@100 |0.9006| 0.9481 |0.9901|0.9640 |
| nDCG@1000 | 0.9236| 0.9615| 0.9886 |0.9688 |


**Contributor**

The codes/model is contributed  by

<a href="https://www.linkedin.com/in/asimmanna17/">Asim Manna</a>, </br>
Centre of Excellence in Artificial Intelligence, </br>
Indian Institute of Technology Kharagpur </br>
email: asimmanna17@kgpian.iitkgp.ac.in </br> 

