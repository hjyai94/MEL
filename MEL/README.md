# Mutual Ensemble Learning for Brain Tumor Segmentation
> This repository is the work of "Mutual Ensemble Learning for Brain Tumor Segmentation" based on pytorch implementation.


## Requirements
* The code has been written in Python 3.6.13 and Pytorch 1.8.0
* Make sure to install all the libraries given in requirement.txt, and you can do so by the following command

```pip install -r requirement.txt```

* (Optional) install ANTs N4BiasFieldCorrection and add the location of the ANTs binaries to the PATH environmental variable. 

## Dataset
[BRATS 2018](https://www.med.upenn.edu/sbia/brats2018/data.html) dataset was chosen to substantiate
our proposed method. It contains the 3D multimodal brain MRI data of 285 labeled training subjects and 
66 subjects which have to be submitted for online validation. We randomly split the 285 labeled
training data into training and testing set with the rate of 4:1. Our split is provided in 'datalist' directory. 

## How to preprocess the dataset?
* Download the BRATS 2018 data and place it in data folder.(Visit [this](https://www.med.upenn.edu/sbia/brats2018/data.html) link to downlaod the data. You
need to register for the challenge)

* Generate brain mask

To crop image patch, you have to generate brain mask. 
```
$ cd data_preprocessing
$ python generate_mask.py
```
* To perform bias field correction, you have to perform correction for each directory of dataset. (Optional)

```
$ cd data_preprocessing
$ python n4correction Brats2018/MICCAI_BraTS_2018_Data_Training/HGG
```

## How to run $MEL-DM^2$?
* training

```
$ python train.py --num_epochs=1000 --net1=DM --net2=DM
--num_round=1
```

* testing

It is corresponding to $MEL-DM^2$ in below table. 


```$ python test.py --net1=DM --net2=DM --num_round=1 --test_epoch=1000```

## How to run $MEL-DM-Unet$?
* training

```
$ python train.py --num_epochs=1000 --net1=DM --net2=Unet --num_round=2
```

* testing

It is corresponding to $MEL-DM-Unet$ in below table. 

```
$ python test.py --net1=DM --net2=Unet --num_round=2 --test_epoch=1000 
```


## How to run $MEL-Unet^2$?
* training

```
$ python train.py --net1=Unet --net2=Unet --num_round=3 --gpu=0,1 --num_epochs=1000
```

* testing
It is corresponding to $MEL-Unet^2$ in below table. 

```
$ python test.py --net1=Unet --net2=Unet --num_round=3 --test_epoch=1000
```

# Results
| Method       | Dice-Com. (%)| Dice-Core (%) | Dice-Enh.  (%) |
| ------       | ------ | ------   | ------ |
| $MEL-DM^2$   | 88.19 | 81.67   | 75.68 |
| $MEL-DM-Unet$  | 89.41 | 84.43   | 77.60 |
| $MEL-Unet^2$ | 89.34 | 84.89   | 78.45 |

* test online
You can generate submission results for  BRATS 2018 online validation set to evaluate the segmentation performance. The performance on the online validation set of the BRATS 2018 can only be assessed by the online evaluation server because the groud truth is unavailable. Moreover, We use entire training set of BRATS 2018 to train our proposed model and submit the result.
```
$ python test_online.py --net1=Unet --net2=Unet --num_round=3 --test_epoch=1000 
```


