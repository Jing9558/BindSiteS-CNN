# BindSiteS-CNN
**This is a repository for  BindSiteS-CNN under preparation. Not yet completed and may contain mistakes / non-optimal code**

This is a demo of the official PyTorch implementation of our paper in submission: *Classification of Protein Binding Sites using a Spherical Convolutional Neural Network*

## Dependencies
The PyTorch implementation of ```DeepSphere``` needs to be installed first. (https://github.com/deepsphere/deepsphere-pytorch)

It is recommended to create the corresponding environment according to the requirements of the ```DeepSphere``` page.

To load the dataset we use the libaray ```trimesh``` and make it work efficiently you will aslo need to install ```pyembree```.

## Dataset setup
Set ```DATA_DIR``` environment variable to a directory that will contain the datasets : ```export DATA_DIR=/path_to_a_dir(Data)```
Download and extract the databases containing the surface(ply) files to this set path (https://drive.google.com/drive/folders/1j-NBDQwtcTSoZXLh8_sPe3Imh9qohWlg?usp=sharing).
Folders of surfaces should be placed in the sub-folder of the corresponding data set within ```Data```

## Training Demo
### TOUGH-C1 Classification task
Create a new folder ```data``` under ```TOUGHC1``` and copy the surface folder corresponding to TOUGH-C1 into it.

Within TOUGHC1: ```python train.py log_dir my_run model_path model.py augmentation 5 batch_size 32 learning_rate 0.05 num_workers 8```

Valid with steroid:  ```python test.py log_dir my_run model_path model.py augmentation 5 batch_size 1 num_workers 8```

### TOUGH-M1 Similarity comparison task
Create a new folder ```data``` under ```TOUGHM1``` and copy the surface folder corresponding to TOUGH-M1 and prospeccts into it.

Within TOUGHM1: ```python train.py log_dir my_run_0 model_path model.py augmentation 5 batch_size 128 learning_rate 0.0005 num_workers 8 test_every_n 1 loss_margin 1.25 fold_n 0```

Valid with prospeccts:  ```python test.py log_dir my_run model_path model.py augmentation 5 batch_size 1 num_workers 8```
