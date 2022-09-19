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
