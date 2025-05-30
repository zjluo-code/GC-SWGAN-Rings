Instructions for the Galaxy Ring Detection Program Based on the Semi-Supervised Deep Learning Model GC-SWGAN

Program Name: GC_SWGAN_ring.py

1）Program Overview

This program is based on the semi-supervised deep learning model GC-SWGAN proposed in the paper "Galaxy Morphology Classification via Deep Semi-Supervised Learning with Limited Labeled Data" (https://arxiv.org/abs/2504.00500v2). The original model can be found at https://github.com/zjluo-code/GC-SWGAN. According to the requirements of the task, we have modified the program, and the modified version is named GC_SWGAN_ring.py. It is designed to automatically detect galactic ring structures from high-resolution galaxy images in the DESI Legacy Imaging Surveys (DESI-LS). The program employs a semi-supervised learning framework, combining a small amount of labeled data with a large amount of unlabeled data to achieve efficient identification of galactic rings and has generated a large catalog containing 62,962 ringed galaxies. 

For further details about the code, please refer to our paper: "Detecting Galactic Rings in the DESI Legacy Imaging Surveys with Semi-Supervised Deep Learning."

2）Installation Dependencies

This project is developed using Keras and TensorFlow 2 libraries and uses NVIDIA L40S GPU platforms for accelerated training. To install the required dependencies, use the following command:

pip install -r requirements.txt

Ensure your environment is properly configured to support GPU acceleration.

3）	Data Preparation

Before running the code, please prepare the data according to the following steps:

a)	Use the image cutout tool on the DESI Legacy Imaging Surveys website (http://legacysurvey.org/viewer/) to download over 15,000 unlabeled DECaLS images. Then, convert them into an h5py format file and name it decals_images.h5.

b)	The files train_rings_pos.dat and train_norings_pos.dat in this directory provide the celestial coordinates (with the first column being the right ascension ra and the second column being the declination dec) for 5,173 ringed galaxies and 15,000 non-ringed galaxies, respectively. Please download the DECaLS images for model training from the DESI Legacy Imaging Surveys website based on the coordinates provided in these two files, and convert them into an h5py format file named decals_train.h5.

The GC-SWGAN_ring.py code will directly use these two files during its execution.。


4）Model Training

To start training, run the following script:

python GC-SWGAN_ring.py

You can adjust the training parameters (e.g., batch size, learning rate) according to your needs. During training, the model optimizes the generator, discriminator, and classifier at different stages.

5）Model Evaluation

After the training is completed, the model's performance metrics will be automatically displayed on the screen.
