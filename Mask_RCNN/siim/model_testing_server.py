#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Inspect Ballon Trained Model
# 
# Code and visualizations to test, debug, and evaluate the Mask R-CNN model.

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
import pneumothorax
from mask_functions import mask2rle


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


config = pneumothorax.SiimConfig()
SIIM_DIR = os.path.join(ROOT_DIR, "datasets/pneumothorax")


# In[3]:


print(SIIM_DIR)


# In[4]:


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Notebook Preferences

# In[5]:


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"





# ## Load Validation Dataset

# In[7]:

def generatecsv(csvpath, datasetfolder):
    
    # Load validation dataset
    dataset = pneumothorax.SiimDataset()
    dataset.load_siim(SIIM_DIR, datasetfolder)

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


    # ## Load Model

    # In[8]:


    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)


    # In[9]:


    # Set path to balloon weights file

    # Download file from the Releases page and set its path
    # https://github.com/matterport/Mask_RCNN/releases
    weights_path = "/home/sa-279/Mask_RCNN/logs/siim20190801T1940/mask_rcnn_siim_0026.h5"


    # Or, load the last model you trained
    #weights_path = model.find_last()

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)


    # ## TEST

    # In[26]:


    test_images = dataset.image_ids

    #print(len(test_images))

    test_set = np.empty((0,2))
    total_images = len(test_images)
    for row, image_id in enumerate(test_images):  
        info = dataset.image_info[image_id]
        img_id = info["id"]  # image id as in csv file
        print("processing {} :image {} of {}".format(img_id, row+1, total_images))
        image, image_meta, gt_class_id, gt_bbox, gt_mask =  modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        #print(image.shape)
        result = model.detect([image], verbose=0)
        cols = result[0]
        mask = cols['masks']
        mask = np.array(mask)
        if (np.any(mask)):
            mask = mask.astype(np.uint64)
            mask = mask * 255
            mask = np.squeeze(mask, axis = -1)
            width = mask.shape[1]
            height = mask.shape[0]
            mask = mask.T
            rle = mask2rle(mask,width, height)
            #print(rle)
        else:
            rle = -1
        test_set = np.append(test_set,[[img_id, rle]], axis = 0)
    #print(len(test_set))
    # Run object detection
    #print(test_set.shape)
    csvfile = os.path.join(SIIM_DIR, csvpath)
    np.savetxt(csvfile,test_set, delimiter=',', fmt='%s')

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--csvfile', required=True,
                        metavar="/path/to/dataset/csvfile",
                        help='Name of csv file')
    parser.add_argument('--datasetfolder', required=True,
                        metavar="train test val sample",
                        help='Datset folder name under pneuomothorax')
    args = parser.parse_args()
    assert args.csvfile, "Argument --csvfile is required. Provide name for csv"
    assert args.datasetfolder, "Argument --datasetfolder is required. Provide folder name"
    
    if args.csvfile:
        generatecsv(args.csvfile, args.datasetfolder)

