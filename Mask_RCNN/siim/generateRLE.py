import numpy as np
import pydicom
from mask_functions import mask2rle, rle2mask
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(1)
dataset_dir =""
dataset_dir = os.path.join(dataset_dir, "train")
annotations = pd.read_csv(image_path=os.path.join(dataset_dir,"train-rle.csv"),header=None)
annotations.columns = ['ImageId', 'ImageEncoding']
annotations = annotations[annotations.iloc[:, 1] != "-1"]
image_ids = annotations.iloc[:, 0].values
len = len(image_ids)
rles = annotations.iloc[:, 1].values
index = np.random.randint(0,len-1)
img_id = image_ids[index]
encoding = rles[index]
image_path = os.path.join(dataset_dir, img_id + ".dcm")
image = pydicom.dcmread(image_path)
height = image.Rows
width = image.Columns
mask = rle2mask(encoding, width, height)
mask = mask.reshape(height, width, 1)


ax1.imshow(image.pixel_array)
rle_m1 = rle2mask(encoding, 1024, 1024)
ax1.imshow(rle_m1.T, alpha = 0.2)


plt.show()