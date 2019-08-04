import pandas as pd
import numpy as np
import os, shutil
annotations = pd.read_csv("/home/sa-279/Mask_RCNN/datasets/pneumothorax/annotations.csv")
annotations.columns = ['ImageId', 'ImageEncoding']
annotations = annotations.sample(frac=1).reset_index(drop=True)
annotations = annotations[annotations.iloc[:, 1] != "-1"]

mask = np.random.rand(len(annotations)) < 0.9
train = annotations[mask]
val = annotations[~mask]
train.to_csv('/home/sa-279/Mask_RCNN/datasets/pneumothorax/train/train-rle.csv',header=False, index=False)
val.to_csv('/home/sa-279/Mask_RCNN/datasets/pneumothorax/val/val-rle.csv',header=False, index = False)

for row in val.itertuples():
    id = row.ImageId
    source_path = os.path.join("/home/sa-279/Mask_RCNN/datasets/pneumothorax/train/", id + ".dcm")
    dest_path =   os.path.join("/home/sa-279/Mask_RCNN/datasets/pneumothorax/val/", id + ".dcm")
    shutil.copy2(source_path, dest_path)

