import tensorflow as tf
from tensorflow import keras
from mrcnn.model import MaskRCNN
from mrcnn.config import Config

class SiimConfig(Config):
    NAME = "siim"

siimconfig = SiimConfig()

import sys, os
project_path = os.getcwd()
DEFAULT_LOGS_DIR = os.path.join(project_path, "logs")
model_path = os.path.join(project_path, "mask_rcnn_coco.h5")
siim_model = MaskRCNN(mode='training', config=siimconfig, model_dir=DEFAULT_LOGS_DIR)

siim_model.keras_model.load_weights(model_path)

