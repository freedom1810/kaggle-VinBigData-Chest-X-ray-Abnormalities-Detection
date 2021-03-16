import torch
import numpy as np
import json
import os
import cv2
import random
import glob
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import tqdm

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path, force=True)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

def save_image(dicom_name):
    folder_dicom = '/home/hana/sonnh/kaggle-vin/dataset/images_only/train'
    folder_save = '/home/hana/sonnh/kaggle-vin/dataset/images_only/train_jpg'
    # img = read_xray(os.path.join(folder_dicom, dicom_name), voi_lut = True, fix_monochrome = True)
    # print(os.path.join(os.path.join(folder_dicom, dicom_name + '.png')))
    img = cv2.imread(os.path.join(folder_dicom, dicom_name))
    cv2.imwrite('{}.jpg'.format(os.path.join(folder_save, dicom_name.split('.')[0])), img)
    return dicom_name, img.shape
    
import os
from multiprocessing import Pool

folder_dicom = '/home/hana/sonnh/kaggle-vin/dataset/images_only/train'
list_dicom_name = os.listdir(folder_dicom)
pool = Pool()
import time
tic = time.time()
data = pool.map(save_image, list_dicom_name)
print(time.time() - tic)